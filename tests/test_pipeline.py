"""Testes automatizados — processamento, atributos, alvo, previsão e publicação."""
import json
import sys
import threading
from datetime import date
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML))

from sereno_ml.data import agregar_diario, carregar_brutos, limpar_horarios  # noqa: E402
from sereno_ml.features import FEATURES, construir_alvo, construir_atributos, limiar_p90  # noqa: E402
from sereno_ml.modeling import TESTE_INI, TREINO_FIM, montar_conjuntos  # noqa: E402
import predict_publish as pp  # noqa: E402

HEADER = ('"Data";"Hora (UTC)";"Temp. Ins. (C)";"Temp. Max. (C)";"Temp. Min. (C)";"Umi. Ins. (%)";'
          '"Umi. Max. (%)";"Umi. Min. (%)";"Pto Orvalho Ins. (C)";"Pto Orvalho Max. (C)";"Pto Orvalho Min. (C)";'
          '"Pressao Ins. (hPa)";"Pressao Max. (hPa)";"Pressao Min. (hPa)";"Vel. Vento (m/s)";"Dir. Vento (m/s)";'
          '"Raj. Vento (m/s)";"Radiacao (KJ/m²)";"Chuva (mm)"')


def _linha(data, hora, temp="20,0", rad="", chuva="0,0"):
    return (f'"{data}";"{hora}";"{temp}";"{temp}";"{temp}";"70,0";"72,0";"68,0";"14,0";"14,1";"13,9";'
            f'"925,0";"925,1";"924,9";"1,5";"180,0";"4,0";"{rad}";"{chuva}"')


@pytest.fixture
def csv_sintetico(tmp_path):
    """CSV no formato real do INMET (UTF-8 BOM), 3 dias UTC completos (estrutura de teste, não dado meteorológico)."""
    linhas = [HEADER]
    for dia in ["01/01/2024", "02/01/2024", "03/01/2024"]:
        for h in range(24):
            rad = "1500,0" if 12 <= h <= 21 else ""
            linhas.append(_linha(dia, f"{h:02d}00", rad=rad))
    linhas[5] = _linha("01/01/2024", "0400", temp="-9999")           # sentinela
    linhas.append(_linha("03/01/2024", "2300"))                       # duplicata
    p = tmp_path / "a701.csv"
    p.write_bytes(("\n".join(linhas) + "\n").encode("utf-8-sig"))
    return p


# ------------------------------------------------------------------ processamento
def test_leitura_formato_real(csv_sintetico):
    d = carregar_brutos([csv_sintetico])
    assert d.shape[0] == 73 and d["temp_ins"].dtype == float and d["hora_utc"].iloc[0] == "0000"


def test_leitura_latin1_com_metadados(tmp_path):
    p = tmp_path / "zip_anual.csv"
    txt = "REGIAO:;SE\nESTACAO:;SAO PAULO - MIRANTE\n" + HEADER + "\n" + _linha("01/01/2024", "0000") + "\n"
    p.write_bytes(txt.encode("latin-1"))
    assert len(carregar_brutos([p])) == 1


def test_limpeza_sentinela_duplicata_fuso(csv_sintetico):
    h, rel = limpar_horarios(carregar_brutos([csv_sintetico]))
    assert rel["valores_sentinela_-9999"] == 3  # temp ins/max/min da linha marcada
    assert rel["duplicatas_removidas"] == 1 and len(h) == 72
    assert str(h["ts_local"].dt.tz) == "America/Sao_Paulo"
    primeira = h.iloc[0]
    assert primeira["ts_local"].hour == 21 and primeira["data_local"] == pd.Timestamp("2023-12-31")


def test_faixa_fisica_invalida(csv_sintetico, tmp_path):
    txt = csv_sintetico.read_bytes().decode("utf-8-sig").replace('"03/01/2024";"1000";"20,0"', '"03/01/2024";"1000";"75,0"')
    p = tmp_path / "x.csv"; p.write_bytes(txt.encode("utf-8-sig"))
    _, rel = limpar_horarios(carregar_brutos([p]))
    assert rel["valores_fora_faixa_fisica"]["temp_ins"] == 1


def test_agregacao_exclui_dias_incompletos(csv_sintetico):
    h, _ = limpar_horarios(carregar_brutos([csv_sintetico]))
    d, rel = agregar_diario(h)
    # 31/12 local só tem 3 h (21h-23h) -> excluído; 03/01 local tem 21 h -> válido
    assert not d.loc["2023-12-31", "dia_valido"] and np.isnan(d.loc["2023-12-31", "tmax"])
    assert d.loc["2024-01-01", "dia_valido"] and d.loc["2024-01-01", "n_horas_validas"] == 23
    assert d.loc["2024-01-02", "radiacao_mj"] == pytest.approx(15.0)  # 10 h x 1500 kJ/m²
    assert rel["dias_excluidos_cobertura"] == 1


# ------------------------------------------------------------------ atributos e alvo
def _diario_fake(n=30, ini="2024-01-01"):
    idx = pd.date_range(ini, periods=n, freq="D", name="data")
    r = np.random.default_rng(0)
    return pd.DataFrame({"tmax": r.uniform(20, 35, n), "tmin": r.uniform(10, 19, n), "tmean": r.uniform(18, 26, n),
                         "umi_min": r.uniform(20, 60, n), "umi_mean": r.uniform(50, 90, n), "chuva_mm": 0.0,
                         "radiacao_mj": 15.0, "vento_mean": 1.5, "pres_mean": 925.0, "dia_valido": True}, index=idx)


def test_atributos_sem_vazamento():
    d = _diario_fake()
    D = pd.Timestamp("2024-01-15")
    f1 = construir_atributos(d).loc[D, FEATURES]
    d2 = d.copy(); d2.loc[D + pd.Timedelta(days=1):, ["tmax", "tmin", "tmean"]] += 100
    pd.testing.assert_series_equal(f1, construir_atributos(d2).loc[D, FEATURES])


def test_lags_por_calendario_com_lacuna():
    d = _diario_fake().drop(pd.Timestamp("2024-01-14"))  # lacuna real
    f = construir_atributos(d)
    assert np.isnan(f.loc["2024-01-15", "tmax_d1"])           # não "pula" para 13/01
    assert f.loc["2024-01-15", "tmax_d2"] == d.loc["2024-01-13", "tmax"]


def test_alvo_e_limiar():
    d = _diario_fake()
    lim = limiar_p90(d["tmax"])
    y = construir_alvo(d, lim)
    assert y.loc["2024-01-05"] == float(d.loc["2024-01-06", "tmax"] >= lim)
    assert np.isnan(y.iloc[-1])  # não existe D+1


def test_split_real_sem_sobreposicao():
    p = ML / "data" / "processed" / "diario_a701.csv"
    if not p.exists():
        pytest.skip("execute o notebook antes")
    d = pd.read_csv(p, index_col="data", parse_dates=["data"])
    Xtr, ytr, Xte, yte, meta, tr, te = montar_conjuntos(d)
    assert tr["data_prevista"].max() <= TREINO_FIM < TESTE_INI <= Xte.index.min()
    assert meta["limiar_p90_c"] == pytest.approx(np.percentile(d.loc[:TREINO_FIM, "tmax"].dropna(), 90))
    assert not Xtr.isna().any().any() and not Xte.isna().any().any()


# ------------------------------------------------------------------ previsão e integração
ENTRADA = {"dataBase": "2026-01-20",
           "hoje": {"tmax": 33.5, "tmin": 20.1, "tmean": 26.0, "umi_min": 30, "umi_mean": 60, "chuva_mm": 0,
                    "radiacao_mj": 24.0, "vento_mean": 1.6, "pres_mean": 924.0},
           "ontem": {"tmax": 32.8, "tmean": 25.4}, "anteontem": {"tmax": 31.9}}


@pytest.fixture(scope="module")
def bundle():
    try:
        return pp.carregar_modelo()
    except FileNotFoundError:
        pytest.skip("modelo ainda não treinado")


def test_previsao_json(bundle):
    X, d0 = pp.atributos_de_json(ENTRADA)
    assert list(X.columns) == FEATURES and X.notna().all().all()
    p = pp.prever(bundle, X, d0, "operacional")
    pp.validar_payload(p)
    assert p["dataReferencia"] == "2026-01-21" and 0 <= p["probabilidade"] <= 1
    assert p["nivel"] == pp.nivel(p["probabilidade"])
    json.dumps(p)


@pytest.mark.parametrize("campo,valor", [("tmax", 80), ("umi_min", -1), ("tmax", None), ("tmax", "quente")])
def test_validacao_entrada(campo, valor):
    e = json.loads(json.dumps(ENTRADA)); e["hoje"][campo] = valor
    with pytest.raises(pp.EntradaInvalida):
        pp.atributos_de_json(e)


def test_niveis():
    assert pp.nivel(0.71) == "Perigo" and pp.nivel(0.70) == "Atencao"
    assert pp.nivel(0.40) == "Atencao" and pp.nivel(0.39) == "SemAlerta"


def test_replay_cli(bundle, capsys):
    assert pp.main(["--replay", "2025-12-24", "--dry-run"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["modo"] == "replay" and "retrospectiva" in out["observacao"]


def test_replay_data_invalida(bundle):
    assert pp.main(["--replay", "2019-01-01", "--dry-run"]) == 2


def test_modelo_padrao_existencia():
    p = ML / "models" / "sereno_rf.joblib"
    assert p.exists()
    b = pp.carregar_modelo(p)
    assert b["versao"] == "sereno-rf-1.0"
    assert len(b["features"]) == len(FEATURES)


def test_data_readme_existencia():
    p = ML / "data" / "README.md"
    assert p.exists()
    conteudo = p.read_text(encoding="utf-8")
    assert "INMET" in conteudo
    assert "A701" in conteudo
    assert len(conteudo) > 200


def test_valores_aceitos_nivel():
    niveis_validos = {"Perigo", "Atencao", "SemAlerta"}
    for prob in [0.0, 0.1, 0.39, 0.40, 0.55, 0.70, 0.71, 0.95, 1.0]:
        n = pp.nivel(prob)
        assert n in niveis_validos


class _Mock(BaseHTTPRequestHandler):
    recebidos, status = [], [201]

    def do_POST(self):
        corpo = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Mock.recebidos.append((self.path, self.headers.get("Authorization"), corpo))
        s = _Mock.status.pop(0) if len(_Mock.status) > 1 else _Mock.status[0]
        self.send_response(s); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(b'{"id": 1}')

    def log_message(self, *a):
        pass


@pytest.fixture
def api_mock():
    srv = HTTPServer(("127.0.0.1", 0), _Mock)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _Mock.recebidos.clear()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_publicacao_com_retry(bundle, api_mock):
    _Mock.status[:] = [503, 201]
    X, d0 = pp.atributos_de_json(ENTRADA)
    p = pp.prever(bundle, X, d0, "operacional")
    r = pp.publicar(p, url_base=api_mock, token="abc", tentativas=3)
    assert r["status"] == 201 and len(_Mock.recebidos) == 2
    path, auth, corpo = _Mock.recebidos[-1]
    assert path == "/api/previsoes" and auth == "Bearer abc" and corpo["dataReferencia"] == "2026-01-21"


def test_publicacao_erro_cliente_sem_retry(bundle, api_mock):
    _Mock.status[:] = [400]
    X, d0 = pp.atributos_de_json(ENTRADA)
    with pytest.raises(RuntimeError, match="HTTP 400"):
        pp.publicar(pp.prever(bundle, X, d0, "operacional"), url_base=api_mock, tentativas=3)
    assert len(_Mock.recebidos) == 1


def test_cli_publica_via_env(bundle, api_mock, monkeypatch, capsys):
    _Mock.status[:] = [201]
    monkeypatch.setenv("SERENO_API_URL", api_mock)
    assert pp.main(["--replay", "2025-12-24"]) == 0
    assert _Mock.recebidos[-1][2]["modo"] == "replay"
