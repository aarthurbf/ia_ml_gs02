"""Leitura, limpeza e agregação diária dos dados horários do INMET (estação A701).

Formato real verificado nos arquivos exportados da "Tabela de Dados das Estações"
(https://tempo.inmet.gov.br/TabelaEstacoes/A701):
  * codificação UTF-8 com BOM (não latin-1), sem linhas de metadados;
  * separador ';', decimal ',', valores entre aspas;
  * colunas 'Data' (dd/mm/aaaa) e 'Hora (UTC)' (HHMM);
  * radiação vazia no período noturno.
O leitor também aceita latin-1 / linhas de metadados (formato dos ZIPs anuais) como fallback.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TZ_LOCAL = "America/Sao_Paulo"
SENTINELAS = (-9999, -9999.0)
MIN_HORAS_VALIDAS = 18  # dia excluído se > 25% das 24 horas estiverem ausentes

COLMAP = {
    "Data": "data",
    "Hora (UTC)": "hora_utc",
    "Temp. Ins. (C)": "temp_ins",
    "Temp. Max. (C)": "temp_max",
    "Temp. Min. (C)": "temp_min",
    "Umi. Ins. (%)": "umi_ins",
    "Umi. Max. (%)": "umi_max",
    "Umi. Min. (%)": "umi_min",
    "Pto Orvalho Ins. (C)": "orv_ins",
    "Pto Orvalho Max. (C)": "orv_max",
    "Pto Orvalho Min. (C)": "orv_min",
    "Pressao Ins. (hPa)": "pres_ins",
    "Pressao Max. (hPa)": "pres_max",
    "Pressao Min. (hPa)": "pres_min",
    "Vel. Vento (m/s)": "vento_vel",
    "Dir. Vento (m/s)": "vento_dir",  # rótulo original do INMET; na prática é direção em graus
    "Raj. Vento (m/s)": "vento_raj",
    "Radiacao (KJ/m²)": "radiacao_kj",
    "Chuva (mm)": "chuva_mm",
}
MEDIDAS = [c for c in COLMAP.values() if c not in ("data", "hora_utc")]

# Faixas físicas plausíveis (valores fora são marcados como inválidos -> NaN)
FAIXAS = {
    "temp_ins": (-10, 50), "temp_max": (-10, 50), "temp_min": (-10, 50),
    "umi_ins": (0, 100), "umi_max": (0, 100), "umi_min": (0, 100),
    "orv_ins": (-40, 40), "orv_max": (-40, 40), "orv_min": (-40, 40),
    "pres_ins": (850, 1050), "pres_max": (850, 1050), "pres_min": (850, 1050),
    "vento_vel": (0, 60), "vento_dir": (0, 360), "vento_raj": (0, 80),
    "radiacao_kj": (0, 5000), "chuva_mm": (0, 200),
}


def _ler_csv(path: Path) -> pd.DataFrame:
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "latin-1"):
        try:
            texto = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    linhas = texto.splitlines()
    # pula eventuais linhas de metadados até encontrar o cabeçalho com "Data"
    ini = next(i for i, l in enumerate(linhas) if l.replace('"', "").startswith("Data"))
    df = pd.read_csv(io.StringIO("\n".join(linhas[ini:])), sep=";", decimal=",",
                     dtype={"Hora (UTC)": str}, na_values=["", "null"])
    return df


def carregar_brutos(paths: Iterable[Path]) -> pd.DataFrame:
    """Lê e concatena os CSVs, mantendo o nome do arquivo de origem."""
    dfs = []
    for p in sorted(paths):
        d = _ler_csv(Path(p))
        faltando = set(COLMAP) - set(d.columns)
        if faltando:
            raise ValueError(f"{Path(p).name}: colunas ausentes {faltando}")
        d = d.rename(columns=COLMAP)[list(COLMAP.values())]
        d["arquivo"] = Path(p).name
        dfs.append(d)
    if not dfs:
        raise FileNotFoundError("Nenhum CSV encontrado.")
    return pd.concat(dfs, ignore_index=True)


def limpar_horarios(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Converte tipos, trata sentinelas/faixas, remove duplicatas e converte UTC->local."""
    rel: dict = {"linhas_lidas": int(len(df))}
    d = df.copy()
    d["hora_utc"] = d["hora_utc"].astype(str).str.zfill(4)
    d["ts_utc"] = pd.to_datetime(d["data"] + " " + d["hora_utc"], format="%d/%m/%Y %H%M",
                                 errors="coerce").dt.tz_localize("UTC")
    rel["timestamps_invalidos"] = int(d["ts_utc"].isna().sum())
    d = d.dropna(subset=["ts_utc"])

    for c in MEDIDAS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    sent = d[MEDIDAS].isin(SENTINELAS)
    rel["valores_sentinela_-9999"] = int(sent.values.sum())
    d[MEDIDAS] = d[MEDIDAS].mask(sent)

    fora = {}
    for c, (lo, hi) in FAIXAS.items():
        m = d[c].notna() & ((d[c] < lo) | (d[c] > hi))
        fora[c] = int(m.sum())
        d.loc[m, c] = np.nan
    # consistência: mínima horária não pode superar a máxima horária
    inc = d["temp_min"] > d["temp_max"]
    fora["temp_min>temp_max"] = int(inc.sum())
    d.loc[inc, ["temp_min", "temp_max"]] = np.nan
    rel["valores_fora_faixa_fisica"] = fora

    # duplicatas (sobreposição de 1 dia entre arquivos semestrais)
    dup_ts = d.duplicated("ts_utc", keep="first")
    conflitantes = d[d.duplicated("ts_utc", keep=False)].groupby("ts_utc")[MEDIDAS] \
        .nunique(dropna=False).gt(1).any(axis=1).sum()
    rel["duplicatas_removidas"] = int(dup_ts.sum())
    rel["duplicatas_com_valores_conflitantes"] = int(conflitantes)
    d = d[~dup_ts].sort_values("ts_utc").reset_index(drop=True)

    d["ts_local"] = d["ts_utc"].dt.tz_convert(TZ_LOCAL)
    d["data_local"] = d["ts_local"].dt.tz_localize(None).dt.normalize()
    d["hora_local"] = d["ts_local"].dt.hour

    esperado = int((d["ts_utc"].max() - d["ts_utc"].min()) / pd.Timedelta("1h")) + 1
    rel["horas_esperadas"] = esperado
    rel["horas_unicas"] = int(len(d))
    rel["horas_sem_temperatura"] = int(d["temp_ins"].isna().sum())
    rel["horas_sem_nenhuma_medida"] = int(d[MEDIDAS].isna().all(axis=1).sum())
    rel["inicio_utc"] = str(d["ts_utc"].min())
    rel["fim_utc"] = str(d["ts_utc"].max())
    return d.drop(columns=["data", "hora_utc"]), rel


def agregar_diario(h: pd.DataFrame, min_horas: int = MIN_HORAS_VALIDAS) -> tuple[pd.DataFrame, dict]:
    """Agrega a série horária (horário local) em uma base diária.

    Nenhum valor é imputado. Dias com menos de `min_horas` horas válidas de temperatura
    são marcados como inválidos (dia_valido=False) e suas variáveis ficam NaN.
    """
    g = h.groupby("data_local")
    diario = pd.DataFrame({
        "n_horas_registro": g.size(),
        "n_horas_validas": g["temp_ins"].count(),
        "tmax": g["temp_max"].max(),
        "tmin": g["temp_min"].min(),
        "tmean": g["temp_ins"].mean(),
        "umi_min": g["umi_min"].min(),
        "umi_mean": g["umi_ins"].mean(),
        "chuva_mm": g["chuva_mm"].sum(min_count=1),
        "n_horas_chuva": g["chuva_mm"].count(),
        "radiacao_mj": g["radiacao_kj"].sum(min_count=1) / 1000.0,
        "vento_mean": g["vento_vel"].mean(),
        "pres_mean": g["pres_ins"].mean(),
    })
    # radiação: lacuna noturna é esperada; lacuna diurna (10h-15h local com temperatura
    # válida e radiação ausente) invalida o acumulado do dia.
    diurno = h[(h["hora_local"] >= 10) & (h["hora_local"] <= 15) & h["temp_ins"].notna()]
    lac_rad = diurno["radiacao_kj"].isna().groupby(diurno["data_local"]).sum()
    diario["lacunas_rad_diurnas"] = lac_rad.reindex(diario.index).fillna(0).astype(int)
    diario.loc[diario["lacunas_rad_diurnas"] > 0, "radiacao_mj"] = np.nan
    diario.loc[diario["n_horas_chuva"] < min_horas, "chuva_mm"] = np.nan

    idx = pd.date_range(diario.index.min(), diario.index.max(), freq="D", name="data")
    diario = diario.reindex(idx)
    diario["n_horas_registro"] = diario["n_horas_registro"].fillna(0).astype(int)
    diario["n_horas_validas"] = diario["n_horas_validas"].fillna(0).astype(int)
    diario["dia_valido"] = diario["n_horas_validas"] >= min_horas
    vars_ = ["tmax", "tmin", "tmean", "umi_min", "umi_mean", "chuva_mm",
             "radiacao_mj", "vento_mean", "pres_mean"]
    diario.loc[~diario["dia_valido"], vars_] = np.nan

    rel = {
        "dias_calendario": int(len(diario)),
        "dias_validos": int(diario["dia_valido"].sum()),
        "dias_excluidos_cobertura": int((~diario["dia_valido"]).sum()),
        "datas_excluidas": [d.strftime("%Y-%m-%d") for d in diario.index[~diario["dia_valido"]]],
        "dias_radiacao_invalida": int(diario.loc[diario["dia_valido"], "radiacao_mj"].isna().sum()),
        "dias_chuva_invalida": int(diario.loc[diario["dia_valido"], "chuva_mm"].isna().sum()),
    }
    return diario, rel


def processar(paths: Iterable[Path]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    bruto = carregar_brutos(paths)
    horario, r1 = limpar_horarios(bruto)
    diario, r2 = agregar_diario(horario)
    return horario, diario, {"horario": r1, "diario": r2}
