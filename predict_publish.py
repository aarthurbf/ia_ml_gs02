#!/usr/bin/env python
"""Sereno SP — gera a previsão de calor extremo para D+1 e publica em POST /api/previsoes.

Modos de entrada (escolha um):
  --replay AAAA-MM-DD   Demonstração RETROSPECTIVA: usa dados já observados do INMET (data/processed)
                        para reproduzir o que teria sido publicado na véspera. NÃO é previsão operacional.
  --input  arquivo.json Operacional: observações diárias de hoje, ontem e anteontem (ver README).
  --csv    arq1.csv ... Operacional: CSV(s) exportados da Tabela de Dados das Estações do INMET (A701)
                        cobrindo pelo menos os 3 últimos dias completos.

Configuração por variáveis de ambiente:
  SERENO_API_URL      URL base do backend .NET (ex.: https://localhost:7001). Obrigatória sem --dry-run.
  SERENO_API_TOKEN    Token Bearer (opcional).
  SERENO_API_TIMEOUT  Timeout em segundos (padrão 10).
  SERENO_MODEL_PATH   Caminho do modelo (padrão models/sereno_rf.joblib).

Códigos de saída: 0 sucesso · 2 entrada inválida · 3 falha ao publicar · 4 modelo indisponível.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from sereno_ml.features import FEATURES, construir_atributos  # noqa: E402

log = logging.getLogger("sereno.predict_publish")
ENDPOINT = "/api/previsoes"

# Faixas aceitas na entrada operacional (mesmas da limpeza horária)
FAIXAS_ENTRADA = {
    "tmax": (-10, 50), "tmin": (-10, 50), "tmean": (-10, 50),
    "umi_min": (0, 100), "umi_mean": (0, 100), "chuva_mm": (0, 500),
    "radiacao_mj": (0, 45), "vento_mean": (0, 60), "pres_mean": (850, 1050),
}
CAMPOS_HOJE = list(FAIXAS_ENTRADA)
CAMPOS_ONTEM = ["tmax", "tmean"]
CAMPOS_ANTEONTEM = ["tmax"]


class EntradaInvalida(ValueError):
    pass


# --------------------------------------------------------------------------- modelo
def carregar_modelo(path: str | Path | None = None) -> dict:
    p = Path(path or os.getenv("SERENO_MODEL_PATH", BASE / "models" / "sereno_rf.joblib"))
    if not p.exists():
        raise FileNotFoundError(f"Modelo não encontrado em {p}. Execute o notebook para treiná-lo.")
    b = joblib.load(p)
    for k in ("modelo", "features", "limiar_decisao", "limiar_calor_c", "versao"):
        if k not in b:
            raise ValueError(f"Pacote do modelo incompleto: falta '{k}'.")
    if list(b["features"]) != FEATURES:
        raise ValueError("Lista de atributos do modelo difere da versão do código.")
    return b


def nivel(prob: float) -> str:
    if prob > 0.70:
        return "Perigo"
    if prob >= 0.40:
        return "Atencao"
    return "SemAlerta"


# --------------------------------------------------------------------------- entradas
def _validar_dia(nome: str, dia: dict, campos: list[str]) -> None:
    if not isinstance(dia, dict):
        raise EntradaInvalida(f"'{nome}' deve ser um objeto JSON.")
    for c in campos:
        v = dia.get(c)
        if v is None or isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise EntradaInvalida(f"'{nome}.{c}' ausente ou não numérico.")
        lo, hi = FAIXAS_ENTRADA[c]
        if not lo <= v <= hi:
            raise EntradaInvalida(f"'{nome}.{c}'={v} fora da faixa física [{lo}, {hi}].")
    if "tmin" in campos and dia["tmin"] > dia["tmax"]:
        raise EntradaInvalida(f"'{nome}': tmin > tmax.")


def atributos_de_json(payload: dict) -> tuple[pd.DataFrame, date]:
    """Converte a entrada operacional {dataBase, hoje, ontem, anteontem} em atributos."""
    try:
        d0 = date.fromisoformat(str(payload["dataBase"]))
    except (KeyError, ValueError) as e:
        raise EntradaInvalida("'dataBase' ausente ou fora do formato AAAA-MM-DD.") from e
    hoje, ontem, ante = payload.get("hoje"), payload.get("ontem"), payload.get("anteontem")
    _validar_dia("hoje", hoje, CAMPOS_HOJE)
    _validar_dia("ontem", ontem, CAMPOS_ONTEM)
    _validar_dia("anteontem", ante, CAMPOS_ANTEONTEM)
    diario = pd.DataFrame(
        [{"tmax": ante["tmax"]}, {"tmax": ontem["tmax"], "tmean": ontem["tmean"]}, {**hoje}],
        index=pd.DatetimeIndex([pd.Timestamp(d0 - timedelta(days=k)) for k in (2, 1, 0)], name="data"),
    ).reindex(columns=CAMPOS_HOJE)
    return construir_atributos(diario).loc[[pd.Timestamp(d0)], FEATURES], d0


def atributos_de_diario(diario: pd.DataFrame, d0: date) -> pd.DataFrame:
    ts = pd.Timestamp(d0)
    if ts not in diario.index:
        raise EntradaInvalida(f"Data {d0} fora da base diária disponível.")
    x = construir_atributos(diario).loc[[ts], FEATURES]
    faltando = [c for c in FEATURES if pd.isna(x.iloc[0][c])]
    if faltando:
        raise EntradaInvalida(f"Dados insuficientes para {d0} (D, D-1 ou D-2 inválido): {faltando}")
    return x


def carregar_diario_processado() -> pd.DataFrame:
    p = BASE / "data" / "processed" / "diario_a701.csv"
    if p.exists():
        return pd.read_csv(p, index_col="data", parse_dates=["data"])
    log.info("Base diária processada ausente; reconstruindo a partir de data/raw.")
    from sereno_ml.data import processar
    _, diario, _ = processar(sorted((BASE / "data" / "raw").glob("*.csv")))
    return diario


def diario_de_csv(paths: list[str]) -> tuple[pd.DataFrame, date]:
    from sereno_ml.data import processar
    _, diario, _ = processar([Path(p) for p in paths])
    validos = diario.index[diario["dia_valido"]]
    if validos.empty:
        raise EntradaInvalida("Nenhum dia completo (>= 18 h válidas) nos CSVs informados.")
    return diario, validos.max().date()


# --------------------------------------------------------------------------- previsão
def prever(bundle: dict, X: pd.DataFrame, d0: date, modo: str) -> dict:
    prob = float(bundle["modelo"].predict_proba(X[bundle["features"]])[:, 1][0])
    prob = round(prob, 4)
    payload = {
        "dataReferencia": (d0 + timedelta(days=1)).isoformat(),  # data prevista (D+1)
        "dataBase": d0.isoformat(),                                # último dia observado (D)
        "probabilidade": prob,
        "nivel": nivel(prob),
        "alertaExtremo": bool(prob >= bundle["limiar_decisao"]),
        "limiarCalorC": round(float(bundle["limiar_calor_c"]), 1),
        "versaoModelo": bundle["versao"],
        "estacao": bundle.get("estacao", "A701"),
        "modo": modo,
        "geradoEm": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if modo == "replay":
        payload["observacao"] = ("Demonstração retrospectiva com dados históricos do INMET; "
                                 "não é previsão meteorológica operacional.")
    return payload


def validar_payload(p: dict) -> None:
    obrig = {"dataReferencia": str, "probabilidade": float, "nivel": str, "versaoModelo": str}
    for k, t in obrig.items():
        if not isinstance(p.get(k), t):
            raise ValueError(f"Payload inválido: campo '{k}'.")
    if not 0.0 <= p["probabilidade"] <= 1.0:
        raise ValueError("Payload inválido: probabilidade fora de [0, 1].")
    if p["nivel"] not in {"Perigo", "Atencao", "SemAlerta"}:
        raise ValueError("Payload inválido: nível desconhecido.")
    date.fromisoformat(p["dataReferencia"])


# --------------------------------------------------------------------------- publicação
def publicar(payload: dict, url_base: str | None = None, token: str | None = None,
             timeout: float | None = None, tentativas: int = 3) -> dict:
    import requests

    url_base = url_base or os.getenv("SERENO_API_URL")
    if not url_base:
        raise RuntimeError("SERENO_API_URL não definida (use --dry-run para apenas gerar o JSON).")
    token = token if token is not None else os.getenv("SERENO_API_TOKEN")
    timeout = timeout or float(os.getenv("SERENO_API_TIMEOUT", "10"))
    url = url_base.rstrip("/") + ENDPOINT
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    ultimo_erro = None
    for k in range(1, tentativas + 1):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if r.status_code < 300:
                log.info("Publicado em %s (HTTP %s).", url, r.status_code)
                try:
                    return {"status": r.status_code, "resposta": r.json()}
                except ValueError:
                    return {"status": r.status_code, "resposta": r.text}
            if 400 <= r.status_code < 500:  # erro do cliente: não adianta repetir
                raise RuntimeError(f"API recusou a previsão (HTTP {r.status_code}): {r.text[:300]}")
            ultimo_erro = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            ultimo_erro = f"{type(e).__name__}: {e}"
        log.warning("Tentativa %d/%d falhou: %s", k, tentativas, ultimo_erro)
        if k < tentativas:
            time.sleep(min(2 ** (k - 1), 8))
    raise RuntimeError(f"Falha ao publicar após {tentativas} tentativas: {ultimo_erro}")


# --------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--replay", metavar="AAAA-MM-DD", help="dia base D (prevê D+1) — modo retrospectivo")
    g.add_argument("--input", metavar="JSON", help="arquivo JSON com observações de hoje/ontem/anteontem")
    g.add_argument("--csv", nargs="+", metavar="CSV", help="CSV(s) da Tabela de Dados das Estações (INMET)")
    ap.add_argument("--dry-run", action="store_true", help="apenas imprime o JSON, sem enviar")
    ap.add_argument("--modelo", help="caminho alternativo do modelo .joblib")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    try:
        bundle = carregar_modelo(a.modelo)
    except Exception as e:
        log.error("%s", e)
        return 4
    try:
        if a.replay:
            d0 = date.fromisoformat(a.replay)
            diario = carregar_diario_processado()
            X = atributos_de_diario(diario, d0)
            modo = "replay"
            ini, fim = bundle["teste"]["inicio"], bundle["teste"]["fim_ref"]
            if not ini <= d0.isoformat() <= fim:
                log.warning("Data fora do período de teste (%s a %s): resultado é in-sample, "
                            "não serve como avaliação.", ini, fim)
            obs = diario["tmax"].get(pd.Timestamp(d0 + timedelta(days=1)))
            log.info("[REPLAY] Tmax observada em %s: %s °C (limiar %.1f °C).", d0 + timedelta(days=1),
                     "indisponível" if obs is None or pd.isna(obs) else f"{obs:.1f}", bundle["limiar_calor_c"])
        elif a.input:
            X, d0 = atributos_de_json(json.loads(Path(a.input).read_text(encoding="utf-8")))
            modo = "operacional"
        else:
            diario, d0 = diario_de_csv(a.csv)
            X = atributos_de_diario(diario, d0)
            modo = "operacional"
    except (EntradaInvalida, ValueError, FileNotFoundError, json.JSONDecodeError) as e:
        log.error("Entrada inválida: %s", e)
        return 2

    payload = prever(bundle, X, d0, modo)
    validar_payload(payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if a.dry_run:
        log.info("--dry-run: nada foi enviado.")
        return 0
    try:
        publicar(payload)
    except Exception as e:
        log.error("%s", e)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
