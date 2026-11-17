"""Engenharia de atributos e variável-alvo (sem vazamento temporal).

Todos os deslocamentos são feitos por DIA DE CALENDÁRIO (o índice diário é contínuo),
nunca por posição de linha. Assim, uma lacuna não faz o "ontem" virar outro dia.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "tmax_d0", "tmin_d0", "tmean_d0",
    "tmax_d1", "tmean_d1", "tmax_d2",
    "umi_min_d0", "umi_mean_d0",
    "chuva_d0", "rad_d0", "vento_d0", "pres_d0",
    "amplitude_d0", "mes_sin", "mes_cos",
]

DESCRICAO_FEATURES = {
    "tmax_d0": "Temperatura máxima do dia de referência (hoje), °C",
    "tmin_d0": "Temperatura mínima de hoje, °C",
    "tmean_d0": "Temperatura média horária de hoje, °C",
    "tmax_d1": "Temperatura máxima de ontem (D-1), °C",
    "tmean_d1": "Temperatura média de ontem (D-1), °C",
    "tmax_d2": "Temperatura máxima de anteontem (D-2), °C",
    "umi_min_d0": "Umidade relativa mínima de hoje, %",
    "umi_mean_d0": "Umidade relativa média de hoje, %",
    "chuva_d0": "Precipitação acumulada hoje, mm",
    "rad_d0": "Radiação solar global acumulada hoje, MJ/m²",
    "vento_d0": "Velocidade média do vento hoje, m/s",
    "pres_d0": "Pressão atmosférica média hoje (nível da estação), hPa",
    "amplitude_d0": "Amplitude térmica de hoje (tmax_d0 - tmin_d0), °C",
    "mes_sin": "sen(2π·mês/12) do dia de referência",
    "mes_cos": "cos(2π·mês/12) do dia de referência",
}


def _garantir_continuo(diario: pd.DataFrame) -> pd.DataFrame:
    idx = pd.date_range(diario.index.min(), diario.index.max(), freq="D", name="data")
    return diario.reindex(idx)


def construir_atributos(diario: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por dia de referência D (hoje). Usa apenas D, D-1 e D-2."""
    d = _garantir_continuo(diario)
    f = pd.DataFrame(index=d.index)
    f["tmax_d0"] = d["tmax"]
    f["tmin_d0"] = d["tmin"]
    f["tmean_d0"] = d["tmean"]
    f["tmax_d1"] = d["tmax"].shift(1, freq="D").reindex(d.index)
    f["tmean_d1"] = d["tmean"].shift(1, freq="D").reindex(d.index)
    f["tmax_d2"] = d["tmax"].shift(2, freq="D").reindex(d.index)
    f["umi_min_d0"] = d["umi_min"]
    f["umi_mean_d0"] = d["umi_mean"]
    f["chuva_d0"] = d["chuva_mm"]
    f["rad_d0"] = d["radiacao_mj"]
    f["vento_d0"] = d["vento_mean"]
    f["pres_d0"] = d["pres_mean"]
    f["amplitude_d0"] = d["tmax"] - d["tmin"]
    mes = d.index.month
    f["mes_sin"] = np.sin(2 * np.pi * mes / 12)
    f["mes_cos"] = np.cos(2 * np.pi * mes / 12)
    f["data_prevista"] = f.index + pd.Timedelta(days=1)
    return f


def tmax_amanha(diario: pd.DataFrame) -> pd.Series:
    """Tmax observada em D+1, alinhada à linha do dia D (usada SOMENTE para o alvo)."""
    d = _garantir_continuo(diario)
    return d["tmax"].shift(-1, freq="D").reindex(d.index)


def limiar_p90(tmax_treino: pd.Series) -> float:
    """Percentil 90 das Tmax diárias válidas do período de TREINO."""
    s = tmax_treino.dropna()
    if s.empty:
        raise ValueError("Sem Tmax válidas para calcular o limiar.")
    return float(np.percentile(s, 90))


def construir_alvo(diario: pd.DataFrame, limiar: float) -> pd.Series:
    """extremo_amanha = 1 se Tmax(D+1) >= limiar; NaN se D+1 não for um dia válido."""
    t1 = tmax_amanha(diario)
    alvo = (t1 >= limiar).astype(float)
    alvo[t1.isna()] = np.nan
    return alvo.rename("extremo_amanha")
