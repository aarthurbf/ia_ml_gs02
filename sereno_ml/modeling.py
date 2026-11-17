"""Divisão temporal, baselines, métricas e utilidades de modelagem."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             confusion_matrix)
from sklearn.model_selection import TimeSeriesSplit

from .features import FEATURES, construir_alvo, construir_atributos, limiar_p90

SEED = 42
# Períodos (datas locais, inclusivas). Ver justificativa no RELATORIO_ML.md.
TREINO_INI, TREINO_FIM = pd.Timestamp("2023-05-17"), pd.Timestamp("2025-11-16")
TESTE_INI, TESTE_FIM = pd.Timestamp("2025-11-17"), pd.Timestamp("2026-11-16")

PARAM_GRID = {
    "n_estimators": [100, 300],
    "max_depth": [5, 10, None],
    "min_samples_leaf": [1, 5],
}


def novo_rf(**kw) -> RandomForestClassifier:
    return RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1, **kw)


def montar_conjuntos(diario: pd.DataFrame):
    """Retorna (X_tr, y_tr, X_te, y_te, meta) respeitando a ordem cronológica.

    * O limiar P90 é calculado apenas com Tmax de dias do TREINO.
    * Uma linha só entra em um conjunto se o dia de referência E o dia previsto
      (D+1) pertencem ao mesmo período -> nenhum alvo de treino usa dado do teste.
    * Linhas com qualquer atributo ausente ou alvo ausente são descartadas (sem imputação).
    """
    tmax_tr = diario.loc[TREINO_INI:TREINO_FIM, "tmax"]
    limiar = limiar_p90(tmax_tr)
    X = construir_atributos(diario)
    y = construir_alvo(diario, limiar)
    base = X.join(y)
    base["tmax_d0_extremo"] = (base["tmax_d0"] >= limiar).astype(int)

    def recorte(ini, fim):
        b = base[(base.index >= ini) & (base["data_prevista"] <= fim)]
        n0 = len(b)
        b = b.dropna(subset=FEATURES + ["extremo_amanha"])
        return b, n0 - len(b)

    tr, desc_tr = recorte(TREINO_INI, TREINO_FIM)
    te, desc_te = recorte(TESTE_INI, TESTE_FIM)
    meta = {
        "limiar_p90_c": limiar,
        "n_dias_tmax_para_limiar": int(tmax_tr.notna().sum()),
        "treino": {"inicio": str(tr.index.min().date()), "fim_ref": str(tr.index.max().date()),
                   "fim_previsto": str(tr["data_prevista"].max().date()), "n": int(len(tr)),
                   "positivos": int(tr["extremo_amanha"].sum()), "descartadas": int(desc_tr)},
        "teste": {"inicio": str(te.index.min().date()), "fim_ref": str(te.index.max().date()),
                  "fim_previsto": str(te["data_prevista"].max().date()), "n": int(len(te)),
                  "positivos": int(te["extremo_amanha"].sum()), "descartadas": int(desc_te)},
    }
    return (tr[FEATURES], tr["extremo_amanha"].astype(int),
            te[FEATURES], te["extremo_amanha"].astype(int), meta, tr, te)


def metricas(y_true, y_pred, score=None) -> dict:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    s = y_pred if score is None else score
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = roc_auc_score(y_true, s)
        out["pr_auc"] = average_precision_score(y_true, s)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in out.items()}


def baseline_persistencia(df: pd.DataFrame) -> np.ndarray:
    """Prevê 1 amanhã se HOJE já foi extremo (tmax_d0 >= limiar do treino)."""
    return df["tmax_d0_extremo"].to_numpy()


def baseline_climatologico(y_treino: pd.Series, n: int) -> np.ndarray:
    """Prevê sempre a classe mais frequente do treino."""
    return np.full(n, int(y_treino.mode().iloc[0]))


def probas_oof(params: dict, X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
    """Probabilidades fora-da-amostra nas dobras de validação do TimeSeriesSplit (só treino)."""
    p = pd.Series(np.nan, index=X.index)
    for tr_i, va_i in TimeSeriesSplit(n_splits=n_splits).split(X):
        m = novo_rf(**params).fit(X.iloc[tr_i], y.iloc[tr_i])
        p.iloc[va_i] = m.predict_proba(X.iloc[va_i])[:, 1]
    return p.dropna()


def nivel_alerta(prob: float, alto: float = 0.70, medio: float = 0.40) -> str:
    if prob > alto:
        return "Perigo"
    if prob >= medio:
        return "Atencao"
    return "SemAlerta"
