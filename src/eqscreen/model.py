"""Stretch goal: learn factor weights instead of fixing them, and test that honestly.

Cross-validation splits on DATES, not rows, with a gap equal to the label
horizon: a 12-month forward label at date t overlaps every date up to t+12, so
without the gap the last training labels would contain test-period returns.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .factors import RAW_FACTORS

FEATURES = [f"z_{c}" for c in RAW_FACTORS]


def build_dataset(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    fwd = f"ret_fwd_{horizon}m"
    df = panel[panel["composite"].notna() & panel[fwd].notna()].copy()
    df[FEATURES] = df[FEATURES].fillna(0.0)    # z-scores: 0 = cross-sectional average
    df["label"] = (df[fwd] > df.groupby("month_end")[fwd].transform("median")).astype(int)
    # Winsorise the regression target per date so a few extreme returns do not dominate.
    lo = df.groupby("month_end")[fwd].transform(lambda s: s.quantile(0.01))
    hi = df.groupby("month_end")[fwd].transform(lambda s: s.quantile(0.99))
    df["target"] = df[fwd].clip(lo, hi)
    return df


def same_cohort_panel(panel: pd.DataFrame, oos: pd.DataFrame) -> pd.DataFrame:
    """Attach out-of-sample model scores and a composite restricted to exactly the same stock-months.

    The model only scores rows with a 12-month label, so comparing it with the composite
    over all scored stocks would compare different universes.
    """
    o = oos[["month_end", "ticker", "ml_logit"]].rename(columns={"ml_logit": "ml_logit_oos"})
    p = panel.merge(o, on=["month_end", "ticker"], how="left")
    p["composite_same_cohort"] = p["composite"].where(p["ml_logit_oos"].notna())
    return p


def date_folds(dates: np.ndarray, n_splits: int, gap: int, min_train: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """TimeSeriesSplit over unique dates; reduces n_splits until each train fold has min_train dates."""
    for k in range(n_splits, 1, -1):
        if len(dates) < k + 1 + gap:
            continue
        try:
            folds = list(TimeSeriesSplit(n_splits=k, gap=gap).split(dates))
        except ValueError:      # not enough dates for k folds with this gap
            continue
        if all(len(tr) >= min_train for tr, _ in folds):
            return [(dates[tr], dates[te]) for tr, te in folds]
    return []


def _mean_ic(df: pd.DataFrame, score: str, target: str) -> float:
    ics = [g[score].rank().corr(g[target].rank()) for _, g in df.groupby("month_end") if len(g) >= 20]
    return float(np.nanmean(ics)) if ics else np.nan


def cross_validate(panel: pd.DataFrame, cfg: dict, logger: logging.Logger) -> dict:
    mcfg = cfg["model"]
    h = int(mcfg["label_horizon_months"])
    data = build_dataset(panel, h)
    dates = np.array(sorted(data["month_end"].unique()))
    folds = date_folds(dates, int(mcfg["n_splits"]), int(mcfg["gap_months"]), int(mcfg["min_train_dates"]))
    if not folds:
        msg = (f"Only {len(dates)} months have a {h}-month label; need at least "
               f"{int(mcfg['min_train_dates']) + int(mcfg['gap_months']) + 2} for purged CV. Skipping model stage.")
        logger.warning(msg)
        return {"status": "skipped", "message": msg}

    fwd = f"ret_fwd_{h}m"
    rows, oos = [], []
    for i, (tr_dates, te_dates) in enumerate(folds, start=1):
        tr = data[data["month_end"].isin(tr_dates)]
        te = data[data["month_end"].isin(te_dates)].copy()
        # Guard: the last training label must end before the first test date.
        if not tr["month_end"].max() + pd.offsets.MonthEnd(h) < te["month_end"].min() + pd.offsets.MonthEnd(1):
            raise RuntimeError("label overlap between training and test months (purge gap too small)")
        logit = make_pipeline(StandardScaler(), LogisticRegression(C=float(mcfg["logistic_C"]), max_iter=1000))
        logit.fit(tr[FEATURES], tr["label"])
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=float(mcfg["ridge_alpha"])))
        ridge.fit(tr[FEATURES], tr["target"])
        te["ml_logit"] = logit.predict_proba(te[FEATURES])[:, 1]
        te["ml_ridge"] = ridge.predict(te[FEATURES])
        both = te["label"].nunique() == 2
        rows.append({
            "fold": i,
            "train_start": tr["month_end"].min().strftime("%Y-%m"), "train_end": tr["month_end"].max().strftime("%Y-%m"),
            "test_start": te["month_end"].min().strftime("%Y-%m"), "test_end": te["month_end"].max().strftime("%Y-%m"),
            "n_train": len(tr), "n_test": len(te),
            "auc_logit": roc_auc_score(te["label"], te["ml_logit"]) if both else np.nan,
            "auc_composite": roc_auc_score(te["label"], te["composite"]) if both else np.nan,
            "ic_logit": _mean_ic(te, "ml_logit", fwd),
            "ic_ridge": _mean_ic(te, "ml_ridge", fwd),
            "ic_composite": _mean_ic(te, "composite", fwd),
        })
        oos.append(te[["month_end", "ticker", "ml_logit", "ml_ridge"]])

    cv = pd.DataFrame(rows)
    final = make_pipeline(StandardScaler(), LogisticRegression(C=float(mcfg["logistic_C"]), max_iter=1000))
    final.fit(data[FEATURES], data["label"])
    coefs = pd.Series(final[-1].coef_[0], index=FEATURES, name="logit_coef_standardized")

    pca = PCA().fit(data[FEATURES])
    loadings = pd.DataFrame(pca.components_.T, index=FEATURES,
                            columns=[f"PC{i + 1}" for i in range(len(FEATURES))])
    explained = pd.Series(pca.explained_variance_ratio_, index=loadings.columns, name="explained_variance_ratio")

    logger.info("Model CV: %d folds, mean AUC logit %.3f vs composite %.3f; mean IC logit %.3f vs composite %.3f",
                len(cv), cv["auc_logit"].mean(), cv["auc_composite"].mean(), cv["ic_logit"].mean(), cv["ic_composite"].mean())
    return {"status": "ok", "cv": cv, "oos_scores": pd.concat(oos, ignore_index=True),
            "coefficients": coefs, "pca_loadings": loadings, "pca_explained": explained, "horizon": h}
