import numpy as np
import pandas as pd

from eqscreen.model import date_folds


def test_folds_are_purged_by_the_label_horizon():
    dates = pd.date_range("2010-01-31", periods=120, freq="ME").to_numpy()
    folds = date_folds(dates, n_splits=5, gap=12, min_train=24)
    assert len(folds) >= 2
    pos = {d: i for i, d in enumerate(dates)}
    for train, test in folds:
        assert len(train) >= 24
        # 12-month labels from the last training date end before the first test date.
        assert pos[test.min()] - pos[train.max()] > 12
        assert set(train).isdisjoint(test)


def test_too_little_history_returns_no_folds():
    dates = pd.date_range("2020-01-31", periods=30, freq="ME").to_numpy()
    assert date_folds(dates, n_splits=5, gap=12, min_train=24) == []


def test_model_outputs(demo_run):
    from pathlib import Path
    import json

    out = Path(demo_run["project"]["output_dir"])
    assert json.loads((out / "model_status.json").read_text())["status"] == "ok"
    cv = pd.read_csv(out / "model_cv.csv")
    assert {"auc_logit", "auc_composite", "ic_logit", "ic_composite"} <= set(cv.columns)
    assert cv["auc_logit"].between(0, 1).all()
    summary = pd.read_csv(out / "backtest_summary.csv")
    assert {"ml_logit_oos", "composite_same_cohort"} <= set(summary["strategy"])


def test_same_cohort_drops_rows_the_model_could_not_score():
    from eqscreen.model import same_cohort_panel

    m = pd.Timestamp("2020-01-31")
    panel = pd.DataFrame({"month_end": [m] * 3, "ticker": ["A", "B", "C"], "composite": [1.0, 0.5, -0.2]})
    # C has no 12-month label, so the model never scored it.
    oos = pd.DataFrame({"month_end": [m, m], "ticker": ["A", "B"], "ml_logit": [0.6, 0.4], "ml_ridge": [0.1, 0.0]})
    p = same_cohort_panel(panel, oos).set_index("ticker")
    assert p.loc["C", "composite"] == -0.2 and np.isnan(p.loc["C", "composite_same_cohort"])
    assert p["composite_same_cohort"].notna().sum() == p["ml_logit_oos"].notna().sum() == 2
