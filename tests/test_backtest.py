import numpy as np
import pandas as pd
import pytest

from eqscreen import backtest as bt
from eqscreen.factors import load_panel


def test_perf_stats_known_values():
    r = pd.Series([0.02, 0.0] * 6)
    s = bt.perf_stats(r)
    assert s["cagr"] == pytest.approx((1.02 ** 6) - 1)
    assert s["sharpe"] == pytest.approx(np.sqrt(11))
    assert s["max_drawdown"] == 0
    assert bt.perf_stats(pd.Series([0.1, -0.5, 0.2]))["max_drawdown"] == pytest.approx(-0.5)


def _toy(scores: dict, fwd_rows: list[list[float]]):
    dates = pd.date_range("2020-01-31", periods=len(fwd_rows), freq="ME")
    panel = pd.DataFrame([{"month_end": d, "ticker": t, "score": v} for d in dates for t, v in scores.items()])
    fwd = pd.DataFrame(fwd_rows, index=dates, columns=list(scores))
    return panel, fwd


def test_monthly_backtest_costs_and_turnover():
    panel, fwd = _toy({"A": 3, "B": 2, "C": 1}, [[0.05, 0.0, -0.01], [0.02, 0.01, 0.0], [np.nan] * 3])
    res = bt.run_backtest(panel, fwd, ["A", "B", "C"], None, "score", top_q=0.67, holding=1,
                          cost_bps=10, min_names=3, min_holdings=1)
    assert len(res) == 2
    assert res["gross"].tolist() == pytest.approx([0.05, 0.02])
    assert res.iloc[0]["net"] == pytest.approx(0.05 - 0.001)      # 100% of weight traded at 10 bps
    assert res.iloc[1]["turnover"] == 0 and res.iloc[1]["net"] == pytest.approx(0.02)
    assert res.iloc[0]["ew_universe"] == pytest.approx(0.04 / 3)


def test_buy_and_hold_weights_drift_within_holding_period():
    panel, fwd = _toy({"A": 3, "B": 2, "C": 1}, [[1.0, 0.0, 0.0], [0.0, 0.3, 0.0], [np.nan] * 3])
    res = bt.run_backtest(panel, fwd, ["A", "B", "C"], None, "score", top_q=0.5, holding=2,
                          cost_bps=0, min_names=3, min_holdings=2)
    # Equal weight A/B, A doubles -> weights 2/3, 1/3, then B +30% -> 0.1
    assert res["gross"].tolist() == pytest.approx([0.5, 0.1])


def test_planted_value_signal_is_recovered(demo_run, demo_conn):
    """Known-answer test: the synthetic market has a value effect and no momentum effect."""
    panel = load_panel(demo_conn)
    value = bt.ic_summary(bt.ic_series(panel, "value_score", "ret_fwd_1m"), 1)
    mom = bt.ic_summary(bt.ic_series(panel, "momentum_score", "ret_fwd_1m"), 1)
    assert value["mean_ic"] > 0.02 and value["t_stat"] > 3
    assert abs(mom["mean_ic"]) < value["mean_ic"] / 2
    dec = bt.decile_returns(panel, "composite", (12,))
    assert dec.set_index("decile").loc[10, "avg_excess_return"] > dec.set_index("decile").loc[1, "avg_excess_return"]


def test_shuffled_scores_carry_no_signal(demo_run, demo_conn):
    """Permutation test: shuffling scores within each month destroys the signal.

    One shuffle is noisy (its mean IC has a standard deviation of roughly 0.007 here), so
    the test averages 20 shuffles and also checks the real value IC beats all of them.
    """
    panel = load_panel(demo_conn)
    rng = np.random.default_rng(0)
    shuffled = []
    for _ in range(20):
        p = panel.assign(shuffled=panel.groupby("month_end")["composite"].transform(
            lambda s: pd.Series(rng.permutation(s.to_numpy()), index=s.index)))
        shuffled.append(bt.ic_series(p, "shuffled", "ret_fwd_1m").mean())
    real = bt.ic_series(panel, "value_score", "ret_fwd_1m").mean()
    assert abs(np.mean(shuffled)) < 0.006
    assert real > max(shuffled)


def test_execution_lag_shifts_returns_by_one_month():
    panel, fwd = _toy({"A": 3, "B": 2, "C": 1}, [[0.05, 0.0, 0.0], [0.02, 0.0, 0.0], [0.03, 0.0, 0.0], [np.nan] * 3])
    kw = dict(top_q=0.67, holding=1, cost_bps=0, min_names=3, min_holdings=1)
    now = bt.run_backtest(panel, fwd, ["A", "B", "C"], None, "score", lag=0, **kw)
    later = bt.run_backtest(panel, fwd, ["A", "B", "C"], None, "score", lag=1, **kw)
    assert now["gross"].tolist() == pytest.approx([0.05, 0.02, 0.03])
    assert later["gross"].tolist() == pytest.approx([0.02, 0.03])          # first month skipped
    assert (later["signal_date"] < later["formation_date"]).all()


def test_newey_west_matches_iid_without_lags_and_widens_with_autocorrelation():
    rng = np.random.default_rng(1)
    iid = pd.Series(rng.normal(0.01, 0.02, 400))
    naive = iid.mean() / (iid.std(ddof=0) / np.sqrt(len(iid)))
    assert bt.newey_west_tstat(iid, lags=0) == pytest.approx(naive)
    e = rng.normal(0, 0.02, 400)
    ar = np.zeros(400)
    for t in range(1, 400):
        ar[t] = 0.8 * ar[t - 1] + e[t]
    ar = pd.Series(ar + 0.005)
    assert abs(bt.newey_west_tstat(ar, lags=12)) < abs(bt.newey_west_tstat(ar, lags=0))


@pytest.mark.parametrize("returns, expected", [
    ([-0.2, 0.1, 0.0], -0.2),      # loss in the first month (audit P1 reproduction)
    ([-0.5, 0.0, 0.0], -0.5),
    ([0.1, -0.5, 0.2], -0.5),      # peak after a gain
    ([0.05, 0.05], 0.0),           # never below a prior peak
])
def test_max_drawdown_counts_starting_nav(returns, expected):
    assert bt.perf_stats(pd.Series(returns))["max_drawdown"] == pytest.approx(expected)


def test_missing_return_policies():
    panel, fwd = _toy({"A": 3, "B": 2, "C": 1}, [[0.10, np.nan, 0.0], [0.10, 0.20, 0.0], [np.nan] * 3])
    kw = dict(top_q=0.5, holding=2, cost_bps=0, min_names=3, min_holdings=2)
    cash = bt.run_backtest(panel, fwd, ["A", "B", "C"], None, "score", missing_return="cash", **kw)
    loss = bt.run_backtest(panel, fwd, ["A", "B", "C"], None, "score", missing_return="loss", **kw)
    assert cash["gross"].tolist() == pytest.approx([0.05, (1.1 * 0.1 + 1.0 * 0.2) / 2.1])
    assert loss.iloc[0]["gross"] == pytest.approx(-0.45)     # B treated as a total loss
    assert loss.iloc[1]["gross"] == pytest.approx(0.10)      # only A is left
    assert cash.iloc[0]["n_missing"] == 1
    summ = bt.summarize(cash, 0.0)
    assert summ.loc["strategy_net", "missing_return_share"] == pytest.approx(1 / 4)


def test_decile_date_counts_are_per_decile():
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    rows = [{"month_end": d, "ticker": f"T{i}", "s": float(i), "ret_fwd_1m": 0.01 * i} for d in dates for i in range(10)]
    panel = pd.DataFrame(rows)
    panel.loc[(panel["month_end"] == dates[0]) & (panel["ticker"] == "T9"), "ret_fwd_1m"] = np.nan
    dec = bt.decile_returns(panel, "s", (1,), min_names=10).set_index("decile")
    assert dec.loc[10, "n_dates"] == 2 and dec.loc[1, "n_dates"] == 3


# --------------------------------------------------------------------------- calendar continuity (audit P1-B)
def test_missing_signal_month_is_carried_not_skipped():
    """The audit's reproduction: scores in Jan and Mar, none in Feb."""
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    panel = pd.DataFrame({"month_end": [dates[0], dates[2]], "ticker": ["A", "A"], "s": [1.0, 1.0]})
    fwd = pd.DataFrame({"A": [0.01, 0.02, 0.03, np.nan]}, index=dates)
    res = bt.run_backtest(panel, fwd, ["A"], None, "s", top_q=0.0, min_names=1, min_holdings=1, holding=1, cost_bps=0)
    assert [d.strftime("%Y-%m-%d") for d in res["period_end"]] == ["2020-02-29", "2020-03-31", "2020-04-30"]
    assert res["gross"].tolist() == pytest.approx([0.01, 0.02, 0.03])
    assert res["carried"].tolist() == [False, True, False] and res["rebalanced"].tolist() == [True, False, True]
    summ = bt.summarize(res, 0.0)
    assert summ.loc["strategy_net", "months"] == 3 and summ.loc["strategy_net", "carried_rebalances"] == 1


def test_one_missing_stock_does_not_cancel_the_month():
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    rows = [{"month_end": d, "ticker": t, "s": v} for d in dates for t, v in (("A", 3), ("B", 2), ("C", 1))]
    panel = pd.DataFrame(rows)
    panel = panel[~((panel["month_end"] == dates[1]) & (panel["ticker"] == "A"))]   # A unscored in Feb
    fwd = pd.DataFrame({"A": [0.1, 0.1, np.nan], "B": [0.0, 0.5, np.nan], "C": [0.0, 0.0, np.nan]}, index=dates)
    res = bt.run_backtest(panel, fwd, ["A", "B", "C"], None, "s", top_q=0.67, min_names=2, min_holdings=1, cost_bps=0)
    assert res["rebalanced"].all() and res["gross"].tolist() == pytest.approx([0.1, 0.5])     # Feb picks B


def test_six_month_schedule_counts_calendar_months_and_carries_a_missed_rebalance():
    dates = pd.date_range("2020-01-31", periods=10, freq="ME")
    panel = pd.DataFrame([{"month_end": d, "ticker": t, "s": v} for d in dates for t, v in (("A", 2), ("B", 1))])
    panel = panel[panel["month_end"] != dates[6]]                    # no signal at the scheduled month 6
    fwd = pd.DataFrame({"A": [0.01] * 9 + [np.nan], "B": [0.02] * 9 + [np.nan]}, index=dates)
    res = bt.run_backtest(panel, fwd, ["A", "B"], None, "s", top_q=0.5, min_names=2, min_holdings=1, holding=6,
                          cost_bps=0)
    assert len(res) == 9
    bt.assert_contiguous(res["period_end"])
    assert res["rebalanced"].tolist() == [True] + [False] * 8         # month 6 had no signal
    assert res["carried"].tolist() == [False] * 6 + [True] + [False] * 2


def test_benchmark_uses_the_same_months_as_the_strategy():
    dates = pd.date_range("2020-01-31", periods=5, freq="ME")
    panel = pd.DataFrame([{"month_end": d, "ticker": t, "s": v} for d in dates[[0, 3]] for t, v in (("A", 2), ("B", 1))])
    fwd = pd.DataFrame({"A": [0.1, 0.2, 0.3, 0.4, np.nan], "B": [0.0, 0.1, 0.0, 0.1, np.nan]}, index=dates)
    res = bt.run_backtest(panel, fwd, ["A", "B"], None, "s", top_q=0.5, min_names=2, min_holdings=1, cost_bps=0)
    for r in res.itertuples():
        m = r.period_end - pd.offsets.MonthEnd(1)
        assert r.ew_universe == pytest.approx(fwd.loc[m].mean())


def test_summary_refuses_a_gapped_series():
    gapped = pd.DataFrame({"period_end": pd.to_datetime(["2020-02-29", "2020-04-30"]), "gross": [0.01, 0.02],
                           "net": [0.01, 0.02], "turnover": [1.0, 0.0], "n_holdings": [1, 1], "n_missing": [0, 0],
                           "ew_universe": [0.0, 0.0], "benchmark": [np.nan, np.nan]})
    with pytest.raises(ValueError, match="not consecutive"):
        bt.summarize(gapped, 0.0)


def test_starts_at_first_signal_and_ends_at_last_return():
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    panel = pd.DataFrame([{"month_end": d, "ticker": "A", "s": 1.0} for d in dates[2:]])
    fwd = pd.DataFrame({"A": [0.01, 0.01, 0.01, 0.01, np.nan, np.nan]}, index=dates)
    res = bt.run_backtest(panel, fwd, ["A"], None, "s", top_q=0.0, min_names=1, min_holdings=1, cost_bps=0)
    assert res["period_end"].iloc[0] == pd.Timestamp("2020-04-30") and res["period_end"].iloc[-1] == pd.Timestamp("2020-05-31")
