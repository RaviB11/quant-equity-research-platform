import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd

from eqscreen import validate


def test_validation_report_and_archive(demo_run):
    path = validate.build(demo_run, demo_run, archive=True)
    text = path.read_text()
    for heading in ("Filing availability", "Corporate actions", "Survivorship",
                    "Real results against the synthetic benchmark", "Preserved inputs"):
        assert f"## {heading}" in text
    assert "Demo mode" in text and "passed on" in text                # point-in-time check ran
    comp = pd.read_csv(path.parent / "real_vs_synthetic.csv")
    assert {"Value IC, 1 month", "Composite CAGR, net", "Fiscal periods with revisions"} <= set(comp["metric"])
    lags = pd.read_csv(path.parent / "filing_lags.csv")
    assert (lags["lag_days"] > 0).all()

    archive = sorted((Path(demo_run["project"]["db_path"]).parent / "archive").glob("inputs_*.zip"))[-1]
    with zipfile.ZipFile(archive) as z:
        names = set(z.namelist())
        assert {"demo.db", "config.yaml", "run_summary.json", "ARCHIVE_MANIFEST.json"} <= names
        meta = json.loads(z.read("ARCHIVE_MANIFEST.json"))
        assert meta["database_sha256"] == hashlib.sha256(z.read("demo.db")).hexdigest()


def test_observations_state_significance_from_numbers_only():
    obs = validate.observations({"ic_value_score_1": (0.03, 3.1), "ic_momentum_score_1": (0.005, 0.8),
                                 "excess": 0.02, "excess_t": 1.2, "cagr": 0.08, "lag0_cagr": 0.09})
    text = " ".join(obs)
    assert "Value: mean 1-month rank IC 0.030 with Newey-West t 3.1, statistically distinguishable" in text
    assert "Momentum" in text and "not distinguishable from zero" in text
    assert "weak evidence" in text and "Same-close trading would have reported 9.0%" in text


def test_filing_validation_uses_only_the_current_universe(fresh_demo_db):
    cfg, conn = fresh_demo_db
    keep = [r[0] for r in conn.execute("SELECT ticker FROM securities ORDER BY ticker LIMIT 10")]
    conn.execute(f"DELETE FROM securities WHERE ticker NOT IN ({','.join('?' * len(keep))})", keep)
    conn.commit()
    lines, lags = validate.filing_validation(cfg)
    assert set(lags["ticker"]) <= set(keep)
    assert "| Companies / fiscal periods | 10 /" in "\n".join(lines)
