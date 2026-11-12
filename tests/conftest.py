import shutil
from pathlib import Path

import pytest
import yaml

from eqscreen.config import ROOT, load_config
from eqscreen.db import connect


def _write_cfg(tmp: Path, overrides: dict | None = None) -> Path:
    cfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    cfg["demo"]["db_path"] = str(tmp / "demo.db")
    cfg["demo"]["output_dir"] = str(tmp / "out")
    cfg["demo"]["n_tickers"] = 120
    cfg["project"]["log_dir"] = str(tmp / "logs")
    cfg["project"]["raw_dir"] = str(tmp / "raw")
    for section, vals in (overrides or {}).items():
        cfg[section].update(vals)
    path = tmp / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def demo_run(tmp_path_factory):
    """Run the full demo pipeline once; tests read its database and outputs."""
    from eqscreen.pipeline import run

    tmp = tmp_path_factory.mktemp("demo")
    cfg_path = _write_cfg(tmp)
    run(mode="demo", config_path=str(cfg_path))
    cfg = load_config(cfg_path, mode="demo")
    return cfg


@pytest.fixture()
def demo_conn(demo_run):
    conn = connect(demo_run["project"]["db_path"])
    yield conn
    conn.close()


@pytest.fixture()
def fresh_demo_db(tmp_path):
    """A writable copy-free demo database for tests that mutate data."""
    from eqscreen.db import init_schema
    from eqscreen.synthetic import load_into_db

    cfg_path = _write_cfg(tmp_path)
    cfg = load_config(cfg_path, mode="demo")
    conn = connect(cfg["project"]["db_path"])
    init_schema(conn)
    load_into_db(conn, n_tickers=80, seed=11)
    yield cfg, conn
    conn.close()
