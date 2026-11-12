"""Logging to console + file, and an audit trail of every data fetch in SQLite."""
from __future__ import annotations

import logging
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

_FMT = "%(asctime)s | %(levelname)-7s | %(message)s"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]


def get_logger(log_dir: str | None = None, name: str = "eqscreen", console_level: int | str = logging.INFO
               ) -> logging.Logger:
    """Console handler at `console_level`; the log file always records INFO and above."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    consoles = [h for h in logger.handlers
                if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
    if not consoles:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter(_FMT))
        logger.addHandler(sh)
        consoles = [sh]
    for h in consoles:
        h.setLevel(console_level)
    if log_dir:
        log_file = str(Path(log_dir) / "pipeline.log")
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(Path(log_file).resolve())
                   for h in logger.handlers):
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(logging.Formatter(_FMT))
            logger.addHandler(fh)
    return logger


def log_fetch(conn: sqlite3.Connection, run_id: str | None, source: str, endpoint: str,
              ticker: str | None, status: str, n_rows: int = 0, message: str = "") -> None:
    """Record one API call (or cache hit) with its UTC timestamp."""
    conn.execute(
        "INSERT INTO fetch_log (run_id, fetched_at_utc, source, endpoint, ticker, status, n_rows, message) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, utc_now(), source, endpoint, ticker, status, int(n_rows), str(message)[:500]),
    )
    conn.commit()
