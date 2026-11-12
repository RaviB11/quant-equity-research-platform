"""SQLite helpers."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT

SQL_DIR = ROOT / "sql"


def connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def init_schema(conn: sqlite3.Connection) -> None:
    _migrate_unversioned_fundamentals(conn)
    conn.executescript(read_sql_file("schema.sql"))
    _add_missing_columns(conn, "fundamentals", {"operating_cash_flow": "REAL", "capex": "REAL", "total_assets": "REAL"})
    conn.commit()


def _add_missing_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    """Columns added in v0.4 (cash-flow items) for databases created earlier."""
    have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    for col, typ in columns.items():
        if col not in have:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")


def _migrate_unversioned_fundamentals(conn: sqlite3.Connection) -> None:
    """Databases from v0.1 keyed fundamentals by (ticker, period_end, source) only."""
    cols = [r[1] for r in conn.execute("PRAGMA table_info(fundamentals)")]
    if not cols or "version_type" in cols:
        return
    conn.execute("ALTER TABLE fundamentals RENAME TO fundamentals_v01")
    conn.executescript(read_sql_file("schema.sql"))
    keep = ", ".join(cols)
    conn.execute(f"INSERT INTO fundamentals ({keep}) SELECT {keep} FROM fundamentals_v01")
    conn.execute("DROP TABLE fundamentals_v01")
    conn.commit()


def read_sql_file(name: str) -> str:
    return (SQL_DIR / name).read_text(encoding="utf-8")


def _to_py(value):
    """Convert numpy/pandas scalars to types sqlite3 accepts; NaN/NA -> NULL."""
    if value is None or value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if np.isnan(value) else float(value)
    return value


def upsert_df(conn: sqlite3.Connection, table: str, df: pd.DataFrame) -> int:
    """INSERT OR REPLACE every row of ``df`` (column names must match the table)."""
    if df is None or df.empty:
        return 0
    cols = list(df.columns)
    sql = f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})"
    records = [tuple(_to_py(v) for v in row) for row in df.itertuples(index=False, name=None)]
    conn.executemany(sql, records)
    conn.commit()
    return len(records)


def table_df(conn: sqlite3.Connection, sql: str, params: dict | tuple | None = None,
             date_cols: tuple[str, ...] = ()) -> pd.DataFrame:
    df = pd.read_sql_query(sql, conn, params=params)
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df
