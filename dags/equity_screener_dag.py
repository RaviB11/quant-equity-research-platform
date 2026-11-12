"""Airflow DAG: monthly equity screen refresh.

One task per pipeline stage so a failure (say, a provider outage in `fundamentals`)
is visible and retryable on its own. The pipeline uses a local SQLite file, so run
this with a single-machine executor (LocalExecutor / SequentialExecutor) on the box
that holds the repo.

Airflow does not run natively on Windows: use WSL2 or Docker, or use
scripts/register_monthly_task.ps1 (Windows Task Scheduler) instead.

Retries re-run a stage under the same run id; each attempt is recorded in
`pipeline_stage_runs`, and the run's status reflects the latest attempt of every stage.

Setup: `pip install -e /path/to/equity-screener` in the Airflow environment, then
copy or symlink this file into $AIRFLOW_HOME/dags.
"""
from __future__ import annotations

from datetime import datetime, timedelta

try:  # Airflow 3
    from airflow.sdk import DAG
except ImportError:  # Airflow 2.x
    from airflow import DAG
try:
    from airflow.providers.standard.operators.python import PythonOperator
except ImportError:
    from airflow.operators.python import PythonOperator

from eqscreen.pipeline import LIVE_STAGES, run


def _run_stage(stage: str, run_id: str) -> str:
    # Every task of one DAG run records into the same pipeline_runs row (shared lineage).
    return run(mode="live", stages=[stage], run_id=f"airflow-{run_id}")


with DAG(
    dag_id="equity_screener_monthly",
    description="Fetch prices and fundamentals, run quality checks, recompute factors, backtest, report",
    schedule="0 6 2 * *",            # 06:00 on the 2nd: the previous month-end close is settled
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=10)},
    tags=["equity", "factors"],
) as dag:
    tasks = [PythonOperator(task_id=stage, python_callable=_run_stage,
                            op_kwargs={"stage": stage, "run_id": "{{ run_id }}"})   # templated by Airflow
             for stage in LIVE_STAGES]
    for upstream, downstream in zip(tasks, tasks[1:]):
        upstream >> downstream
