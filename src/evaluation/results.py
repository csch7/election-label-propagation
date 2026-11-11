"""Where experiment outputs are written: results/runs/<date>_<experiment>[_<variant>].csv"""
from datetime import date
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parents[2] / "results" / "runs"


def run_path(experiment, variant=None, day=None):
    """Path for one run's CSV, e.g. results/runs/2026-11-11_previous_year_block.csv.
    A rerun on the same day overwrites the file."""
    name = "_".join([str(day or date.today()), experiment] + ([variant] if variant else []))
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return RUNS_DIR / f"{name}.csv"
