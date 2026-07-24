"""Standalone backfill for docs/data/timeseries.json (spec 04 §4).

Rebuilds the whole timeseries artifact from every archive on disk. Run once to
backfill before the feature goes live; the daily pipeline rebuilds it in-process
via src.writer.write_timeseries, so this script is only needed for manual runs.

    uv run python scripts/build_timeseries.py
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow "python scripts/build_timeseries.py" without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.writer import write_timeseries  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    write_timeseries(datetime.now(timezone.utc))


if __name__ == "__main__":
    main()
