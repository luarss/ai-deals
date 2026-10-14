from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.writer import DOCS_DIR, compact_archive_data, dump_compact_json  # noqa: E402

ARCHIVE_PATHS_TO_SKIP = {"manifest.json"}


def compact_file(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    compacted = dump_compact_json(compact_archive_data(json.loads(original)))
    if compacted == original:
        return False
    path.write_text(compacted, encoding="utf-8")
    return True


def archive_files() -> list[Path]:
    archive_paths = sorted(
        path
        for path in (DOCS_DIR / "archive").glob("*.json")
        if path.name not in ARCHIVE_PATHS_TO_SKIP
    )
    return archive_paths + [DOCS_DIR / "data" / "latest.json"]


def main() -> None:
    changed = 0
    for path in archive_files():
        if path.exists() and compact_file(path):
            changed += 1
    print(f"Compacted {changed} file(s); {len(archive_files()) - changed} already compact")


if __name__ == "__main__":
    main()
