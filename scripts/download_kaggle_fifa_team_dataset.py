"""Download harrachimustapha/fifa-world-cup-team-dataset from Kaggle (no API key required)."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from kaggle_team_features import default_kaggle_dir, download_kaggle_dataset  # noqa: E402


def main() -> None:
    output_dir = default_kaggle_dir()
    download_kaggle_dataset(output_dir)
    print(f"Saved train.csv and test.csv to {output_dir}")


if __name__ == "__main__":
    main()
