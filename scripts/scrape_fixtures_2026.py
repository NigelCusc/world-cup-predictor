#!/usr/bin/env python3
"""Scrape 2026 FIFA World Cup fixtures from Wikipedia."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from scrape_historical_matches import REQUEST_DELAY_SECONDS, USER_AGENT, scrape_year

TOURNAMENT_YEAR = 2026
EXPECTED_FIXTURE_COUNT = 104


def scrape_fixtures_2026(delay_seconds: float = REQUEST_DELAY_SECONDS) -> pd.DataFrame:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    rows = scrape_year(session, TOURNAMENT_YEAR)
    if delay_seconds:
        time.sleep(delay_seconds)
    df = pd.DataFrame(rows)
    if len(df) != EXPECTED_FIXTURE_COUNT:
        print(
            f"Warning: expected {EXPECTED_FIXTURE_COUNT} fixtures, got {len(df)}. "
            "Wikipedia may have changed; verify the output."
        )
    return df


def clean_fixtures(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df["home"] = df["home"].str.strip()
    df["away"] = df["away"].str.strip()
    df["score"] = df["score"].str.strip()
    df["year"] = df["year"].astype(int)
    return df[["home", "score", "away", "year"]]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Scrape 2026 FIFA World Cup fixtures from Wikipedia."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "Data",
        help="Directory for CSV output (default: Data/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_df = scrape_fixtures_2026()
    fixtures_df = clean_fixtures(raw_df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "clean_fifa_worldcup_fixture.csv"
    fixtures_df.to_csv(output_path, index=False)
    print(f"Scraped {len(fixtures_df)} fixtures for {TOURNAMENT_YEAR}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
