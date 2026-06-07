#!/usr/bin/env python3
"""Scrape FIFA World Cup match results from Wikipedia (1930-2022)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from kaggle_team_features import (  # noqa: E402
    MIN_YEAR_WITH_KAGGLE,
    default_kaggle_dir,
    enrich_matches,
    ensure_kaggle_dataset,
    training_matches,
)

WORLD_CUP_YEARS = [
    1930, 1934, 1938, 1950, 1954, 1958, 1962, 1966, 1970,
    1974, 1978, 1982, 1986, 1990, 1994, 1998, 2002, 2006,
    2010, 2014, 2018, 2022,
]

USER_AGENT = "FifaWorldCupAnalysis/1.0 (educational data project)"
REQUEST_DELAY_SECONDS = 0.15


def _fetch_soup(session: requests.Session, url: str) -> BeautifulSoup:
    response = session.get(url, timeout=30)
    response.raise_for_status()
    return BeautifulSoup(response.text, "lxml")


def _parse_footballboxes(soup: BeautifulSoup) -> list[tuple[str, str, str]]:
    matches: list[tuple[str, str, str]] = []
    for match in soup.find_all("div", class_="footballbox"):
        home_el = match.find("th", class_="fhome")
        score_el = match.find("th", class_="fscore")
        away_el = match.find("th", class_="faway")
        if not all([home_el, score_el, away_el]):
            continue
        matches.append(
            (
                home_el.get_text(strip=True),
                score_el.get_text(strip=True),
                away_el.get_text(strip=True),
            )
        )
    return matches


def scrape_year(session: requests.Session, year: int) -> list[dict[str, object]]:
    seen: set[tuple[str, str, str]] = set()
    rows: list[dict[str, object]] = []

    def add_matches(matches: list[tuple[str, str, str]]) -> None:
        for home, score, away in matches:
            key = (home, score, away)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"home": home, "score": score, "away": away, "year": year})

    tournament_url = f"https://en.wikipedia.org/wiki/{year}_FIFA_World_Cup"
    add_matches(_parse_footballboxes(_fetch_soup(session, tournament_url)))

    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        group_url = f"https://en.wikipedia.org/wiki/{year}_FIFA_World_Cup_Group_{letter}"
        response = session.get(group_url, timeout=30)
        if response.status_code == 404:
            break
        add_matches(_parse_footballboxes(BeautifulSoup(response.text, "lxml")))

    return rows


def scrape_all_years(
    years: list[int] | None = None,
    delay_seconds: float = REQUEST_DELAY_SECONDS,
) -> pd.DataFrame:
    years = years or WORLD_CUP_YEARS
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    all_rows: list[dict[str, object]] = []
    for year in years:
        year_rows = scrape_year(session, year)
        all_rows.extend(year_rows)
        print(f"{year}: scraped {len(year_rows)} matches")
        if delay_seconds:
            time.sleep(delay_seconds)

    return pd.DataFrame(all_rows)


def clean_matches(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    walkover = df["home"].str.contains("Sweden", na=False) & df["away"].str.contains(
        "Austria", na=False
    )
    df = df.loc[~walkover].copy()

    df["score"] = df["score"].str.replace(r"[^\d–]", "", regex=True)
    df["home"] = df["home"].str.strip()
    df["away"] = df["away"].str.strip()
    df = df[df["score"].str.contains("–", na=False)].copy()

    df[["HomeGoals", "AwayGoals"]] = df["score"].str.split("–", expand=True)
    df = df.drop(columns=["score"])
    df = df.rename(columns={"home": "HomeTeam", "away": "AwayTeam", "year": "Year"})
    df = df.astype({"HomeGoals": int, "AwayGoals": int, "Year": int})
    df["TotalGoals"] = df["HomeGoals"] + df["AwayGoals"]
    return df


def enrich_with_kaggle_features(
    clean_df: pd.DataFrame, kaggle_dir: Path, download: bool = True
) -> pd.DataFrame:
    if download:
        ensure_kaggle_dataset(kaggle_dir)
    enriched = enrich_matches(clean_df, kaggle_dir=kaggle_dir)
    trainable = training_matches(enriched)
    print(
        f"Kaggle features: {len(trainable)} matches from {MIN_YEAR_WITH_KAGGLE}+ "
        f"with complete home/away stats"
    )
    return enriched


def write_outputs(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    output_dir: Path,
    *,
    skip_kaggle: bool = False,
    kaggle_dir: Path | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "fifa_worldcup_historical_data.csv"
    clean_path = output_dir / "clean_fifa_worldcup_matches.csv"
    raw_df.to_csv(raw_path, index=False)

    if not skip_kaggle:
        kaggle_dir = kaggle_dir or default_kaggle_dir(output_dir.parent)
        clean_df = enrich_with_kaggle_features(clean_df, kaggle_dir)

    clean_df.to_csv(clean_path, index=False)
    print(f"Wrote raw data ({len(raw_df)} rows) to {raw_path}")
    print(f"Wrote cleaned data ({len(clean_df)} rows) to {clean_path}")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Scrape FIFA World Cup match scores from Wikipedia (1930-2022)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "Data",
        help="Directory for CSV output (default: Data/)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=REQUEST_DELAY_SECONDS,
        help="Delay between tournament requests in seconds.",
    )
    parser.add_argument(
        "--skip-kaggle",
        action="store_true",
        help="Do not merge Kaggle pre-tournament team features into the clean CSV.",
    )
    parser.add_argument(
        "--kaggle-dir",
        type=Path,
        default=None,
        help="Directory for train.csv / test.csv (default: Data/kaggle_fifa_world_cup_team_dataset).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_df = scrape_all_years(delay_seconds=args.delay)
    clean_df = clean_matches(raw_df)
    write_outputs(
        raw_df,
        clean_df,
        args.output_dir,
        skip_kaggle=args.skip_kaggle,
        kaggle_dir=args.kaggle_dir,
    )


if __name__ == "__main__":
    main()
