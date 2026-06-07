"""Merge Kaggle pre-tournament team features into match-level datasets."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

# Wikipedia / fixture names -> Kaggle `team` column
TEAM_NAME_ALIASES: dict[str, str] = {
    "China": "China PR",
    "Curaçao": "Cura?o",
}

KAGGLE_DATASET_REF = "harrachimustapha/fifa-world-cup-team-dataset"
KAGGLE_DOWNLOAD_URL = (
    f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_DATASET_REF}"
)

# Pre-tournament signals (exclude tournament outcome labels)
KAGGLE_TEAM_COLUMNS: list[str] = [
    "fifa_rank_pre_tournament",
    "fifa_points_pre_tournament",
    "goals_scored_last_4y",
    "goals_received_last_4y",
    "wins_last_4y",
    "losses_last_4y",
    "draws_last_4y",
    "is_host",
    "squad_total_market_value_eur",
    "squad_avg_age",
    "world_cup_titles_before",
]

# Match-level column names after merge (home / away prefixes)
KAGGLE_TO_MATCH_SUFFIX: dict[str, str] = {
    "fifa_rank_pre_tournament": "FifaRank",
    "fifa_points_pre_tournament": "FifaPoints",
    "goals_scored_last_4y": "GoalsScoredLast4Y",
    "goals_received_last_4y": "GoalsReceivedLast4Y",
    "wins_last_4y": "WinsLast4Y",
    "losses_last_4y": "LossesLast4Y",
    "draws_last_4y": "DrawsLast4Y",
    "is_host": "IsHost",
    "squad_total_market_value_eur": "SquadMarketValueEur",
    "squad_avg_age": "SquadAvgAge",
    "world_cup_titles_before": "WorldCupTitlesBefore",
}

# Home/away sides that also get a home-minus-away diff column
DIFF_SOURCE_SUFFIXES: list[str] = [
    "FifaRank",
    "FifaPoints",
    "GoalsScoredLast4Y",
    "GoalsReceivedLast4Y",
    "WinsLast4Y",
    "LossesLast4Y",
    "DrawsLast4Y",
    "SquadMarketValueEur",
    "SquadAvgAge",
    "WorldCupTitlesBefore",
    "IsHost",
]

MATCH_KAGGLE_FEATURE_COLUMNS: list[str] = [
    f"Home{suffix}"
    for suffix in KAGGLE_TO_MATCH_SUFFIX.values()
] + [
    f"Away{suffix}"
    for suffix in KAGGLE_TO_MATCH_SUFFIX.values()
]

MATCH_DIFF_FEATURE_COLUMNS: list[str] = [
    f"Diff{suffix}" for suffix in DIFF_SOURCE_SUFFIXES
]

MATCH_DERIVED_FEATURE_COLUMNS: list[str] = ["SameContinent"]

# Market value is missing for 2002; impute at model time instead of dropping rows
IMPUTABLE_MATCH_COLUMNS: frozenset[str] = frozenset(
    {"HomeSquadMarketValueEur", "AwaySquadMarketValueEur", "DiffSquadMarketValueEur"}
)

MIN_YEAR_WITH_KAGGLE = 2002


def default_kaggle_dir(project_root: Path | None = None) -> Path:
    root = project_root or Path(__file__).resolve().parents[1]
    return root / "Data" / "kaggle_fifa_world_cup_team_dataset"


def normalize_team_for_kaggle(team: str) -> str:
    return TEAM_NAME_ALIASES.get(team, team)


def download_kaggle_dataset(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    response = requests.get(KAGGLE_DOWNLOAD_URL, timeout=60)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        zf.extractall(output_dir)


def ensure_kaggle_dataset(kaggle_dir: Path | None = None) -> Path:
    kaggle_dir = kaggle_dir or default_kaggle_dir()
    train_path = kaggle_dir / "train.csv"
    test_path = kaggle_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        print(f"Downloading Kaggle team dataset to {kaggle_dir} ...")
        download_kaggle_dataset(kaggle_dir)
    return kaggle_dir


def load_kaggle_teams(kaggle_dir: Path | None = None) -> pd.DataFrame:
    kaggle_dir = ensure_kaggle_dataset(kaggle_dir)
    train = pd.read_csv(kaggle_dir / "train.csv")
    test = pd.read_csv(kaggle_dir / "test.csv")
    teams = pd.concat([train, test], ignore_index=True)
    teams = teams.rename(columns={"version": "world_cup_year"})
    teams["team"] = teams["team"].astype(str)
    return teams


def _rename_team_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    renamed = {}
    for kaggle_col, suffix in KAGGLE_TO_MATCH_SUFFIX.items():
        renamed[kaggle_col] = f"{prefix}{suffix}"
    out = df[["world_cup_year", "team", *KAGGLE_TEAM_COLUMNS]].copy()
    return out.rename(columns=renamed)


def _drop_enriched_columns(matches: pd.DataFrame) -> pd.DataFrame:
    to_drop = [
        c
        for c in (
            *MATCH_KAGGLE_FEATURE_COLUMNS,
            *MATCH_DIFF_FEATURE_COLUMNS,
            *MATCH_DERIVED_FEATURE_COLUMNS,
        )
        if c in matches.columns
    ]
    if to_drop:
        matches = matches.drop(columns=to_drop)
    return matches


def _add_diff_features(matches: pd.DataFrame) -> pd.DataFrame:
    out = matches.copy()
    for suffix in DIFF_SOURCE_SUFFIXES:
        home_col = f"Home{suffix}"
        away_col = f"Away{suffix}"
        out[f"Diff{suffix}"] = out[home_col] - out[away_col]
    return out


def _add_same_continent(
    matches: pd.DataFrame,
    teams: pd.DataFrame,
    year_col: str,
) -> pd.DataFrame:
    out = matches.copy()
    continents = teams[["world_cup_year", "team", "continent"]].copy()
    home_continents = continents.rename(
        columns={
            "world_cup_year": year_col,
            "team": "_home_kaggle",
            "continent": "_home_continent",
        }
    )
    away_continents = continents.rename(
        columns={
            "world_cup_year": year_col,
            "team": "_away_kaggle",
            "continent": "_away_continent",
        }
    )
    out = out.merge(home_continents, on=[year_col, "_home_kaggle"], how="left")
    out = out.merge(away_continents, on=[year_col, "_away_kaggle"], how="left")
    both_known = out["_home_continent"].notna() & out["_away_continent"].notna()
    out["SameContinent"] = (
        out["_home_continent"] == out["_away_continent"]
    ).where(both_known).astype(float)
    return out.drop(columns=["_home_continent", "_away_continent"])


def enrich_matches(
    matches: pd.DataFrame,
    kaggle_dir: Path | None = None,
    year_col: str = "Year",
    home_col: str = "HomeTeam",
    away_col: str = "AwayTeam",
) -> pd.DataFrame:
    """Attach home/away Kaggle features and home-minus-away diffs.

    Pre-2002 rows keep NaN for Kaggle columns. Squad market value may be NaN
    (e.g. 2002); impute those columns when training the model.
    """
    matches = _drop_enriched_columns(matches)

    teams = load_kaggle_teams(kaggle_dir)

    out = matches.copy()
    out["_home_kaggle"] = out[home_col].map(normalize_team_for_kaggle)
    out["_away_kaggle"] = out[away_col].map(normalize_team_for_kaggle)

    home_feats = _rename_team_features(teams, "Home")
    away_feats = _rename_team_features(teams, "Away")

    home_feats = home_feats.rename(
        columns={"world_cup_year": year_col, "team": "_home_kaggle"}
    )
    away_feats = away_feats.rename(
        columns={"world_cup_year": year_col, "team": "_away_kaggle"}
    )

    out = out.merge(home_feats, on=[year_col, "_home_kaggle"], how="left")
    out = out.merge(away_feats, on=[year_col, "_away_kaggle"], how="left")
    out = _add_same_continent(out, teams, year_col)
    out = out.drop(columns=["_home_kaggle", "_away_kaggle"])
    return _add_diff_features(out)


def training_matches(
    matches: pd.DataFrame,
    min_year: int = MIN_YEAR_WITH_KAGGLE,
    year_col: str = "Year",
) -> pd.DataFrame:
    """Rows suitable for RF training (2002+ with complete non-imputable Kaggle features)."""
    df = matches.loc[matches[year_col] >= min_year].copy()
    required = [
        c
        for c in (
            *MATCH_KAGGLE_FEATURE_COLUMNS,
            *MATCH_DIFF_FEATURE_COLUMNS,
            *MATCH_DERIVED_FEATURE_COLUMNS,
        )
        if c not in IMPUTABLE_MATCH_COLUMNS
    ]
    return df.dropna(subset=required)


def match_model_feature_columns(include_team_ids: bool = True) -> list[str]:
    cols: list[str] = []
    if include_team_ids:
        cols.extend(["HomeTeamEncoded", "AwayTeamEncoded"])
    cols.extend(MATCH_KAGGLE_FEATURE_COLUMNS)
    cols.extend(MATCH_DIFF_FEATURE_COLUMNS)
    cols.extend(MATCH_DERIVED_FEATURE_COLUMNS)
    return cols
