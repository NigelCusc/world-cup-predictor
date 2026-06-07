"""FIFA World Cup 2026 tournament format: groups, qualification, and knockout bracket."""

from __future__ import annotations

import pickle
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

GROUP_LETTERS = list("ABCDEFGHIJKL")
NUM_GROUPS = len(GROUP_LETTERS)
GROUP_STAGE_MATCHES = 72
KNOCKOUT_TEAMS = 32
BEST_THIRD_PLACE_COUNT = 8

KNOCKOUT_ROUND_BY_MATCH: dict[int, str] = {}
for _n in range(73, 89):
    KNOCKOUT_ROUND_BY_MATCH[_n] = "Round of 32"
for _n in range(89, 97):
    KNOCKOUT_ROUND_BY_MATCH[_n] = "Round of 16"
for _n in range(97, 101):
    KNOCKOUT_ROUND_BY_MATCH[_n] = "Quarterfinals"
KNOCKOUT_ROUND_BY_MATCH[101] = "Semifinals"
KNOCKOUT_ROUND_BY_MATCH[102] = "Semifinals"
KNOCKOUT_ROUND_BY_MATCH[103] = "Third place"
KNOCKOUT_ROUND_BY_MATCH[104] = "Final"

_PLACEHOLDER_MARKERS = (
    "Winner Group",
    "Runner-up Group",
    "3rd Group",
    "Winner Match",
    "Loser Match",
)


def default_group_standings_path(project_root: Path | None = None) -> Path:
    root = project_root or Path(__file__).resolve().parents[1]
    return root / "Data" / "group_standings_2026.pkl"


def is_placeholder_team(name: str) -> bool:
    return any(name.startswith(marker) for marker in _PLACEHOLDER_MARKERS)


def load_group_teams(pkl_path: Path | None = None) -> dict[str, list[str]]:
    """Return ``{'Group A': [teams...], ...}`` from Wikipedia standings extract."""
    path = pkl_path or default_group_standings_path()
    with path.open("rb") as handle:
        tables: dict[str, pd.DataFrame] = pickle.load(handle)
    return {group: list(table["Team"]) for group, table in tables.items()}


def split_fixtures(fixtures_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split scraped fixtures into group-stage (72) and knockout (32) rows."""
    home_col = "home" if "home" in fixtures_df.columns else "HomeTeam"
    away_col = "away" if "away" in fixtures_df.columns else "AwayTeam"
    mask = fixtures_df[home_col].apply(is_placeholder_team) | fixtures_df[away_col].apply(
        is_placeholder_team
    )
    group_df = fixtures_df.loc[~mask].copy()
    knockout_df = fixtures_df.loc[mask].copy()
    if len(group_df) != GROUP_STAGE_MATCHES:
        raise ValueError(
            f"Expected {GROUP_STAGE_MATCHES} group-stage fixtures, found {len(group_df)}"
        )
    return group_df, knockout_df


def _empty_stats() -> dict[str, int]:
    return {"Points": 0, "GF": 0, "GA": 0, "GD": 0}


def _apply_result(
    standings: dict[str, dict[str, int]], home: str, away: str, home_goals: int, away_goals: int
) -> None:
    standings.setdefault(home, _empty_stats())
    standings.setdefault(away, _empty_stats())
    standings[home]["GF"] += home_goals
    standings[home]["GA"] += away_goals
    standings[away]["GF"] += away_goals
    standings[away]["GA"] += home_goals
    standings[home]["GD"] = standings[home]["GF"] - standings[home]["GA"]
    standings[away]["GD"] = standings[away]["GF"] - standings[away]["GA"]
    if home_goals > away_goals:
        standings[home]["Points"] += 3
    elif away_goals > home_goals:
        standings[away]["Points"] += 3
    else:
        standings[home]["Points"] += 1
        standings[away]["Points"] += 1


def compute_group_standings(
    predictions_df: pd.DataFrame,
    group_teams: dict[str, list[str]],
    *,
    home_col: str = "HomeTeam",
    away_col: str = "AwayTeam",
    home_goals_col: str = "PredictedHomeGoals",
    away_goals_col: str = "PredictedAwayGoals",
) -> dict[str, pd.DataFrame]:
    """Build per-group tables from predicted group-stage results only."""
    group_matches = predictions_df.loc[
        ~predictions_df[home_col].apply(is_placeholder_team)
        & ~predictions_df[away_col].apply(is_placeholder_team)
    ]

    tables: dict[str, pd.DataFrame] = {}
    for group_name, teams in group_teams.items():
        stats = {team: _empty_stats() for team in teams}
        team_set = set(teams)
        for _, row in group_matches.iterrows():
            home, away = row[home_col], row[away_col]
            if home not in team_set or away not in team_set:
                continue
            _apply_result(
                stats,
                home,
                away,
                int(row[home_goals_col]),
                int(row[away_goals_col]),
            )

        df = pd.DataFrame.from_dict(stats, orient="index").reset_index().rename(
            columns={"index": "Team"}
        )
        df = df.sort_values(
            by=["Points", "GD", "GF", "Team"],
            ascending=[False, False, False, True],
        )
        df.insert(0, "Group", group_name)
        df["GroupRank"] = range(1, len(df) + 1)
        tables[group_name] = df.reset_index(drop=True)
    return tables


def rank_third_place_teams(
    group_tables: dict[str, pd.DataFrame],
    fifa_ranks: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Rank all 12 third-placed teams; top 8 qualify for the Round of 32."""
    thirds: list[dict[str, Any]] = []
    for group_name, table in group_tables.items():
        third = table.loc[table["GroupRank"] == 3].iloc[0]
        letter = group_name.split()[-1]
        thirds.append(
            {
                "Group": letter,
                "Team": third["Team"],
                "Points": int(third["Points"]),
                "GD": int(third["GD"]),
                "GF": int(third["GF"]),
                "FifaRank": (fifa_ranks or {}).get(third["Team"], 9999.0),
            }
        )

    ranked = pd.DataFrame(thirds).sort_values(
        by=["Points", "GD", "GF", "FifaRank", "Team"],
        ascending=[False, False, False, True, True],
    )
    ranked["ThirdPlaceRank"] = range(1, len(ranked) + 1)
    ranked["Qualified"] = ranked["ThirdPlaceRank"] <= BEST_THIRD_PLACE_COUNT
    return ranked.reset_index(drop=True)


def build_qualification_lookup(
    group_tables: dict[str, pd.DataFrame],
    third_place_ranking: pd.DataFrame,
) -> dict[str, str]:
    """Map bracket placeholders (e.g. ``Winner Group A``) to team names."""
    lookup: dict[str, str] = {}
    for group_name, table in group_tables.items():
        letter = group_name.split()[-1]
        winner = table.loc[table["GroupRank"] == 1, "Team"].iloc[0]
        runner_up = table.loc[table["GroupRank"] == 2, "Team"].iloc[0]
        lookup[f"Winner Group {letter}"] = winner
        lookup[f"Runner-up Group {letter}"] = runner_up

    for _, row in third_place_ranking.iterrows():
        if row["Qualified"]:
            lookup[f"Third Group {row['Group']}"] = row["Team"]
    return lookup


def parse_third_place_groups(placeholder: str) -> list[str]:
    """``3rd Group A/B/C/D/F`` -> ``['A', 'B', 'C', 'D', 'F']``."""
    match = re.match(r"3rd Group (.+)", placeholder)
    if not match:
        raise ValueError(f"Not a third-place placeholder: {placeholder}")
    return match.group(1).split("/")


def parse_match_number(label: str) -> int | None:
    match = re.search(r"Match (\d+)", label)
    return int(match.group(1)) if match else None


def round_name_for_fixture(score_label: str) -> str:
    match_no = parse_match_number(score_label)
    if match_no is None:
        return "Knockout"
    return KNOCKOUT_ROUND_BY_MATCH.get(match_no, "Knockout")


def collect_third_place_slots(
    knockout_fixtures: pd.DataFrame,
    *,
    home_col: str | None = None,
    away_col: str | None = None,
    score_col: str | None = None,
) -> list[tuple[int, str, list[str]]]:
    """Return Round-of-32 third-place slots as (match_no, placeholder, eligible_groups)."""
    home_col, away_col, score_col = _fixture_column_names(
        knockout_fixtures, home_col, away_col, score_col
    )
    fixtures = knockout_fixtures.copy()
    fixtures["_match_no"] = fixtures[score_col].apply(parse_match_number)
    slots: list[tuple[int, str, list[str]]] = []
    for _, row in fixtures.sort_values("_match_no").iterrows():
        match_no = int(row["_match_no"])
        if KNOCKOUT_ROUND_BY_MATCH.get(match_no) != "Round of 32":
            continue
        for team in (row[home_col], row[away_col]):
            if str(team).startswith("3rd Group "):
                slots.append((match_no, str(team), parse_third_place_groups(str(team))))
    return slots


def assign_third_place_slots(
    slots: list[tuple[int, str, list[str]]],
    third_place_ranking: pd.DataFrame,
) -> dict[str, str]:
    """
    Map each ``3rd Group …`` placeholder to a qualified third-placed team.

    Uses backtracking so all eight slots are filled consistently (greedy
    assignment by match order can fail when eligible groups overlap).
    """
    qualified = third_place_ranking.loc[third_place_ranking["Qualified"]].copy()
    group_to_team = {row["Group"]: row["Team"] for _, row in qualified.iterrows()}
    remaining_groups = set(group_to_team)
    rank_order = {
        row["Group"]: int(row["ThirdPlaceRank"]) for _, row in qualified.iterrows()
    }

    assignment: dict[str, str] = {}

    def backtrack(index: int) -> bool:
        if index == len(slots):
            return True
        _, placeholder, eligible = slots[index]
        candidates = sorted(
            (g for g in eligible if g in remaining_groups),
            key=lambda g: rank_order[g],
        )
        for group in candidates:
            assignment[placeholder] = group_to_team[group]
            remaining_groups.remove(group)
            if backtrack(index + 1):
                return True
            remaining_groups.add(group)
            del assignment[placeholder]
        return False

    if not backtrack(0):
        qualified_groups = sorted(group_to_team)
        slot_summary = [f"{ph} ({'/'.join(elig)})" for _, ph, elig in slots]
        raise ValueError(
            "Could not assign all third-placed teams to Round-of-32 slots. "
            f"Qualified third groups: {qualified_groups}. Slots: {slot_summary}"
        )
    return assignment


def resolve_bracket_team(
    placeholder: str,
    lookup: dict[str, str],
    third_place_assignments: dict[str, str],
    match_results: dict[int, dict[str, str]],
) -> str:
    if placeholder.startswith("Winner Group ") or placeholder.startswith("Runner-up Group "):
        return lookup[placeholder]
    if placeholder.startswith("3rd Group "):
        if placeholder not in third_place_assignments:
            raise ValueError(f"No third-place assignment for {placeholder}")
        return third_place_assignments[placeholder]
    if placeholder.startswith("Winner Match "):
        match_no = int(placeholder.removeprefix("Winner Match ").strip())
        return match_results[match_no]["winner"]
    if placeholder.startswith("Loser Match "):
        match_no = int(placeholder.removeprefix("Loser Match ").strip())
        return match_results[match_no]["loser"]
    raise ValueError(f"Unknown placeholder: {placeholder}")


def group_standings_to_dataframe(group_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(group_tables.values(), ignore_index=True)


def qualification_summary(
    group_tables: dict[str, pd.DataFrame],
    third_place_ranking: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_name, table in group_tables.items():
        letter = group_name.split()[-1]
        for _, team_row in table.iterrows():
            rank = int(team_row["GroupRank"])
            if rank <= 2:
                route = f"Top {rank} in {group_name}"
                qualified = True
            elif rank == 3:
                third_info = third_place_ranking.loc[
                    third_place_ranking["Group"] == letter
                ].iloc[0]
                qualified = bool(third_info["Qualified"])
                route = (
                    f"Best third-placed (#{int(third_info['ThirdPlaceRank'])})"
                    if qualified
                    else "Eliminated in group stage"
                )
            else:
                qualified = False
                route = "Eliminated in group stage"
            rows.append(
                {
                    "Team": team_row["Team"],
                    "Group": group_name,
                    "GroupRank": rank,
                    "Points": int(team_row["Points"]),
                    "GF": int(team_row["GF"]),
                    "GA": int(team_row["GA"]),
                    "GD": int(team_row["GD"]),
                    "Qualified": qualified,
                    "QualificationRoute": route,
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(by=["Qualified", "Group", "GroupRank"], ascending=[False, True, True])
        .reset_index(drop=True)
    )


MatchPredictor = Callable[[str, str], tuple[float, float]]


def _fixture_column_names(
    fixtures_df: pd.DataFrame,
    home_col: str | None,
    away_col: str | None,
    score_col: str | None,
) -> tuple[str, str, str]:
    home = home_col or ("home" if "home" in fixtures_df.columns else "HomeTeam")
    away = away_col or ("away" if "away" in fixtures_df.columns else "AwayTeam")
    score = score_col or ("score" if "score" in fixtures_df.columns else "score")
    return home, away, score


def simulate_knockout_bracket(
    knockout_fixtures: pd.DataFrame,
    lookup: dict[str, str],
    third_place_ranking: pd.DataFrame,
    predict_match: MatchPredictor,
    rng: np.random.Generator | None = None,
    *,
    home_col: str | None = None,
    away_col: str | None = None,
    score_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Play out the official Wikipedia knockout bracket (Matches 73–104).

    Returns (match_results, elimination_standings, champion).
    """
    rng = rng or np.random.default_rng()
    home_col, away_col, score_col = _fixture_column_names(
        knockout_fixtures, home_col, away_col, score_col
    )
    match_results: dict[int, dict[str, str]] = {}
    tournament_rows: list[dict[str, Any]] = []
    elimination_rows: list[tuple[str, str]] = []

    third_place_slots = collect_third_place_slots(
        knockout_fixtures, home_col=home_col, away_col=away_col, score_col=score_col
    )
    third_place_assignments = assign_third_place_slots(third_place_slots, third_place_ranking)

    fixtures = knockout_fixtures.copy()
    fixtures["_match_no"] = fixtures[score_col].apply(parse_match_number)
    fixtures = fixtures.sort_values("_match_no")

    current_round: str | None = None
    for _, row in fixtures.iterrows():
        match_no = int(row["_match_no"])
        round_name = round_name_for_fixture(row[score_col])
        if round_name != current_round:
            current_round = round_name

        home_team = resolve_bracket_team(
            row[home_col], lookup, third_place_assignments, match_results
        )
        away_team = resolve_bracket_team(
            row[away_col], lookup, third_place_assignments, match_results
        )

        home_goals, away_goals = predict_match(home_team, away_team)
        if home_goals > away_goals:
            winner, loser = home_team, away_team
        elif away_goals > home_goals:
            winner, loser = away_team, home_team
        else:
            winner = str(rng.choice([home_team, away_team]))
            loser = away_team if winner == home_team else home_team

        match_results[match_no] = {"winner": winner, "loser": loser}
        tournament_rows.append(
            {
                "Match": match_no,
                "Round": round_name,
                "Team1": home_team,
                "Team2": away_team,
                "Predicted Goals Team1": home_goals,
                "Predicted Goals Team2": away_goals,
                "Winner": winner,
            }
        )
        if round_name == "Final":
            elimination_rows.append((loser, "Runner-up"))
            elimination_rows.append((winner, "Champion"))
        elif round_name == "Third place":
            elimination_rows.append((winner, "Third place"))
            elimination_rows.append((loser, "Fourth place"))
        else:
            elimination_rows.append((loser, round_name))

    champion = match_results[104]["winner"]
    results_df = pd.DataFrame(tournament_rows)
    standings_df = pd.DataFrame(elimination_rows, columns=["Team", "Position"])
    return results_df, standings_df, champion


def fifa_ranks_from_predictions(predictions_df: pd.DataFrame) -> dict[str, float]:
    """Build team -> FIFA rank map from enriched prediction rows when available."""
    ranks: dict[str, float] = {}
    if "HomeFifaRank" not in predictions_df.columns:
        return ranks
    for _, row in predictions_df.iterrows():
        if pd.notna(row.get("HomeFifaRank")):
            ranks[row["HomeTeam"]] = float(row["HomeFifaRank"])
        if pd.notna(row.get("AwayFifaRank")):
            ranks[row["AwayTeam"]] = float(row["AwayFifaRank"])
    return ranks
