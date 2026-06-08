"""Project configuration: paths, feature flags, and environment overrides.

Set values in a local ``.env`` file (see ``.env.example``) or export environment
variables before running scripts or notebooks. All env vars use the ``WCP_`` prefix.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent


def _load_env_file(path: Path) -> None:
    """Minimal ``.env`` parser used when python-dotenv is unavailable."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if value.startswith(("'", '"')) and value.endswith(("'", '"')):
            value = value[1:-1]
        os.environ[key] = value


_env_path = PROJECT_ROOT / ".env"
try:
    from dotenv import load_dotenv

    load_dotenv(_env_path, override=True)
except ImportError:
    _load_env_file(_env_path)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).expanduser() if raw else default


# --- Paths ---

DATA_DIR = _env_path("WCP_DATA_DIR", PROJECT_ROOT / "Data")
MODELS_DIR = _env_path("WCP_MODELS_DIR", PROJECT_ROOT / "models")
KAGGLE_DATA_DIR = _env_path(
    "WCP_KAGGLE_DATA_DIR", DATA_DIR / "kaggle_fifa_world_cup_team_dataset"
)
GROUP_STANDINGS_PATH = _env_path(
    "WCP_GROUP_STANDINGS_PATH", DATA_DIR / "group_standings_2026.pkl"
)

# --- Feature flags ---

USE_KAGGLE_FEATURES = _env_bool("WCP_USE_KAGGLE_FEATURES", True)
SKIP_KAGGLE_DOWNLOAD = _env_bool("WCP_SKIP_KAGGLE_DOWNLOAD", False)
VERBOSE = _env_bool("WCP_VERBOSE", False)
ITS_COMING_HOME = _env_bool("ITS_COMING_HOME", False)

# --- Scraping ---

USER_AGENT = os.getenv(
    "WCP_USER_AGENT", "FifaWorldCupAnalysis/1.0 (educational data project)"
)
REQUEST_DELAY_SECONDS = float(os.getenv("WCP_REQUEST_DELAY_SECONDS", "0.15"))

# --- Fun ---

COMING_HOME_TEAM = "England"


def _coming_home_active(its_coming_home: bool | None = None) -> bool:
    """Resolve the effective flag value (explicit arg beats module default)."""
    return ITS_COMING_HOME if its_coming_home is None else its_coming_home


def _coming_home_draw_goals(
    home_goals: float, away_goals: float
) -> tuple[float, float]:
    level = max(home_goals, away_goals)
    return level, level


def _england_would_lose(
    home_team: str, away_team: str, home_goals: float, away_goals: float
) -> bool:
    if home_team == COMING_HOME_TEAM and home_goals < away_goals:
        return True
    if away_team == COMING_HOME_TEAM and away_goals < home_goals:
        return True
    return False


def adjust_coming_home_group_goals(
    home_team: str,
    away_team: str,
    home_goals: int,
    away_goals: int,
    *,
    its_coming_home: bool | None = None,
) -> tuple[int, int]:
    """Turn predicted England losses into draws when ``ITS_COMING_HOME`` is enabled."""
    if not _coming_home_active(its_coming_home):
        return home_goals, away_goals
    if _england_would_lose(home_team, away_team, home_goals, away_goals):
        draw_home, draw_away = _coming_home_draw_goals(
            float(home_goals), float(away_goals)
        )
        return int(draw_home), int(draw_away)
    return home_goals, away_goals


def resolve_coming_home_knockout(
    home_team: str,
    away_team: str,
    home_goals: float,
    away_goals: float,
    rng: np.random.Generator,
    *,
    its_coming_home: bool | None = None,
) -> tuple[str, str, float, float, str | None]:
    """
    Resolve a knockout match when ``ITS_COMING_HOME`` may apply.

    England losses become draws; knockout draws involving England are won on penalties.
    Returns ``(winner, loser, home_goals, away_goals, decided_by)``.
    """
    active = _coming_home_active(its_coming_home)
    if active and _england_would_lose(home_team, away_team, home_goals, away_goals):
        home_goals, away_goals = _coming_home_draw_goals(home_goals, away_goals)

    if home_goals > away_goals:
        return home_team, away_team, home_goals, away_goals, None
    if away_goals > home_goals:
        return away_team, home_team, home_goals, away_goals, None

    if active and COMING_HOME_TEAM in (home_team, away_team):
        winner = COMING_HOME_TEAM
        loser = away_team if winner == home_team else home_team
        return winner, loser, home_goals, away_goals, "Penalties"

    winner = str(rng.choice([home_team, away_team]))
    loser = away_team if winner == home_team else home_team
    return winner, loser, home_goals, away_goals, None
