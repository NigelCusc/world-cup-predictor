# FIFA World Cup 2026 Predictor

Machine learning project that forecasts **FIFA World Cup 2026** results from the group stage through the final. Historical match data (1930–2022) is scraped from Wikipedia, enriched with [Kaggle pre-tournament team features](https://www.kaggle.com/datasets/harrachimustapha/fifa-world-cup-team-dataset), and used to train Random Forest models that predict home and away goals. A tournament simulator applies FIFA qualification rules and runs the official 2026 knockout bracket.

## What it does

| Stage | Description |
|-------|-------------|
| **Data collection** | Scrape historical results and the 2026 fixture list from Wikipedia |
| **Feature engineering** | Merge FIFA ranks, form, squad value, host flags, and related team stats |
| **Model training** | Random Forest regressors predict goals per match (trained on 2002–2022) |
| **Group stage** | Predict 72 group matches across 12 groups (A–L) |
| **Qualification** | Top two per group + 8 best third-placed teams → 32 knockout entrants |
| **Knockout** | Simulate Round of 32 → Final + third-place match (32 knockout fixtures) |

**Total:** 104 matches in the 48-team format.

## Quick start

### 1. Clone and install

```bash
git clone <repo-url>
cd world-cup-predictor
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure (optional)

Copy `.env.example` to `.env` and adjust paths or feature flags. All variables use the `WCP_` prefix (see [Configuration](#configuration)).

### 3. Refresh data (if needed)

```bash
# Historical matches (1930–2022) → Data/clean_fifa_worldcup_matches.csv
python scripts/scrape_historical_matches.py

# 2026 fixtures → Data/clean_fifa_worldcup_fixture.csv
python scripts/scrape_fixtures_2026.py
```

The historical scraper downloads the Kaggle team dataset automatically when missing. Pass `--skip-kaggle` to skip enrichment.

### 4. Group membership

Run `Table_Extraction.ipynb` to build `Data/group_standings_2026.pkl` (teams per group from Wikipedia).

### 5. Run predictions

Open and run **`Fifa_worldcup2026_TournamentPrediction.ipynb`** — trains models, predicts group stage, qualifies teams, and simulates knockouts.

### 6. View results

Open **`Display_2026_Predictions.ipynb`** for group tables, qualification summary, knockout scores, and a bracket chart.

## Project layout

```
world-cup-predictor/
├── config.py                          # Paths, env flags, ITS_COMING_HOME logic
├── scripts/
│   ├── scrape_historical_matches.py   # Wikipedia 1930–2022 + Kaggle merge
│   ├── scrape_fixtures_2026.py        # 2026 schedule from Wikipedia
│   ├── download_kaggle_fifa_team_dataset.py
│   ├── kaggle_team_features.py      # Team feature merge helpers
│   └── tournament_2026.py             # Groups, qualification, knockout bracket
├── Data/                              # CSV outputs and cached datasets
├── tests/
│   └── test_config.py
├── Fifa_worldcup2026_TournamentPrediction.ipynb   # Main pipeline
├── Display_2026_Predictions.ipynb                   # Results viewer
├── Table_Extraction.ipynb                           # Group standings extract
└── Prediction 2026 fifa Scores.ipynb              # Score-model experiments
```

## Output files

| File | Contents |
|------|----------|
| `Data/fifa_worldcup_2026_predictions.csv` | Group-stage predicted scores |
| `Data/fifa_worldcup_2026_group_tables.csv` | Per-group standings |
| `Data/fifa_worldcup_2026_standings.csv` | Qualification summary (48 teams) |
| `Data/predicted_tournament_results.csv` | All 32 knockout match results |
| `Data/final_tournament_standings.csv` | Elimination round per team |

## Configuration

Settings load from `.env` on import (via `python-dotenv`). Common options:

| Variable | Default | Purpose |
|----------|---------|---------|
| `WCP_DATA_DIR` | `Data` | Data directory |
| `WCP_USE_KAGGLE_FEATURES` | `true` | Include Kaggle team features in training |
| `WCP_SKIP_KAGGLE_DOWNLOAD` | `false` | Skip automatic Kaggle dataset download |
| `WCP_VERBOSE` | `false` | Verbose script output |
| `ITS_COMING_HOME` | `false` | England losses → draws; knockout draws → penalties |
| `WCP_REQUEST_DELAY_SECONDS` | `0.15` | Delay between Wikipedia requests |

## 2026 tournament format

| Phase | Details |
|-------|---------|
| Group stage | 12 groups × 4 teams → 72 matches |
| Qualification | 24 automatic (top 2 per group) + 8 best third-placed → 32 teams |
| Knockout | Round of 32 → Round of 16 → QFs → SFs → Third place → Final |

Third-place bracket slots use eligible-group lists from the scraped fixtures. FIFA's full Annex C mapping for all 495 third-place combinations is **not** modelled.

## Model features

Training uses matches from **2002 onward** with:

- FIFA rank and points (home/away and diffs)
- 4-year form, World Cup titles, squad market value, squad age
- Host-nation flags, same-confederation matchups
- Label-encoded team IDs

Models: `RandomForestRegressor` for home goals and away goals separately.

## Tests

```bash
python -m unittest tests/test_config.py
```

## Tech stack

Python · Pandas · NumPy · scikit-learn · BeautifulSoup · Matplotlib · Jupyter

## Limitations

- Predictions are illustrative, not betting advice.
- Knockout ties (except `ITS_COMING_HOME` England cases) are broken at random.
- Wikipedia scraping depends on page structure; re-run scrapers if layouts change.

## License

Open source — use and adapt freely.
