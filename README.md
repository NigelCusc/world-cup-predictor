# Fifa-WorldCup-Data-Analysis-1930-2026

## 🏆 FIFA World Cup 2026 Predictor

A **Machine Learning** project that predicts the **FIFA World Cup 2026** results, from **group stage to the final champion**! ⚽🔥  

This project involves **data scraping, data cleaning, predictive modeling, and knockout round simulation** to forecast match results using historical FIFA World Cup data (1930-2022).

---

## 🚀 Project Overview

🔹 **Scraped Data**: Extracted historical World Cup match results from **Wikipedia (1930-2022)**  
🔹 **Data Cleaning & Structuring**: Processed and formatted data into a structured dataset  
🔹 **Fixture Scraping**: Fetches the full **2026 match schedule (104 fixtures)** from Wikipedia  
🔹 **Match Predictions**: Trained ML models to **predict match outcomes (home & away goals)**  
🔹 **Knockout Simulation**: Simulated each stage, determining the **winner of the tournament**  
🔹 **Data Export**: Outputs match predictions & tournament standings in CSV format  

---

## ⚡ Technologies & Skills Used

✅ **Python** – Core programming language  
✅ **Pandas** – Data cleaning, structuring & manipulation  
✅ **BeautifulSoup** – Web scraping from Wikipedia  
✅ **Scikit-learn** – Machine learning models (Random Forest) for match predictions  
✅ **NumPy** – Mathematical operations & simulations  
✅ **Matplotlib & Seaborn** – Data visualization  
✅ **Jupyter Notebook** – Development & testing environment  

---
## 🎯 Features & Workflow  

### 1️⃣ Scraping & Data Preprocessing  
- Extracts **FIFA World Cup match history (1930-2022)** from Wikipedia  
- Cleans and structures data into a usable format  

Refresh historical match data from Wikipedia:

```bash
source .venv/bin/activate
python scripts/scrape_historical_matches.py
```

Outputs `Data/fifa_worldcup_historical_data.csv` (raw) and `Data/clean_fifa_worldcup_matches.csv` (cleaned + Kaggle pre-tournament features for home/away). Downloads [harrachimustapha/fifa-world-cup-team-dataset](https://www.kaggle.com/datasets/harrachimustapha/fifa-world-cup-team-dataset) when missing. Use `--skip-kaggle` to omit enrichment.

### 2️⃣ 2026 Fixtures  
- Scrapes the official Wikipedia schedule for **2026 FIFA World Cup** (group stage + knockout)  
- Saves to `Data/clean_fifa_worldcup_fixture.csv`  

Refresh 2026 fixtures from Wikipedia:

```bash
source .venv/bin/activate
python scripts/scrape_fixtures_2026.py
```

All scraped CSVs and notebook outputs live under `Data/`.

For 2026 **group membership** (teams A–L), run `Table_Extraction.ipynb` → `Data/group_standings_2026.pkl`.

### 3️⃣ Machine Learning Model Training  
- **Encodes team strengths**  
- Trains **ML models** to predict **home & away goals**  
- Uses **Random Forest Regressor** for score prediction  

### 4️⃣ Tournament Simulation (48-team format)

The 2026 tournament uses an expanded format:

| Phase | Details |
|-------|---------|
| Group stage | **12 groups of 4** → 72 matches |
| Qualification | Top 2 per group (24) + **8 best third-placed** teams → **32** advance |
| Knockout | **Round of 32** → Round of 16 → QFs → SFs → **Third place** → Final |
| Total | **104 matches** |

The main notebook (`Fifa_worldcup2026_TournamentPrediction.ipynb`):

- Predicts **72 group-stage** results only
- Builds **per-group standings** and applies FIFA qualification rules
- Simulates the **official Wikipedia knockout bracket** via `scripts/tournament_2026.py`

Knockout rounds: **Round of 32, Round of 16, Quarterfinals, Semifinals, Third place, Final**.

> Third-place bracket slots use eligible-group lists from the scraped fixtures. FIFA's full Annex C mapping for all 495 third-place combinations is not modelled.

### 5️⃣ View results

Open `Display_2026_Predictions.ipynb` after running the tournament prediction notebook. It shows group-stage match scores, qualification tables, knockout match lists, and a bracket graphic.

### 6️⃣ Data Export  
- `Data/fifa_worldcup_2026_predictions.csv` — group-stage predictions  
- `Data/fifa_worldcup_2026_standings.csv` — qualification summary (48 teams)  
- `Data/fifa_worldcup_2026_group_tables.csv` — per-group tables  
- `Data/predicted_tournament_results.csv` — all 32 knockout matches  
- `Data/final_tournament_standings.csv` — elimination round per team  

---
## 🤝 Contributing
Want to improve the model or add new features? Feel free to fork & contribute!

🔹 Improve ML accuracy with advanced models
🔹 Add expected goals (xG) analysis
🔹 Implement full FIFA Annex C third-place bracket mapping
---
📜 License
This project is open-source!
🔥 Star this repo if you found it useful! 🚀⚽
