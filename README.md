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

For 2026 **group standings tables** (not match fixtures), run `Table_Extraction.ipynb` → `Data/group_standings_2026.pkl`.

### 3️⃣ Machine Learning Model Training  
- **Encodes team strengths**  
- Trains **ML models** to predict **home & away goals**  
- Uses **Random Forest Regressor** for score prediction  

### 4️⃣ Tournament Simulation  
- Predicts **group stage results**  
- Simulates **knockout rounds** (Round of 16, Quarterfinals, Semifinals, Final)  
- Determines **FIFA World Cup 2026 Champion** 🏆  

### 5️⃣ Data Export  
- Saves all results, including **match predictions & tournament standings**  

---
## 🤝 Contributing
Want to improve the model or add new features? Feel free to fork & contribute!

🔹 Improve ML accuracy with advanced models
🔹 Add expected goals (xG) analysis
🔹 Optimize knockout stage simulation
---
📜 License
This project is open-source!
🔥 Star this repo if you found it useful! 🚀⚽
