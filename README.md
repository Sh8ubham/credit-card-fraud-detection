# Credit Card Fraud Detection

An end-to-end machine learning pipeline that detects fraudulent credit card transactions using behavioral and velocity-based features, trained on the Sparkov simulated transactions dataset.

## Overview

Rule-based fraud checks don't scale well against the volume and variety of modern transaction data. This project builds a pipeline that learns fraud patterns directly from historical transactions — how much a cardholder usually spends, how often, at what times, and how the current transaction compares to their recent behavior — and trains a classifier (XGBoost) on top of those signals.

## Dataset

- **Source:** [Sparkov Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) (Kaggle, `kartik2112/fraud-detection`)
- **Size:** ~1.3 million simulated transaction records
- **Note:** The raw CSV is not included in this repo due to size. Download `fraudTrain.csv` / `fraudTest.csv` from the Kaggle link above and place them in a `data/` folder.

## Pipeline

| Stage | Notebook | Description |
|---|---|---|
| 1. EDA | `notebooks/01_eda.ipynb` | Class imbalance, transaction amount distribution, time-of-day patterns, merchant/category patterns, demographic and geographic patterns |
| 2. Feature Engineering | `notebooks/02_feature_engineering.ipynb` | Time-based features, cardholder age, card-level velocity features, rolling-window behavioral features, categorical encoding |
| 3. Modeling | `notebooks/03_modelling.ipynb` | XGBoost classifier training and evaluation |

## Features Engineered

- **Time-based:** hour of transaction, day of week, night-time flag
- **Demographic:** cardholder age at time of transaction
- **Velocity:** running transaction count per card, expanding average spend, deviation from average
- **Rolling-window (10 transactions):** rolling average/median spend and deviation, corrected to avoid full-history contamination
- **Categorical:** merchant frequency encoding, one-hot encoded transaction category

Final feature set: `models/model_columns.pkl` (27 features)

## Model

- **Algorithm:** XGBoost (`XGBClassifier`)
- **Config:** `n_estimators=100`, `eval_metric='logloss'`
- **Artifact:** `models/xgb_fraud_model.pkl`

## Project Structure

```
fraud-detection/
├── data/                          # raw data (not committed, see .gitignore)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modelling.ipynb
├── models/
│   ├── xgb_fraud_model.pkl
│   └── model_columns.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download the dataset from Kaggle and place it under `data/`, then run the notebooks in order (`01` → `02` → `03`).

## Roadmap

- [ ] Extend the pipeline into a Power BI dashboard for visualizing flagged transactions and fraud trends
- [ ] Add model evaluation metrics (precision, recall, ROC-AUC) to this README
- [ ] Package feature engineering into a reusable module for real-time scoring

## Author

Kumar Shubham — B.Tech, Birla Institute of Technology (BIT) Mesra