# DS311 PBL: Predicting Stock Market Returns (Torgo, Data Mining with R, Ch. 3)

Baseline: the book's workflow (target T, technical indicators, random-forest feature selection,
SVM + neural network, walk-forward evaluation, trading simulation) re-implemented in Python.
Extension: long-only trading, extra algorithms (RandomForest, XGBoost, direct classifiers),
10 bps transaction costs, threshold sensitivity, random-signal permutation test, sub-period analysis.

## Setup
```
python -m venv venv
venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

## Run order
```
python stock_pipeline.py            # baseline  -> results_signals.csv, results_trading.csv, equity_curves.png
python extension_experiments.py     # extension -> ext1..ext5_*.csv
python stock_pipeline.py --synthetic   # optional offline smoke test
python seed_robustness.py   # ext6: seed robustness
```

## Data
S&P 500 index (Yahoo Finance ticker ^GSPC), daily Open/High/Low/Close/Volume, downloaded with the
`yfinance` library. A frozen copy is stored in `data/sp500.csv` (1995-01-03 to 2026-11-11) and is used
automatically if present, so results are reproducible without internet. Delete it to re-download.

## Reproducibility notes
Random seed 42. SVR, MLP and Random Forest results reproduce exactly across machines. XGBoost-based
results can differ slightly between XGBoost versions (tested with xgboost 3.x), and the XGB-Classifier's
trading return is especially sensitive to this, so compare precision/lift rather than exact profits.