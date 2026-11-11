# DS311 PBL: Predicting Stock Market Returns (Torgo Ch. 3)
Setup:
    python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
    pip install -r requirements.txt
Run:
    python stock_pipeline.py                  # real ^GSPC data via yfinance
    python stock_pipeline.py --synthetic      # offline smoke test
Outputs: results_signals.csv, results_trading.csv, equity_curves.png
Data: Yahoo Finance ticker ^GSPC (downloaded automatically). Seed fixed at 42.
