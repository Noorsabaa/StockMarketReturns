"""
DS311 PBL - Predicting Stock Market Returns (Torgo, Data Mining with R, Ch. 3)
Baseline : book workflow (target T, indicators, RF feature selection, SVM + NN,
           walk-forward evaluation, trading simulation)
Extension: (a) XGBoost vs the book's models, same features/splits
           (b) transaction costs in the trading simulation

Run:  python stock_pipeline.py                # real S&P 500 data (needs internet)
      python stock_pipeline.py --synthetic    # offline smoke test
"""
import argparse, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
SEED = 42
HORIZON = 10          # days ahead used by the target (book: 10)
PCT = 0.025           # book: only moves beyond +-2.5% count
THRESH = 0.1          # book: buy if T>0.1, sell if T<-0.1
N_FEATURES = 10


# ---------------------------------------------------------------- 1. data
def load_data(start, synthetic=False):
    if synthetic:
        rng = np.random.default_rng(SEED)
        n = 6000
        c = 1000 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
        idx = pd.bdate_range("2000-01-03", periods=n)
        h = c * (1 + rng.uniform(0, 0.01, n)); l = c * (1 - rng.uniform(0, 0.01, n))
        o = np.r_[c[0], c[:-1]]
        return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c,
                             "Volume": rng.integers(1e8, 5e8, n)}, index=idx)
    import yfinance as yf
    df = yf.download("^GSPC", start=start, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


# ------------------------------------------------- 2. target (book's T)
def make_target(df):
    c = df["Close"]
    p = (df["High"] + df["Low"] + df["Close"]) / 3
    fut = pd.concat([p.shift(-k) for k in range(1, HORIZON + 1)], axis=1)
    v = fut.sub(c, axis=0).div(c, axis=0)
    t = v.where(v.abs() > PCT, 0).sum(axis=1)
    t[v.isna().any(axis=1)] = np.nan          # last 10 rows have no target
    return t

def to_signal(t, thr=THRESH):
    s = pd.Series(0, index=t.index)
    s[t > thr] = 1
    s[t < -thr] = -1
    return s


# ---------------------------------------- 3. technical indicators (features)
def make_features(df):
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"].replace(0, np.nan)
    f = pd.DataFrame(index=df.index)
    r = c.pct_change()
    for k in (1, 5, 10, 20):
        f[f"ret{k}"] = c.pct_change(k)
    d = c.diff()
    rs = d.clip(lower=0).rolling(14).mean() / (-d.clip(upper=0)).rolling(14).mean()
    f["rsi14"] = 100 - 100 / (1 + rs)
    ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    macd = (ema12 - ema26) / c
    f["macd"] = macd
    f["macd_hist"] = macd - macd.ewm(span=9).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["atr14"] = tr.rolling(14).mean() / c
    m, s = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_pctb"] = (c - (m - 2 * s)) / (4 * s)
    f["c_sma50"] = c / c.rolling(50).mean() - 1
    f["sma10_50"] = c.rolling(10).mean() / c.rolling(50).mean() - 1
    f["vol20"] = r.rolling(20).std()
    f["stoch14"] = (c - l.rolling(14).min()) / (h.rolling(14).max() - l.rolling(14).min())
    f["volratio"] = v / v.rolling(20).mean()
    return f.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------- 4. models
def get_models():
    return {
        "SVR (book)": make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.01, gamma="scale")),
        "MLP (book NN)": make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(10,), alpha=0.01, max_iter=500, random_state=SEED)),
        "XGBoost (ext)": XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                      subsample=0.8, random_state=SEED, verbosity=0),
    }


# ----------------------------------------- 5. walk-forward (growing window)
def walk_forward(X, y, test_start_year, thr=THRESH):
    """Yearly refit. Train = everything before the test year minus a 10-day
    purge gap (target overlaps the future). Scaling lives inside the pipeline,
    so it is fit on the training window only."""
    years = range(test_start_year, X.index.year.max() + 1)
    first_train = X[X.index.year < test_start_year].iloc[:-HORIZON]
    first_y = y.loc[first_train.index]
    ok = first_y.notna()
    rf = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
    rf.fit(first_train[ok], first_y[ok])             # selection on first window ONLY
    feats = list(pd.Series(rf.feature_importances_, X.columns)
                 .sort_values(ascending=False).index[:N_FEATURES])
    print("Selected features:", feats)

    preds = {name: [] for name in get_models()}
    for yr in years:
        tr = X[X.index.year < yr].iloc[:-HORIZON]
        te = X[X.index.year == yr]
        ytr = y.loc[tr.index]; k = ytr.notna()
        if te.empty:
            continue
        for name, model in get_models().items():
            model.fit(tr.loc[k, feats], ytr[k])
            preds[name].append(pd.Series(model.predict(te[feats]), index=te.index))
    return {n: pd.concat(p) for n, p in preds.items()}, feats


# ------------------------------------------------------------ 6. evaluation
def signal_metrics(true_sig, pred_sig):
    out = {}
    for nm, lab in (("buy", 1), ("sell", -1)):
        tp = ((pred_sig == lab) & (true_sig == lab)).sum()
        pp, ap = (pred_sig == lab).sum(), (true_sig == lab).sum()
        out[f"{nm}_prec"] = tp / pp if pp else np.nan
        out[f"{nm}_rec"] = tp / ap if ap else np.nan
        out[f"{nm}_n"] = int(pp)
    return out

def simulate(close, sig, cost_bps=0.0, tp=0.10, sl=0.10, max_hold=HORIZON):
    """One position at a time, full capital, long on buy / short on sell.
    Signal at close of day i -> position earns returns from day i+1 onward.
    Exit on take-profit, stop-loss or after max_hold days."""
    r = close.pct_change().fillna(0).values
    s = sig.values; cost = cost_bps / 1e4
    rets = np.zeros(len(r)); pos = held = n_trades = 0; cum = 0.0
    for i in range(len(r)):
        if pos != 0:
            day = pos * r[i]; rets[i] += day
            cum = (1 + cum) * (1 + day) - 1; held += 1
            if cum >= tp or cum <= -sl or held >= max_hold:
                rets[i] -= cost; pos = 0
        if pos == 0 and s[i] != 0:
            pos, held, cum = int(s[i]), 0, 0.0
            rets[i] -= cost; n_trades += 1
    return pd.Series(rets, index=close.index), n_trades

def perf(rets, n_trades):
    eq = (1 + rets).cumprod()
    yrs = len(rets) / 252
    return {"total_ret_%": 100 * (eq.iloc[-1] - 1),
            "ann_ret_%": 100 * (eq.iloc[-1] ** (1 / yrs) - 1),
            "sharpe": rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else np.nan,
            "max_dd_%": 100 * (eq / eq.cummax() - 1).min(),
            "trades": n_trades}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--start", default="1995-01-01")
    ap.add_argument("--test-start", type=int, default=2005)
    ap.add_argument("--thr", type=float, default=THRESH)
    a = ap.parse_args()

    df = load_data(a.start, a.synthetic)
    T = make_target(df)
    X = make_features(df).dropna()
    df, T = df.loc[X.index], T.loc[X.index]
    print(f"Data: {df.index[0].date()} -> {df.index[-1].date()}  ({len(df)} rows)")

    preds, feats = walk_forward(X, T, a.test_start, a.thr)
    test_idx = next(iter(preds.values())).index
    close = df.loc[test_idx, "Close"]
    true_sig = to_signal(T.loc[test_idx].dropna(), a.thr)
    print("True signal counts:", true_sig.value_counts().to_dict())

    sig_rows, perf_rows, curves = [], [], {}
    for name, p in preds.items():
        ps = to_signal(p, a.thr)
        sig_rows.append({"model": name, **signal_metrics(true_sig, ps.loc[true_sig.index])})
        for cb in (0, 10):                              # extension: costs (bps/side)
            rets, n = simulate(close, ps, cost_bps=cb)
            perf_rows.append({"model": name, "cost_bps": cb, **perf(rets, n)})
            if cb == 0:
                curves[name] = (1 + rets).cumprod()
    bh = close.pct_change().fillna(0)
    perf_rows.append({"model": "Buy & Hold", "cost_bps": 0, **perf(bh, 1)})
    curves["Buy & Hold"] = (1 + bh).cumprod()

    sigdf = pd.DataFrame(sig_rows).set_index("model").round(3)
    perfdf = pd.DataFrame(perf_rows).set_index(["model", "cost_bps"]).round(2)
    print("\n=== Signal quality (precision / recall) ===\n", sigdf)
    print("\n=== Trading results ===\n", perfdf)
    sigdf.to_csv("results_signals.csv"); perfdf.to_csv("results_trading.csv")

    plt.figure(figsize=(9, 5))
    for n, c in curves.items():
        plt.plot(c.index, c.values, label=n)
    plt.yscale("log"); plt.legend(); plt.title("Equity curves (no costs)")
    plt.tight_layout(); plt.savefig("equity_curves.png", dpi=150)


if __name__ == "__main__":
    main()
