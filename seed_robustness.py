"""
Seed-robustness experiment (extension #6). Run AFTER extension_experiments.py works.
Re-trains the models that use randomness with several random seeds and reports the
spread (mean, std, min, max) of precision and long-only trading results.
Purpose: show that precision/lift is stable while trading return is fragile.

Run:  python seed_robustness.py              # 5 seeds
      python seed_robustness.py --n-seeds 3
"""
import argparse, warnings
import numpy as np, pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, XGBClassifier
import stock_pipeline as sp
import extension_experiments as ex

warnings.filterwarnings("ignore")

def factories(seed):
    xgb = dict(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
               random_state=seed, verbosity=0, n_jobs=1)
    return {
        "MLP (book NN)": ("reg", lambda: make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(10,), alpha=0.01, max_iter=500, random_state=seed))),
        "XGBoost (ext)": ("reg", lambda: XGBRegressor(**xgb)),
        "XGB-Classifier (ext)": ("clf", lambda: ex.Clf(XGBClassifier(**xgb))),
    }

def walk_forward_one(make, kind, X, y, test_start, feats):
    out = []
    for yr in range(test_start, X.index.year.max() + 1):
        tr = X[X.index.year < yr].iloc[:-sp.HORIZON]          # 10-day purge gap
        te = X[X.index.year == yr]
        if te.empty: continue
        ytr = y.loc[tr.index]; k = ytr.notna()
        m = make()
        m.fit(tr.loc[k, feats], ytr[k] if kind == "reg" else ytr[k].astype(int))
        out.append(pd.Series(m.predict(te[feats]), index=te.index))
    return pd.concat(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--start", default="1995-01-01")
    ap.add_argument("--test-start", type=int, default=2005)
    ap.add_argument("--n-seeds", type=int, default=5)
    a = ap.parse_args()

    df = sp.load_data(a.start, a.synthetic)
    T = sp.make_target(df)
    X = sp.make_features(df).dropna()
    df, T = df.loc[X.index], T.loc[X.index]
    feats = ex.select_features(X, T, a.test_start)
    ysig = sp.to_signal(T).where(T.notna())

    rows = []
    for seed in range(a.n_seeds):
        for name, (kind, make) in factories(seed).items():
            p = walk_forward_one(make, kind, X, T if kind == "reg" else ysig,
                                 a.test_start, feats)
            close = df.loc[p.index, "Close"]
            true = sp.to_signal(T.loc[p.index].dropna())
            s = sp.to_signal(p, sp.THRESH) if kind == "reg" else p.astype(int)
            prec = sp.signal_metrics(true, s.loc[true.index])["buy_prec"]
            row = {"model": name, "seed": seed, "buy_prec": prec}
            for cb in (0, 10):
                r, n = sp.simulate(close, s.clip(lower=0), cost_bps=cb)
                pf = sp.perf(r, n)
                row[f"ret_%_{cb}bps"] = pf["total_ret_%"]
                row[f"sharpe_{cb}bps"] = pf["sharpe"]
            rows.append(row)
            print(f"seed {seed} | {name}: buy_prec={prec:.3f}  long-only ret={row['ret_%_0bps']:.1f}%")

    res = pd.DataFrame(rows)
    res.to_csv("ext6_seed_runs.csv", index=False)
    summ = res.drop(columns="seed").groupby("model").agg(["mean", "std", "min", "max"]).round(3)
    print("\n=== Seed robustness (long-only) ===\n", summ.T.to_string())
    summ.to_csv("ext6_seed_robustness.csv")

if __name__ == "__main__":
    main()