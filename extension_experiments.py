import argparse, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import stock_pipeline as sp

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


class Clf:
    """Wraps a classifier: handles labels {-1,0,1} and balances class weights."""
    def __init__(self, m): self.m = m
    def fit(self, X, y):
        self.cls = np.unique(y)
        yy = np.searchsorted(self.cls, y)
        self.m.fit(X, yy, sample_weight=compute_sample_weight("balanced", yy))
        return self
    def predict(self, X):
        return self.cls[np.asarray(self.m.predict(X)).astype(int)]


def reg_models():
    m = sp.get_models()                                   # SVR, MLP, XGBoost
    m["RandomForest (ext)"] = RandomForestRegressor(
        n_estimators=200, min_samples_leaf=5, random_state=sp.SEED, n_jobs=-1)
    return m

def clf_models():
    return {
        "XGB-Classifier (ext)": Clf(XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
            random_state=sp.SEED, verbosity=0)),
        "RF-Classifier (ext)": Clf(RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, random_state=sp.SEED, n_jobs=-1)),
    }


def select_features(X, T, test_start):
    tr = X[X.index.year < test_start].iloc[:-sp.HORIZON]
    y = T.loc[tr.index]; k = y.notna()
    rf = RandomForestRegressor(n_estimators=200, random_state=sp.SEED, n_jobs=-1)
    rf.fit(tr[k], y[k])
    feats = list(pd.Series(rf.feature_importances_, X.columns)
                 .sort_values(ascending=False).index[:sp.N_FEATURES])
    print("Selected features:", feats)
    return feats

def walk_forward(X, y, test_start, make_models, feats):
    preds = {n: [] for n in make_models()}
    for yr in range(test_start, X.index.year.max() + 1):
        tr = X[X.index.year < yr].iloc[:-sp.HORIZON]       # 10-day purge gap
        te = X[X.index.year == yr]
        if te.empty: continue
        ytr = y.loc[tr.index]; k = ytr.notna()
        for name, model in make_models().items():
            yk = ytr[k] if y.dtype == float and name in reg_models() else ytr[k].astype(int)
            model.fit(tr.loc[k, feats], yk)
            preds[name].append(pd.Series(model.predict(te[feats]), index=te.index))
    return {n: pd.concat(p) for n, p in preds.items()}

def to_sig(kind, p, thr):
    return sp.to_signal(p, thr) if kind == "reg" else p.astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--start", default="1995-01-01")
    ap.add_argument("--test-start", type=int, default=2005)
    ap.add_argument("--n-random", type=int, default=200)
    a = ap.parse_args()

    df = sp.load_data(a.start, a.synthetic)
    T = sp.make_target(df)
    X = sp.make_features(df).dropna()
    df, T = df.loc[X.index], T.loc[X.index]
    feats = select_features(X, T, a.test_start)
    ysig = sp.to_signal(T).where(T.notna())

    preds = {}
    for n, p in walk_forward(X, T, a.test_start, reg_models, feats).items():
        preds[n] = ("reg", p)
    for n, p in walk_forward(X, ysig, a.test_start, clf_models, feats).items():
        preds[n] = ("clf", p)

    test_idx = next(iter(preds.values()))[1].index
    close = df.loc[test_idx, "Close"]
    true = sp.to_signal(T.loc[test_idx].dropna())          # fixed true signals (book thr)
    base_buy, base_sell = (true == 1).mean(), (true == -1).mean()
    print(f"Base rates in test period: buy={base_buy:.3f}  sell={base_sell:.3f}")
    bh = close.pct_change().fillna(0)

    # ---- 1. more algorithms + lift + IC
    rows = []
    for n, (kind, p) in preds.items():
        s = to_sig(kind, p, sp.THRESH)
        m = sp.signal_metrics(true, s.loc[true.index])
        m["buy_lift"] = m["buy_prec"] / base_buy
        m["sell_lift"] = m["sell_prec"] / base_sell
        m["IC_spearman"] = (spearmanr(p.loc[true.index], T.loc[true.index])[0]
                            if kind == "reg" else np.nan)
        rows.append({"model": n, **m})
    e1 = pd.DataFrame(rows).set_index("model").round(3)
    print("\n=== 1. Signal quality incl. lift over base rate and rank-IC ===\n", e1)
    e1.to_csv("ext1_signal_quality.csv")

    # ---- 2. long-short vs long-only, with/without costs
    rows, sigs = [], {}
    for n, (kind, p) in preds.items():
        s = to_sig(kind, p, sp.THRESH); sigs[n] = s
        for mode, ss in (("long-short", s), ("long-only", s.clip(lower=0))):
            for cb in (0, 10):
                r, nt = sp.simulate(close, ss, cost_bps=cb)
                rows.append({"model": n, "mode": mode, "cost_bps": cb, **sp.perf(r, nt)})
    rows.append({"model": "Buy & Hold", "mode": "-", "cost_bps": 0, **sp.perf(bh, 1)})
    e2 = pd.DataFrame(rows).set_index(["model", "mode", "cost_bps"]).round(2)
    print("\n=== 2. Long-short vs long-only ===\n", e2)
    e2.to_csv("ext2_long_short_vs_long_only.csv")

    # ---- 3. threshold sensitivity (regressors; true signals fixed)
    rows = []
    for n, (kind, p) in preds.items():
        if kind != "reg": continue
        for thr in (0.02, 0.03, 0.05, 0.075, 0.1):
            s = sp.to_signal(p, thr)
            m = sp.signal_metrics(true, s.loc[true.index])
            r, nt = sp.simulate(close, s.clip(lower=0))
            rows.append({"model": n, "pred_thr": thr, "buy_n": m["buy_n"],
                         "buy_prec": m["buy_prec"], "buy_lift": m["buy_prec"] / base_buy,
                         "sharpe_long_only": sp.perf(r, nt)["sharpe"]})
    e3 = pd.DataFrame(rows).set_index(["model", "pred_thr"]).round(3)
    print("\n=== 3. Threshold sensitivity (long-only) ===\n", e3)
    e3.to_csv("ext3_threshold_sensitivity.csv")

    # ---- 4. random-signal baseline (permutation test)
    rng = np.random.default_rng(sp.SEED)
    rows = []
    for n, s in sigs.items():
        act_prec = sp.signal_metrics(true, s.loc[true.index])["buy_prec"]
        act_sh = sp.perf(*sp.simulate(close, s.clip(lower=0)))["sharpe"]
        sv, st = s.loc[true.index].values, true.values
        rp, rs = [], []
        for _ in range(a.n_random):
            sh = rng.permutation(sv)
            rp.append(((sh == 1) & (st == 1)).sum() / max((sh == 1).sum(), 1))
            rs.append(sp.perf(*sp.simulate(close, pd.Series(rng.permutation(s.values),
                                                            index=close.index).clip(lower=0)))["sharpe"])
        rows.append({"model": n, "buy_prec": act_prec,
                     "random_prec_mean": np.mean(rp),
                     "p_value_prec": (np.sum(np.array(rp) >= act_prec) + 1) / (len(rp) + 1),
                     "sharpe_long_only": act_sh, "random_sharpe_mean": np.nanmean(rs),
                     "p_value_sharpe": (np.sum(np.array(rs) >= act_sh) + 1) / (len(rs) + 1)})
    e4 = pd.DataFrame(rows).set_index("model").round(3)
    print("\n=== 4. Better than random? (permutation test) ===\n", e4)
    e4.to_csv("ext4_random_baseline.csv")

    # ---- 5. sub-period robustness (long-only, no cost)
    periods = {"2005-2012": (2005, 2012), "2013-2019": (2013, 2019), "2020-end": (2020, 3000)}
    rows = []
    for n, s in list(sigs.items()) + [("Buy & Hold", None)]:
        r = bh if s is None else sp.simulate(close, s.clip(lower=0))[0]
        for pn, (y0, y1) in periods.items():
            rr = r[(r.index.year >= y0) & (r.index.year <= y1)]
            if len(rr) < 30: continue
            pf = sp.perf(rr, 0); pf.pop("trades")
            rows.append({"model": n, "period": pn, **pf})
    e5 = pd.DataFrame(rows).set_index(["model", "period"]).round(2)
    print("\n=== 5. Sub-period robustness (long-only) ===\n", e5)
    e5.to_csv("ext5_subperiods.csv")


if __name__ == "__main__":
    main()
