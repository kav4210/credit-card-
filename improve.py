"""Systematic fraud-model improvement with honest evaluation.

Protocol (so a gain is real, not noise):
 1. Drop exact duplicate rows (they leak identical transactions across train/test).
 2. Lock away a stratified 20% TEST set. Nothing is tuned on it. It is scored once, at the end.
 3. Choose among candidates by 5-fold stratified CV PR-AUC on the remaining 80% (DEV), comparing candidates on identical folds.
 4. Report the winner on TEST with a bootstrap 95% CI, and sanity-check with a day-1 -> day-2 time split.
"""
import json, sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as AP, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

DATA = sys.argv[1]; SEED = 42
df = pd.read_csv(DATA).drop_duplicates().reset_index(drop=True)
df["hour"] = (df.Time // 3600) % 24
df["Amount_log"] = np.log1p(df.Amount)
BASE = [c for c in df.columns if c not in ("Class", "hour", "Amount_log")]
print(f"rows after dedup={len(df):,} frauds={int(df.Class.sum())} base-rate={df.Class.mean():.5f}", flush=True)

dev, test = train_test_split(df, test_size=0.2, stratify=df.Class, random_state=SEED)
dev, test = dev.reset_index(drop=True), test.reset_index(drop=True)
print(f"DEV {len(dev):,} ({int(dev.Class.sum())} fraud) | TEST {len(test):,} ({int(test.Class.sum())} fraud), locked", flush=True)

def lr():  return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
def rf():  return RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)
def hgb_base(): return HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED)
def hgb_tuned(): return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=500, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0,
                                                      early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, random_state=SEED)
def lgbm(spw=1.0): return lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1,
                                             colsample_bytree=0.7, reg_lambda=1.0, scale_pos_weight=spw, random_state=SEED, n_jobs=2, verbose=-1)
BASE_H = BASE + ["hour"]; ALL = BASE + ["hour", "Amount_log"]
CANDS = {  # name: (builder, features)
    "lr_base": (lr, BASE), "rf_base": (rf, BASE), "hgb_base": (hgb_base, BASE),
    "hgb_tuned": (hgb_tuned, BASE), "hgb_tuned+hour": (hgb_tuned, BASE_H),
    "lgbm": (lgbm, BASE), "lgbm+hour": (lgbm, BASE_H), "lgbm_w24+hour": (lambda: lgbm(24.0), BASE_H),
    "rf+hour": (rf, BASE_H), "lr+hour+log": (lr, ALL)}
skf = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(dev, dev.Class))
y = dev.Class.to_numpy(); oof = {}; res = {}
for name, (mk, feats) in CANDS.items():
    ps = []
    for tr, te in skf:
        m = mk().fit(dev.loc[tr, feats], y[tr]); ps.append(m.predict_proba(dev.loc[te, feats])[:, 1])
    oof[name] = ps; folds = np.array([AP(y[te], p) for (_, te), p in zip(skf, ps)])
    res[name] = folds; print(f"CV {name:18s} PR-AUC {folds.mean():.4f} +/- {folds.std():.4f}", flush=True)

def ens(names):  # average of per-fold rank-normalised scores (rank-averaging is scale-free across model types)
    out = []
    for k, (_, te) in enumerate(skf): out.append(np.mean([rankdata(oof[n][k]) / len(te) for n in names], axis=0))
    return out
ENS = {"ens_lgbm+rf+hgb(+hour)": ["lgbm+hour", "rf+hour", "hgb_tuned+hour"], "ens_lgbm+rf(+hour)": ["lgbm+hour", "rf+hour"], "ens_lgbm+lr(+hour)": ["lgbm+hour", "lr+hour+log"]}
for name, parts in ENS.items():
    ps = ens(parts); oof[name] = ps; folds = np.array([AP(y[te], p) for (_, te), p in zip(skf, ps)]); res[name] = folds
    print(f"CV {name:18s} PR-AUC {folds.mean():.4f} +/- {folds.std():.4f}", flush=True)

best = max(res, key=lambda n: res[n].mean()); b0 = res["rf_base"]
d = res[best] - b0
print(f"\nBEST on CV: {best}  mean {res[best].mean():.4f}  vs original rf_base {b0.mean():.4f}")
print(f"paired fold gain over rf_base: mean {d.mean():+.4f}, wins {int((d>0).sum())}/5, fold diffs {np.round(d,4).tolist()}", flush=True)
json.dump({k: {"mean": round(float(v.mean()), 4), "std": round(float(v.std()), 4)} for k, v in res.items()} | {"_best": best}, open("improve_cv.json", "w"), indent=2)

# ---- one-shot evaluation on locked TEST ----
def fit_predict(names, trn, tst):
    ps = []
    for n in names:
        mk, feats = CANDS[n]; ps.append(mk().fit(trn[feats], trn.Class).predict_proba(tst[feats])[:, 1])
    return ps[0] if len(ps) == 1 else np.mean([rankdata(p) / len(p) for p in ps], axis=0)
parts = ENS.get(best, [best]); yt = test.Class.to_numpy()
p_best, p_base = fit_predict(parts, dev, test), fit_predict(["rf_base"], dev, test)
rng = np.random.default_rng(0); idx = np.arange(len(yt)); bs = []
for _ in range(1000):
    s = rng.choice(idx, len(idx)); 
    if yt[s].sum(): bs.append((AP(yt[s], p_best[s]), AP(yt[s], p_base[s])))
bs = np.array(bs); lo, hi = np.percentile(bs[:, 0], [2.5, 97.5]); dl, dh = np.percentile(bs[:, 0] - bs[:, 1], [2.5, 97.5])
print(f"\nTEST (locked, {int(yt.sum())} frauds)\n  original rf_base : PR-AUC {AP(yt,p_base):.4f}\n  improved {best}: PR-AUC {AP(yt,p_best):.4f}  (95% CI {lo:.3f} to {hi:.3f}), ROC-AUC {roc_auc_score(yt,p_best):.4f}")
print(f"  paired bootstrap gain over rf_base: {AP(yt,p_best)-AP(yt,p_base):+.4f}  (95% CI {dl:+.3f} to {dh:+.3f})", flush=True)

# ---- stricter check: train on day 1, test on day 2 ----
d1, d2 = df[df.Time < 86400], df[df.Time >= 86400]
q1, q0 = fit_predict(parts, d1, d2), fit_predict(["rf_base"], d1, d2)
print(f"\nTIME SPLIT day1->day2 ({int(d2.Class.sum())} frauds in day 2)\n  original rf_base : PR-AUC {AP(d2.Class,q0):.4f}\n  improved {best}: PR-AUC {AP(d2.Class,q1):.4f}", flush=True)
json.dump({"best": best, "test_pr_auc": round(float(AP(yt, p_best)), 4), "test_ci": [round(lo, 3), round(hi, 3)], "base_test_pr_auc": round(float(AP(yt, p_base)), 4),
           "gain_ci": [round(dl, 3), round(dh, 3)], "time_split_best": round(float(AP(d2.Class, q1)), 4), "time_split_base": round(float(AP(d2.Class, q0)), 4)}, open("improve_final.json", "w"), indent=2)
