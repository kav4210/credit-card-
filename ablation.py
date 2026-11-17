"""Does a log-scaled Amount feature actually help? 5-fold stratified CV PR-AUC, with vs without.

Usage:  python ablation.py --data creditcard.csv
        python ablation.py --synthetic            (smoke test only, numbers are NOT yours)

Why CV and not one split: the Kaggle file has only 492 frauds, so a single 20% test split holds ~98 and
PR-AUC swings a lot between splits. We report mean and std across folds so a 'lift' is not just noise.
Expect lift for logistic regression (linear in the features) and ~0 for trees (they ignore monotone transforms).
"""
import argparse, json
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def synthetic():
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=40000, n_features=28, n_informative=10, weights=[0.995, 0.005], class_sep=1.2, random_state=7)
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    rng = np.random.default_rng(7)
    df["Time"] = np.sort(rng.integers(0, 172800, len(df))); df["Amount"] = np.round(np.exp(rng.normal(3.2, 1.4, len(df))), 2); df["Class"] = y
    return df

def make(name):
    if name == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    return HistGradientBoostingClassifier(class_weight="balanced", random_state=42)

def cv_pr_auc(df, name, use_log, folds=5, seed=42):
    X = df.drop(columns=["Class"]).copy(); y = df["Class"].to_numpy()
    if use_log: X["Amount_log"] = np.log1p(X["Amount"])        # keeps raw Amount too, adds the log version
    scores = []
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        m = make(name).fit(X.iloc[tr], y[tr]); scores.append(average_precision_score(y[te], m.predict_proba(X.iloc[te])[:, 1]))
    return np.array(scores)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data"); ap.add_argument("--synthetic", action="store_true"); ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args(); df = synthetic() if a.synthetic else pd.read_csv(a.data)
    print(f"rows={len(df):,} frauds={int(df.Class.sum())} fraud_rate={df.Class.mean():.5f} (random-guess PR-AUC = fraud rate)")
    out = {}
    for name in ["logistic_regression", "hist_gradient_boosting"]:
        base, logv = cv_pr_auc(df, name, False, a.folds), cv_pr_auc(df, name, True, a.folds)
        d = logv - base
        out[name] = {"pr_auc_raw_mean": round(base.mean(), 4), "pr_auc_raw_std": round(base.std(), 4),
                     "pr_auc_with_log_mean": round(logv.mean(), 4), "pr_auc_with_log_std": round(logv.std(), 4),
                     "lift_abs": round(d.mean(), 4), "lift_rel_pct": round(100 * d.mean() / base.mean(), 1),
                     "folds_where_log_wins": f"{int((d > 0).sum())}/{a.folds}"}
        print(name, json.dumps(out[name]))
    json.dump(out, open("ablation_results.json", "w"), indent=2); print("wrote ablation_results.json")
if __name__ == "__main__": main()
