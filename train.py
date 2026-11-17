"""Fraud model upgrade: compare models, tune the decision threshold on cost, report PR-AUC.

Usage
  python train.py --data creditcard.csv --out artifacts
  python train.py --synthetic --out artifacts      (quick test, no download needed)

The Kaggle credit card fraud file has columns Time, V1..V28, Amount, Class.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_data(path, synthetic):
    if synthetic:
        X, y = make_classification(n_samples=60000, n_features=28, n_informative=10,
                                   n_redundant=4, weights=[0.9983, 0.0017],
                                   class_sep=1.2, flip_y=0.0005, random_state=7)
        df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
        rng = np.random.default_rng(7)
        df["Time"] = np.sort(rng.integers(0, 172800, len(df)))
        df["Amount"] = np.round(np.exp(rng.normal(3.2, 1.4, len(df))), 2)
        df["Class"] = y
        return df
    return pd.read_csv(path)


def best_threshold(y_true, proba, cost_fn, cost_fp):
    """Pick the threshold that minimizes total cost = cost_fn * FN + cost_fp * FP."""
    thresholds = np.unique(np.quantile(proba, np.linspace(0.5, 0.9999, 600)))
    best_t, best_cost = 0.5, float("inf")
    for t in thresholds:
        pred = proba >= t
        fn = int(((~pred) & (y_true == 1)).sum())
        fp = int((pred & (y_true == 0)).sum())
        cost = cost_fn * fn + cost_fp * fp
        if cost < best_cost:
            best_t, best_cost = float(t), cost
    return best_t, best_cost


def evaluate(y_true, proba, threshold, cost_fn, cost_fp):
    pred = proba >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"threshold": float(threshold), "precision": round(float(precision), 4),
            "recall": round(float(recall), 4), "tp": int(tp), "fp": int(fp),
            "fn": int(fn), "tn": int(tn), "cost": int(cost_fn * fn + cost_fp * fp)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--cost-fn", type=float, default=100.0, help="cost of a missed fraud case")
    ap.add_argument("--cost-fp", type=float, default=1.0, help="cost of a false alarm")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_data(args.data, args.synthetic)
    features = [c for c in df.columns if c != "Class"]
    X, y = df[features], df["Class"].to_numpy()

    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=args.seed)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=args.seed)

    models = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "random_forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample",
                                                n_jobs=-1, random_state=args.seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(class_weight="balanced", random_state=args.seed),
    }

    rows, fitted = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        proba_val = model.predict_proba(X_val)[:, 1]
        rows.append({"model": name,
                     "val_pr_auc": round(average_precision_score(y_val, proba_val), 4),
                     "val_roc_auc": round(roc_auc_score(y_val, proba_val), 4)})
        fitted[name] = model
    results = pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False).reset_index(drop=True)
    best_name = results.loc[0, "model"]
    best = fitted[best_name]

    proba_val = best.predict_proba(X_val)[:, 1]
    t_star, _ = best_threshold(y_val, proba_val, args.cost_fn, args.cost_fp)

    proba_test = best.predict_proba(X_test)[:, 1]
    report = {
        "best_model": best_name,
        "test_pr_auc": round(float(average_precision_score(y_test, proba_test)), 4),
        "test_roc_auc": round(float(roc_auc_score(y_test, proba_test)), 4),
        "fraud_rate_test": round(float(y_test.mean()), 5),
        "at_default_threshold_0.5": evaluate(y_test, proba_test, 0.5, args.cost_fn, args.cost_fp),
        "at_cost_tuned_threshold": evaluate(y_test, proba_test, t_star, args.cost_fn, args.cost_fp),
        "cost_assumptions": {"missed_fraud": args.cost_fn, "false_alarm": args.cost_fp},
    }

    results.to_csv(out / "model_comparison.csv", index=False)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "threshold.json").write_text(json.dumps({"threshold": t_star, "features": features, "model": best_name}, indent=2))
    joblib.dump(best, out / "model.joblib")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        p, r, _ = precision_recall_curve(y_test, proba_test)
        plt.figure(figsize=(5, 4))
        plt.plot(r, p)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-recall curve ({best_name})")
        plt.tight_layout()
        plt.savefig(out / "pr_curve.png", dpi=150)
    except Exception as exc:  # plotting is optional
        print("skipped plot", exc)

    print(results.to_string(index=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
