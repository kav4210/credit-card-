# Credit Card Fraud Detection

Fraud detection on the ULB credit card dataset (284,807 transactions, 492 frauds, 0.17%). Because fraud is rare, models are compared on **precision-recall AUC (PR-AUC)**, not accuracy or ROC-AUC. A random guesser scores about 0.0017 on this metric.

The focus is **honest evaluation**: a result only counts if it survives de-duplication, a locked test set, paired comparisons and a time-split check.

## Result
Tuned **LightGBM reaches 0.854 PR-AUC in 5-fold cross-validation** (0.829 on a locked test set of 95 frauds, bootstrap 95% CI 0.76 to 0.90).

| Model (5-fold CV on the 80% dev set, de-duplicated) | PR-AUC |
|---|---|
| Logistic regression | 0.738 |
| Random forest (original best) | 0.835 |
| Gradient boosting, default (`class_weight=balanced`) | 0.749 |
| Gradient boosting, tuned (no class weight, regularized) | 0.839 |
| **LightGBM** | **0.854** |
| Best ensemble (LightGBM + random forest) | 0.851 |

Full numbers are in `results/improve_cv.json`, `results/improve_final.json` and `results/improve.log`.

## What actually moved the number
- **Dropping `class_weight="balanced"` from gradient boosting** took it from 0.749 to 0.839. Heavy class weighting distorts probability ranking on rare events, which is what PR-AUC measures.
- **LightGBM over the random forest:** +0.019 in CV (better in 4 of 5 folds) and +0.017 on the locked test set (paired bootstrap 95% CI +0.003 to +0.032). Small but real.

## What did not work (and is kept here on purpose)
- **Time-of-day feature:** hour 2 shows a 1.45% fraud rate versus about 0.1% in daytime, but adding it did not help (0.850 with, 0.854 without). The spike is probably one or two attack bursts, not a repeatable pattern.
- **Log-scaled `Amount`:** +0.0005 PR-AUC for logistic regression (3 of 5 folds, fold spread 0.025, i.e. noise) and exactly 0 for tree models, which ignore monotone transforms (`ablation.py`).
- **Ensembling** did not beat LightGBM alone.
- **Time split:** training on day 1 and testing on day 2 gives 0.788 (random forest) versus 0.786 (LightGBM). The LightGBM gain may not hold as the data drifts.

## Evaluation protocol (`improve.py`)
1. Drop 1,081 exact duplicate rows (19 are frauds). Duplicates leak identical transactions across train and test; removing them cost about 0.02 PR-AUC on the original single split.
2. Lock away a stratified 20% test set. Nothing is tuned on it; it is scored once.
3. Choose models by 5-fold stratified CV on the rest, on identical folds, and compare candidates with paired fold differences.
4. Report the winner on the test set with a bootstrap CI, then sanity-check with a day-1 to day-2 split.

## Run
```
# get creditcard.csv (Kaggle, "Credit Card Fraud Detection" by ULB) next to the scripts
pip install -r requirements.txt
python train.py --data creditcard.csv --out artifacts      # 3-model baseline, cost-tuned threshold
python ablation.py --data creditcard.csv                   # log-feature test, 5-fold CV
python improve.py creditcard.csv                           # full improvement experiment (~15 min)
pytest tests                                               # handler tests, no download needed
```

## Cost-based threshold
`train.py` also tunes the decision threshold under a 100-to-1 missed-fraud-to-false-alarm cost assumption (changeable with `--cost-fn` and `--cost-fp`). On the original single split this lifted recall from about 79% to 88%. That split holds only 99 frauds, so precision at the tuned threshold is unstable (62% on one run, 32% after de-duplication). Treat the threshold as an illustration of the method.

## Limits
- Only 473 de-duplicated frauds: fold-to-fold PR-AUC spread (about 0.03) is larger than most model differences, so small gains are uncertain.
- The features are anonymized PCA components, so there is no feature-level interpretation.
- Two days of data from European cardholders in 2013; no claim about other settings.
- **`handler.py` is an AWS Lambda handler that serves the random-forest model from `train.py`, tested locally (4 tests). It has not been deployed to a live Lambda, and it does not serve the LightGBM model from `improve.py`.**
