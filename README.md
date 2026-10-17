# Credit Risk Scorecard in R

This project uses the [Give Me Some Credit dataset](https://www.kaggle.com/competitions/GiveMeSomeCredit/data) to estimate two-year serious delinquency risk. A WOE-based logistic regression provides an interpretable scorecard; XGBoost is a fixed-settings benchmark. The analysis also asks whether better ranking translates into better probability estimates and lower observed default rates among approved borrowers.

## Data and method

`cs-training.csv` contains the observed outcome (`SeriousDlqin2yrs`, where 1 means serious delinquency within two years). The supplied `cs-test.csv` has no outcome labels, so it is used only for score-distribution comparison. One record with `age <= 0` is removed from the labelled data. A stratified, seeded 70/30 split leaves **104,999 training** and **45,000 test** records; the external file has **101,503** records.

The training medians fill missing `MonthlyIncome` and `NumberOfDependents` values in all three datasets. The training set alone determines WOE bins, their IV, and the scorecard. Variables are retained when **0.02 < binned IV < 0.5**, following the original project's conservative rule. The selected variables and IV are:

| Variable | Binned IV |
|---|---:|
| age | 0.2621 |
| NumberOfOpenCreditLinesAndLoans | 0.0789 |
| DebtRatio | 0.0692 |
| MonthlyIncome | 0.0674 |
| NumberRealEstateLoansOrLines | 0.0576 |
| NumberOfDependents | 0.0351 |

The upper IV cutoff excludes revolving utilization (1.0470) and two past-due counts (0.8407 and 0.7193). These are plausible credit-risk variables; a high IV alone does **not** establish leakage. This conservative rule therefore gives up predictive information and limits the model's discrimination.

Logistic regression and XGBoost use the same selected WOE variables. XGBoost has fixed parameters and 100 rounds; the test set is never used for fitting, tuning, or early stopping. Test-set AUC, KS, and Brier score evaluate the models. Lower Brier score means more accurate probability predictions.

The logistic scorecard uses `scorecard::scorecard()` and `scorecard_ply()` with **600 points at bad:good odds of 1:15** and **PDO = 60**. In terms of predicted default probability `p`, the scaling is `600 - 60/log(2) * log((p/(1-p))/(1/15))`. Higher default risk therefore gives a lower score.

## Results

| Model | Test AUC | KS | Brier score | AUC 95% CI (DeLong) |
|---|---:|---:|---:|---:|
| Logistic regression | 0.6799 | 0.2826 | 0.06085 | 0.6704–0.6895 |
| XGBoost | 0.6957 | 0.2986 | 0.06038 | 0.6863–0.7051 |

XGBoost's AUC advantage is **0.0158** (paired DeLong 95% CI **0.0121–0.0194**, p = **2.93 × 10⁻¹⁷**). The difference is statistically clear in this large test sample, but its size is modest. XGBoost also has a slightly lower Brier score. The [calibration plot](Outputs/Calibration.png) shows both models reasonably close to observed rates across training-defined probability groups; the logistic model overpredicts somewhat in the highest-risk group. AUC measures ranking, not probability accuracy, so the Brier score and calibration plot address a separate question.

![Test ROC curves](Outputs/ROC_Curve.png)

![Probability calibration](Outputs/Calibration.png)

For an illustrative approval exercise, each cutoff is the 60th, 70th, or 80th percentile of **training** predictions. Test applicants below the cutoff are approved. These are historical-data comparisons, not lending recommendations.

| Model | Target approval | Actual test approval | Default rate among approved |
|---|---:|---:|---:|
| Logistic regression | 60% | 59.70% | 3.84% |
| XGBoost | 60% | 59.50% | 3.57% |
| Logistic regression | 70% | 69.56% | 4.42% |
| XGBoost | 70% | 69.58% | 4.15% |
| Logistic regression | 80% | 79.64% | 4.99% |
| XGBoost | 80% | 79.81% | 4.91% |

The XGBoost groups have slightly lower observed default rates at all three targets. Actual approval rates differ a little because the cutoffs come from training data. The labelled test set's overall default rate is **6.68%**.

![Approval analysis](Outputs/Approval_Analysis.png)

Score PSI uses training-score decile boundaries for both comparisons: **0.00056** for training versus the random holdout, and **0.00025** for training versus the supplied unlabeled file. These small values describe similar score distributions in these files. Neither comparison is a time-based stability test, and outcomes are unavailable for the external file.

![Test score distribution](Outputs/Score_Distribution.png)

## Reproduce

From the repository root, install `scorecard`, `ggplot2`, `xgboost`, and `pROC`, then run:

```r
source("Credit_Risk_Scorecard.R")
```

The script prints the selected IV values, sample sizes, model metrics, DeLong intervals and test, PSI, and approval table. It writes four figures to `Outputs/`. Results above were generated with R 4.4.1, `scorecard` 0.4.6, `ggplot2` 4.0.3, `xgboost` 3.2.1.1, and `pROC` 1.19.0.1.
