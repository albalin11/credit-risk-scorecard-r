# Credit Risk Scorecard in R

This project uses the [Give Me Some Credit dataset](https://www.kaggle.com/competitions/GiveMeSomeCredit/data) to estimate two-year serious delinquency risk. A WOE-based logistic regression provides an interpretable scorecard; XGBoost is a fixed-settings benchmark. The analysis also asks whether better ranking translates into better probability estimates and lower observed default rates among approved borrowers.

## Data and method

`cs-training.csv` contains the observed outcome (`SeriousDlqin2yrs`, where 1 means serious delinquency within two years). The supplied `cs-test.csv` has no outcome labels, so it is used only for score-distribution comparison. One record with `age <= 0` is removed from the labelled data. A stratified, seeded 70/30 split leaves **104,999 training** and **45,000 test** records; the external file has **101,503** records.

The training medians fill missing `MonthlyIncome` and `NumberOfDependents` values in all three datasets. The training set alone determines WOE bins, their IV, and the scorecard. Variables are retained when **binned IV > 0.02**. The selected variables and IV are:

| Variable | Binned IV |
|---|---:|
| RevolvingUtilizationOfUnsecuredLines | 1.0470 |
| NumberOfTimes90DaysLate | 0.8407 |
| NumberOfTime30-59DaysPastDueNotWorse | 0.7193 |
| age | 0.2621 |
| NumberOfOpenCreditLinesAndLoans | 0.0789 |
| DebtRatio | 0.0692 |
| MonthlyIncome | 0.0674 |
| NumberRealEstateLoansOrLines | 0.0576 |
| NumberOfDependents | 0.0351 |

Revolving utilization and past-due counts describe credit use and delinquency history that can inform a lending decision. Their high IV warrants review, but is not itself evidence of target leakage. The supplied files do not include observation timestamps, so the historical timing of individual records cannot be independently checked. The 60–89 day past-due count has IV below 0.02 and is excluded.

Logistic regression and XGBoost use the same selected WOE variables. XGBoost has fixed parameters and 100 rounds; the test set is never used for fitting, tuning, or early stopping. Test-set AUC, KS, and Brier score evaluate the models. Lower Brier score means more accurate probability predictions.

The logistic scorecard uses `scorecard::scorecard()` and `scorecard_ply()` with **600 points at bad:good odds of 1:15** and **PDO = 60**. In terms of predicted default probability `p`, the scaling is `600 - 60/log(2) * log((p/(1-p))/(1/15))`. Higher default risk therefore gives a lower score.

## Results

| Model | Test AUC | KS | Brier score | AUC 95% CI (DeLong) |
|---|---:|---:|---:|---:|
| Logistic regression | 0.8539 | 0.5535 | 0.05122 | 0.8467–0.8610 |
| XGBoost | 0.8576 | 0.5626 | 0.05034 | 0.8506–0.8646 |

Both models improve substantially over the previous six-variable specification (AUC 0.6799 and 0.6957), alongside the inclusion of revolving utilization and past-due history. This comparison does not isolate the causal contribution of any one variable. XGBoost's new AUC advantage is **0.0037** (paired DeLong 95% CI **0.0022–0.0052**, p = **9.78 × 10⁻⁷**): statistically detectable in this large sample, but small in practice. XGBoost also has a slightly lower Brier score. The [calibration plot](Outputs/Calibration.png) shows both models close to observed rates across training-defined probability groups. AUC measures ranking, not probability accuracy, so the Brier score and calibration plot address a separate question.

![Test ROC curves](Outputs/ROC_Curve.png)

![Probability calibration](Outputs/Calibration.png)

For an illustrative approval exercise, each cutoff is the 60th, 70th, or 80th percentile of **training** predictions. Test applicants below the cutoff are approved. These are historical-data comparisons, not lending recommendations.

| Model | Target approval | Actual test approval | Default rate among approved |
|---|---:|---:|---:|
| Logistic regression | 60% | 59.88% | 1.37% |
| XGBoost | 60% | 59.92% | 1.33% |
| Logistic regression | 70% | 69.95% | 1.77% |
| XGBoost | 70% | 69.83% | 1.71% |
| Logistic regression | 80% | 79.84% | 2.41% |
| XGBoost | 80% | 79.90% | 2.40% |

The XGBoost groups have slightly lower observed default rates at all three targets; the difference is especially small at 80%. Actual approval rates differ a little because the cutoffs come from training data. The labelled test set's overall default rate is **6.68%**.

![Approval analysis](Outputs/Approval_Analysis.png)

Score PSI uses training-score decile boundaries for both comparisons: **0.00040** for training versus the random holdout, and **0.00031** for training versus the supplied unlabeled file. These small values describe similar score distributions in these files. Neither comparison is a time-based stability test, and outcomes are unavailable for the external file.

![Test score distribution](Outputs/Score_Distribution.png)

## Reproduce

Download `cs-training.csv` and `cs-test.csv` from Kaggle's [Give Me Some Credit dataset](https://www.kaggle.com/c/GiveMeSomeCredit/data) and place them in the project root. From there, install `scorecard`, `ggplot2`, `xgboost`, and `pROC`, then run:

```r
source("Credit_Risk_Scorecard.R")
```

The script prints the selected IV values, sample sizes, model metrics, DeLong intervals and test, PSI, and approval table. It writes four figures to `Outputs/`. Results above were generated with R 4.4.1, `scorecard` 0.4.6, `ggplot2` 4.0.3, `xgboost` 3.2.1.1, and `pROC` 1.19.0.1.
