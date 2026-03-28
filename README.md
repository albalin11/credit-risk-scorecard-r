# Credit Risk Scorecard Model (R)

![R](https://img.shields.io/badge/Language-R-blue)
![Model](https://img.shields.io/badge/Model-Scorecard-green)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)

## Overview
This project develops a **credit risk scorecard** to estimate the probability of default using the Kaggle dataset **Give Me Some Credit**.  

The goal is to build a model that is:

- **Interpretable** (regulatory-friendly)
- **Stable** (robust across populations)
- **Deployable** (convertible to a scorecard system)

A **logistic regression scorecard** is used as the primary model, with **XGBoost** as a  performance benchmark.

## Dataset
- **Source**: Kaggle – [Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit/data)
- Training set: 149,999 rows
- External scoring set: 101,503 rows
- Target variable: `SeriousDlqin2yrs` (1 = default, 0 = non-default)

## Methodology

### 1. Data Preprocessing
- Removed invalid records (`age <= 0`)
- Train/test split: 70% / 30%
- Missing values:
  -  `MonthlyIncome` (19.8%) → median imputation
  -  `NumberOfDependents` (2.6%) → median imputation
 
Median imputation is used for robustness to outliers and consistency with scorecard interpretability.

### 2. WOE Binning and IV Selection
- Applied **WOE (Weight of Evidence)** binning using the `scorecard` package.
- Enforced **monotonicity** in WOE to ensure:
  - consistent risk ordering  
  - improved model stability  
  - regulatory interpretability  

- Special handling:
  - Missing values treated as separate bins  
  - Extreme values capped where necessary  

- **Information Value (IV)** used for feature selection:

> Selected features satisfy: `0.02 < IV < 0.5`

This range balances predictive power while avoiding:
- weak predictors (low IV)
- overfitting or leakage (very high IV)

#### Selected variables:
  - `DebtRatio`
  - `RevolvingUtilizationOfUnsecuredLines`
  - `age`
  - `NumberOfOpenCreditLinesAndLoans`
  - `NumberRealEstateLoansOrLines`
  - `NumberOfDependents`

### 3. Business Interpretation of Features
Key variables align with domain knowledge in credit risk:

- **RevolvingUtilizationOfUnsecuredLines**  
  → High utilization indicates financial stress and increased default risk  

- **DebtRatio**  
  → Higher debt-to-income ratio implies weaker repayment capacity  

- **age**  
  → Younger borrowers tend to have higher risk due to income instability  

- **NumberOfDependents**  
  → More dependents increase financial burden  

- **Credit line counts**  
  → Reflect credit exposure and financial behavior patterns  

This alignment enhances model interpretability and business trust.

### 4. Logistic Regression Model
- Model trained on WOE-transformed features  
- All coefficients statistically significant (p < 0.001)

Additional checks:
- Multicollinearity assessed using **VIF**
- Model kept simple to preserve interpretability
- Coefficient signs consistent with economic intuition

### 5. XGBoost Benchmark
- Trained on same WOE features  
- Parameters:
  - `max_depth = 4`
  - `eta = 0.1`
  - `subsample = 0.8`
  - `colsample_bytree = 0.8`

**Purpose:**
- Evaluate potential performance gain vs. interpretability trade-off

### 6. Scorecard Construction

The logistic model is converted into a points-based scorecard:

  - Base points: 600
  - Odds at base: 1:15 (good:bad)
  - PDO (Points to Double Odds): 60
    
PDO = 60 is chosen as an industry standard to ensure:
- intuitive scaling  
- easy business interpretation
     
Score formula:  
  `Score = A - B * ln(odds)`  
where:
  - `B = PDO / ln(2)`
  - `A = Base + B * ln(odds0)`
  - `odds = p / (1-p)`

### 7. Model Evaluation

Metrics used:

- **AUC** – measures discriminative power.
- **KS** – maximum difference between cumulative good and bad distributions.
- **PSI** – population stability

PSI is used not only for validation but also for:
- monitoring data drift  
- triggering model retraining in production

## Results

### Model Performance

| Model | AUC (Test) | KS |
|------|------------|------|
| Logistic Regression | 0.7806 | 0.4537 |
| XGBoost | 0.7847 | — |

Although XGBoost shows slightly higher AUC, logistic regression is preferred due to:

- interpretability  
- regulatory compliance  
- ease of deployment as a scorecard  

### Population Stability Index (PSI)

| Comparison | PSI | Interpretation |
|------------|------|----------------|
| Train vs. Test | 0.0003 | Very stable |
| Train vs. External | 0.0001 | Very stable |

Both values < 0.1 indicate no significant distribution shift.

## Scorecard example

| Variable | Bin | Points |
|----------|-----|--------|
| age | [-Inf,40) | -22 |
| age | [40,58) | -6 |
| age | [58,64) | 18 |
| age | [64, Inf) | 45 |
| Base points |  | 594 |

Final score is the sum across all variable bins.

## Deployment Considerations

To simulate production usage:

- Implemented reusable scoring logic for new data  
- Ensured binning consistency between training and scoring  
- Model output can be directly used for:
  - credit approval  
  - risk-based pricing  
  - limit assignment

## Code structure

project/
├── data/
├── src/
│ ├── preprocessing.R
│ ├── binning.R
│ ├── model.R
│ ├── evaluation.R
├── main.R
├── README.md

This modular design improves:
- maintainability  
- reproducibility  
- production readiness  

## Plots
The script generates four plots:

### Score Distribution by Good/Bad – shows how scores separate good and bad borrowers
![Score Distribution](Credit_Risk_Scorecard_R/Outputs/Score_Distribution.png)

Bad samples are concentrated in lower score bands, while good samples are more concentrated in higher score bands, which is consistent with scorecard logic.

### ROC Curve – visualizes the model’s discrimination power
![ROC Curve](Credit_Risk_Scorecard_R/Outputs/ROC_Curve.png)

The Logistic Regression model achieved an AUC of 0.7806, indicating strong discriminative power.

### PSI Internal (Train vs. Test) – bar chart comparing score distributions
![PSI Internal](Credit_Risk_Scorecard_R/Outputs/PSI_Internal.png)

### PSI External (Train vs. External) – same comparison with the external scoring set
![PSI External](Credit_Risk_Scorecard_R/Outputs/PSI_External.png)

Internal PSI = 0.0003 and external PSI = 0.0001, both far below 0.1, suggesting stable population distribution.

## Conclusion 

A robust and interpretable credit scorecard was developed using:

- WOE binning  
- Logistic regression  
- Scorecard transformation  

The model achieves:

- Strong discrimination (AUC = 0.78, KS = 0.45)  
- High stability (PSI < 0.01)  

This makes it suitable for real-world credit risk applications.

## Future work

- Feature engineering (behavioral variables, trends)  
- Class imbalance handling (SMOTE, cost-sensitive learning)  
- Probability calibration (Platt scaling)  
- Reject inference (approved vs. rejected applicants bias)  
- Automated monitoring (PSI dashboard)
