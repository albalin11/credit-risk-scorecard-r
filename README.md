# Credit Risk Scorecard Model (R)

## Overview
This project builds a credit risk scorecard to predict the probability of default using the Kaggle dataset **Give Me Some Credit**.  
The goal is to develop an interpretable, stable, and deployment‑ready model for credit risk assessment. Logistic regression is used as the baseline scorecard model, and XGBoost is added as a performance benchmark.

## Dataset
- **Source**: Kaggle – [Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit/data)
- **Training set**: `cs-training.csv` (149,999 rows)  
- **External scoring set**: `cs-test.csv` (101,503 rows)  
- **Target variable**: `SeriousDlqin2yrs` – 1 if the borrower was 90+ days past due or worse within 2 years, 0 otherwise.

### Features
| Variable | Description |
|----------|-------------|
| `RevolvingUtilizationOfUnsecuredLines` | Ratio of total balance to credit limit on revolving lines |
| `age` | Age of borrower in years |
| `NumberOfTime30.59DaysPastDueNotWorse` | Number of times 30‑59 days past due in last 2 years |
| `DebtRatio` | Monthly debt payments / monthly gross income |
| `MonthlyIncome` | Monthly income (contains missing values) |
| `NumberOfOpenCreditLinesAndLoans` | Number of open credit lines and loans |
| `NumberOfTimes90DaysLate` | Number of times 90+ days past due |
| `NumberRealEstateLoansOrLines` | Number of real estate loans or lines |
| `NumberOfTime60.89DaysPastDueNotWorse` | Number of times 60‑89 days past due |
| `NumberOfDependents` | Number of dependents (contains missing values) |

---

## Methodology

### 1. Data Preprocessing
- Remove rows with invalid age (`age <= 0`).
- Split the training data into training (70%) and test (30%) sets.
- Missing values in `MonthlyIncome` (19.8%) and `NumberOfDependents` (2.6%) are imputed using the **median** from the training set.

### 2. WOE Binning and IV Selection
- **Weight of Evidence (WOE)** binning is applied to all continuous variables using the `scorecard` package.
- **Information Value (IV)** is calculated to select features with moderate predictive power (`0.02 < IV < 0.5`).  
  Selected features:
  - `DebtRatio`
  - `RevolvingUtilizationOfUnsecuredLines`
  - `age`
  - `NumberOfOpenCreditLinesAndLoans`
  - `NumberRealEstateLoansOrLines`
  - `NumberOfDependents`

### 3. Logistic Regression Model
- A logistic regression model is fitted on the WOE‑transformed training data using the selected features.
- All coefficients are statistically significant (p < 0.001).

### 4. XGBoost Benchmark
- An XGBoost model is trained on the same WOE features for comparison.
- Hyperparameters: `max_depth = 4`, `eta = 0.1`, `subsample = 0.8`, `colsample_bytree = 0.8`.

### 5. Scorecard Conversion
- The logistic regression coefficients are converted into a points‑based scorecard using the following parameters:
  - Base points: 600
  - Odds at base: 1:15 (good:bad)
  - Points to Double Odds (PDO): 60
- The score for each borrower is calculated as:  
  `Score = A - B * ln(odds)`  
  where `odds = p / (1-p)`, `B = PDO / ln(2)`, and `A = Base + B * ln(odds0)`.

### 6. Model Evaluation
- **AUC** – measures discriminative power.
- **KS** – maximum difference between cumulative good and bad distributions.
- **PSI** – checks score distribution shifts between:
  - Training vs. test set (internal PSI)
  - Training vs. external scoring set (external PSI)

---

## Code Structure
The main R script (`scorecard_script.R`) performs the steps outlined above. Key packages used:
- `scorecard` – WOE binning, IV calculation, scorecard conversion
- `ggplot2` – visualizations
- `ROCR`, `pROC` – performance metrics
- `xgboost` – benchmark model

## How to Reproduce
1. Place `cs-training.csv` and `cs-test.csv` in your working directory.
2. Install required packages:
   ```r
   install.packages(c("scorecard", "ggplot2", "ROCR", "xgboost", "pROC"))
3. Run the script
4. The console will display model summaries, AUC, KS, and PSI values. Plots will appear in the graphics device.

## Results


# Logistic Regression Coefficients (on WOE)

<table>
  <thead>
    <tr>
      <th>Feature</th>
      <th>Estimate</th>
      <th>Std. Error</th>
      <th>z value</th>
      <th>Pr(&gt;|z|)</th>
      <th>Sig.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>(Intercept)</td>
      <td>-2.63645</td>
      <td>0.01432</td>
      <td>-184.09</td>
      <td>&lt; 2e-16</td>
      <td>***</td>
    </tr>
    <tr>
      <td>DebtRatio_woe</td>
      <td>0.76230</td>
      <td>0.05022</td>
      <td>15.179</td>
      <td>&lt; 2e-16</td>
      <td>***</td>
    </tr>
    <tr>
      <td>RevolvingUtilizationOfUnsecuredLines_woe</td>
      <td>0.88987</td>
      <td>0.01434</td>
      <td>62.065</td>
      <td>&lt; 2e-16</td>
      <td>***</td>
    </tr>
    <tr>
      <td>age_woe</td>
      <td>0.50959</td>
      <td>0.02933</td>
      <td>17.372</td>
      <td>&lt; 2e-16</td>
      <td>***</td>
    </tr>
    <tr>
      <td>NumberOfOpenCreditLinesAndLoans_woe</td>
      <td>0.23281</td>
      <td>0.04539</td>
      <td>5.129</td>
      <td>2.91e-07</td>
      <td>***</td>
    </tr>
    <tr>
      <td>NumberRealEstateLoansOrLines_woe</td>
      <td>0.72691</td>
      <td>0.05850</td>
      <td>12.427</td>
      <td>&lt; 2e-16</td>
      <td>***</td>
    </tr>
    <tr>
      <td>NumberOfDependents_woe</td>
      <td>0.35976</td>
      <td>0.07570</td>
      <td>4.752</td>
      <td>2.01e-06</td>
      <td>***</td>
    </tr>
  </tbody>
</table>

## Scorecard Example

| Variable | Bin | Points |
|----------|-----|--------|
| age | [-Inf,40) | -22 |
| age | [40,58) | -6 |
| age | [58,64) | 18 |
| age | [64, Inf) | 45 |
| Base points |  | 594 |

The final score is the sum of base points and points for each bin across all selected variables.

## Performance Metrics

| Model | AUC (Test) | KS |
|------|------------|------|
| Logistic Regression | 0.7806 | 0.4537 |
| XGBoost | 0.7847 | — |

## Population Stability Index (PSI)

| Comparison | PSI | Interpretation |
|------------|------|----------------|
| Train vs. Test | 0.0003 | Very stable |
| Train vs. External | 0.0001 | Very stable |

Both PSI values are far below 0.1, indicating no significant score distribution shift.

## Plots
The script generates four plots:

Score Distribution by Good/Bad – shows how scores separate good and bad borrowers.
![Score Distribution](Score_Distribution.png)

ROC Curve – visualizes the model’s discrimination power.
![ROC Curve](ROC_Curve.png)

PSI Internal (Train vs. Test) – bar chart comparing score distributions.
![PSI Internal](PSI_Internal.png)

PSI External (Train vs. External) – same comparison with the external scoring set.
![PSI External](PSI_External.png)

## Conclusion 
A robust credit scorecard was developed using WOE binning and logistic regression. The model shows strong discriminatory power (AUC = 0.78, KS = 0.45) and stable score distributions across time and populations (PSI < 0.01). The resulting scorecard provides interpretable points per variable bin, suitable for deployment in credit risk assessment. The XGBoost model offers a marginal improvement in AUC, but the logistic model is preferred for its transparency and ease of interpretation.

## Future work
- **Feature Engineering** – Incorporate payment history trends and external credit bureau data.
- **Class Imbalance** – Experiment with SMOTE or cost‑sensitive learning to better capture rare defaults.
- **Model Calibration** – Apply Platt scaling to improve probability estimates for business decision thresholds.
- **Production Deployment** – Containerize the scoring engine and implement automated PSI monitoring.
