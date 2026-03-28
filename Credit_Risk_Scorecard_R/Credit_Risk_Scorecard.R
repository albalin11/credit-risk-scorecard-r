# CREDIT RISK SCORECARD PROJECT

# Load required packages
# install.packages(c("scorecard","ggplot2","ROCR","xgboost","pROC"))

library(scorecard)   
library(ggplot2)     
library(ROCR) 
library(xgboost)
library(pROC)

set.seed(42)

# ---- 1. Load data ----
df <- read.csv("cs-training.csv", row.names = 1)  

head(df)
table(df$SeriousDlqin2yrs)
prop.table(table(df$SeriousDlqin2yrs))

colSums(is.na(df)) / nrow(df)

# Remove rows with invalid age
df <- df[df$age > 0, ]

# ---- 2. Train/test split ----
train_idx <- sample(seq_len(nrow(df)), size = 0.7 * nrow(df))
train_raw <- df[train_idx, ]
test_raw <- df[-train_idx, ]

# ---- 3. Missing value handling (based on training) ----
median_income <- median(train_raw$MonthlyIncome, na.rm = TRUE)
median_dependents <- median(train_raw$NumberOfDependents, na.rm = TRUE)

train_raw$MonthlyIncome[is.na(train_raw$MonthlyIncome)] <- median_income
test_raw$MonthlyIncome[is.na(test_raw$MonthlyIncome)] <- median_income

train_raw$NumberOfDependents[is.na(train_raw$NumberOfDependents)] <- median_dependents
test_raw$NumberOfDependents[is.na(test_raw$NumberOfDependents)] <- median_dependents

# ---- 4. WOE binning and IV screening on TRAIN only ----
features <- setdiff(names(train_raw), "SeriousDlqin2yrs")

bins <- woebin(train_raw, y = "SeriousDlqin2yrs", x = features, 
               bin_num_limit = 5,   
               positive = "1",  
               no_cores = 1)        

bins$age

woebin_plot(bins)

iv_df <- iv(train_raw, y = "SeriousDlqin2yrs", x = features)
iv_df <- as.data.frame(iv_df)
print(iv_df)

selected_features <- iv_df$variable[iv_df$info_value > 0.02 & iv_df$info_value < 0.5]

# Fallback in case selection is empty
if (length(selected_features) == 0) {
  selected_features <- features
}

print(selected_features)

# Convert training and testing data to WOE values
train_woe <- woebin_ply(train_raw, bins)
test_woe  <- woebin_ply(test_raw, bins)

head(train_woe)

# ---- 5. Logistic Regression ----
selected_features_woe <- paste0(selected_features, "_woe")

selected_features_woe <- intersect(selected_features_woe, names(train_woe))

# Fallback if needed
if (length(selected_features_woe) == 0) {
  selected_features_woe <- grep("_woe$", names(train_woe), value = TRUE)
}

lr_formula <- as.formula(
  paste(
    "SeriousDlqin2yrs ~",
    paste(paste0("`", selected_features_woe, "`"), collapse = " + ")
  )
)

model <- glm(lr_formula, data = train_woe, family = binomial(link = "logit"))
summary(model)

# ---- 6. XGBoost benchmark ----
train_woe <- as.data.frame(train_woe)
test_woe  <- as.data.frame(test_woe)
train_matrix <- as.matrix(train_woe[, selected_features_woe])
test_matrix  <- as.matrix(test_woe[, selected_features_woe])

train_label <- train_woe$SeriousDlqin2yrs
test_label  <- test_woe$SeriousDlqin2yrs

dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)
dtest  <- xgb.DMatrix(data = test_matrix, label = test_label)

params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 4,
  eta = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8
)

xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  evals = list(train = dtrain, test = dtest),
  verbose = 0
)

xgb_pred <- predict(xgb_model, dtest)
xgb_auc <- auc(test_label, xgb_pred)
print(paste("XGBoost AUC =", round(as.numeric(xgb_auc), 4)))

# ---- 7. Scorecard conversion ---- 
card <- scorecard(bins, model, 
                  points0 = 600,   
                  odds0 = 1/15,    
                  pdo = 60)        
print(card)

train_prob <- predict(model, newdata = train_woe, type = "response")
test_prob  <- predict(model, newdata = test_woe,  type = "response")

prob_to_score <- function(prob, PDO = 60, Base = 600, odds0 = 1/15) {
  B <- PDO / log(2)
  A <- Base + B * log(odds0)
  odds <- prob / (1 - prob)
  score <- A - B * log(odds)
  return(round(score, 0))
}

train_score <- prob_to_score(train_prob, PDO = 60, Base = 600, odds0 = 1/15)
test_score  <- prob_to_score(test_prob,  PDO = 60, Base = 600, odds0 = 1/15)

train_woe$Score <- train_score
test_woe$Score  <- test_score

head(test_woe$Score)

# ---- 8. External scoring ----
df_test_ext <- read.csv("cs-test.csv", row.names = 1)

df_test_ext <- df_test_ext[df_test_ext$age > 0, ]

df_test_ext$MonthlyIncome[is.na(df_test_ext$MonthlyIncome)] <- median_income
df_test_ext$NumberOfDependents[is.na(df_test_ext$NumberOfDependents)] <- median_dependents

test_ext_woe <- woebin_ply(df_test_ext, bins)

test_ext_prob  <- predict(model, newdata = test_ext_woe, type = "response")
test_ext_score <- prob_to_score(test_ext_prob, PDO = 60, Base = 600, odds0 = 1/15)

df_test_ext$Score <- test_ext_score
head(df_test_ext)

# ---- 9. Model evaluation ----

# AUC
roc_obj <- roc(response = test_woe$SeriousDlqin2yrs, predictor = test_prob, quiet = TRUE)
print(paste("Logistic AUC =", round(as.numeric(auc(roc_obj)), 4)))

# KS
pred <- prediction(test_prob, test_woe$SeriousDlqin2yrs)
perf <- performance(pred, "tpr", "fpr")
ks <- max(slot(perf, "y.values")[[1]] - slot(perf, "x.values")[[1]])

print(paste("KS =", round(ks, 4)))

# PSI
psi_calc <- function(expected, actual, bins = 10) {
  
  # Create bin boundaries
  breaks <- quantile(expected, probs = seq(0, 1, length.out = bins + 1), na.rm = TRUE)
  breaks[1] <- -Inf
  breaks[length(breaks)] <- Inf
  
  # Assign observations to bins
  expected_bin <- cut(expected, breaks = breaks)
  actual_bin   <- cut(actual, breaks = breaks)
  
  # Calculate proportion in each bin
  expected_pct <- prop.table(table(expected_bin))
  actual_pct   <- prop.table(table(actual_bin))
  
  # Avoid division by zero
  expected_pct <- ifelse(expected_pct == 0, 1e-6, expected_pct)
  actual_pct   <- ifelse(actual_pct == 0, 1e-6, actual_pct)
  
  # Compute PSI contribution for each bin
  psi_bin <- (expected_pct - actual_pct) * log(expected_pct / actual_pct)
  
  df <- data.frame(
    bin = names(expected_pct),
    expected = as.numeric(expected_pct),
    actual = as.numeric(actual_pct),
    psi = as.numeric(psi_bin)
  )
  
  psi_total <- sum(psi_bin)
  
  return(list(psi = psi_total, data = df))
}

# Internal PSI
psi_internal <- psi_calc(train_score, test_score)

# External PSI
psi_external <- psi_calc(train_score, test_ext_score)

# PSI values
psi_internal$psi
psi_external$psi

# ---- 10. Plots ----
ggplot(test_woe, aes(x = Score, fill = as.factor(SeriousDlqin2yrs))) +
  geom_histogram(alpha = 0.5, position = "identity", bins = 50) +
  labs(
    title = "Score Distribution by Good/Bad",
    x = "Credit Score",
    y = "Count",
    fill = "Class"
  ) +
  scale_fill_manual(values = c("#2E86C1", "#E74C3C"), labels = c("Good", "Bad"))

# ROC curve
plot(roc_obj, main = "ROC Curve")

# PSI
plot_psi <- function(psi_obj, title_text) {
  ggplot(psi_obj$data, aes(x = bin)) +
    
    #Bar plot for expected distribution
    geom_bar(aes(y = expected, fill = "Expected"), 
             stat = "identity", position = "dodge") +
    
    #Bar plot for actual distribution
    geom_bar(aes(y = actual, fill = "Actual"), 
             stat = "identity", position = "dodge") +
    labs(
      title = paste(title_text, " (PSI =", round(psi_obj$psi, 4), ")"),
      x = "Score Bin",
      y = "Percentage"
    ) +
    scale_fill_manual(values = c("Expected" = "#2E86C1", "Actual" = "#E74C3C")) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}
# Internal PSI plot
plot_psi(psi_internal, "PSI Internal (Train vs Test)")

# External PSI plot
plot_psi(psi_external, "PSI External (Train vs External)")

