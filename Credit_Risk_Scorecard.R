# CREDIT RISK SCORECARD PROJECT

# Packages
# install.packages(c("scorecard", "ggplot2", "ROCR", "xgboost", "pROC", "car"))

library(scorecard)   
library(ggplot2)     
library(ROCR) 
library(xgboost)
library(pROC)
library(car)

set.seed(42)

# ---- 1. Load data ----
data <- read.csv("cs-training.csv", row.names = 1)  
data <- subset(data, age > 0)

# ---- 2. Train / test split ----
idx_by_y <- split(seq_len(nrow(data)), data$SeriousDlqin2yrs)
train_idx <- unlist(lapply(idx_by_y, function(x) sample(x, floor(length(x) * 0.7))))
train_set <- data[train_idx, ]
test_set  <- data[-train_idx, ]

# ---- 3. Missing values ----
income_med <- median(train_set$MonthlyIncome, na.rm = TRUE)
dep_med <- median(train_set$NumberOfDependents, na.rm = TRUE)

train_set$MonthlyIncome[is.na(train_set$MonthlyIncome)] <- income_med
test_set$MonthlyIncome[is.na(test_set$MonthlyIncome)] <- income_med

train_set$NumberOfDependents[is.na(train_set$NumberOfDependents)] <- dep_med
test_set$NumberOfDependents[is.na(test_set$NumberOfDependents)] <- dep_med

# ---- 4. WOE binning and IV ----
feature_names <- setdiff(names(train_set), "SeriousDlqin2yrs")

bins <- woebin(train_set, y = "SeriousDlqin2yrs", x = feature_names, 
               bin_num_limit = 5,   
               positive = "1",  
               no_cores = 1, monotone = TRUE)        

iv_table <- as.data.frame(iv(train_set, y = "SeriousDlqin2yrs", x = feature_names))
selected_vars <- iv_table$variable[iv_table$info_value > 0.02 & iv_table$info_value < 0.5]
if (length(selected_vars) == 0) selected_vars <- feature_names

train_woe <- woebin_ply(train_set, bins)
test_woe <- woebin_ply(test_set, bins)

# ---- 5. Logistic Regression ----
selected_woe_vars <- paste0(selected_vars, "_woe")
selected_woe_vars <- intersect(selected_woe_vars, names(train_woe))
if (length(selected_woe_vars) == 0) {
  selected_woe_vars <- grep("_woe$", names(train_woe), value = TRUE)
}

formula_lr <- as.formula(
  paste("SeriousDlqin2yrs ~",
    paste(sprintf("`%s`", selected_woe_vars), collapse = " + ")))

lr_model <- glm(formula_lr, data = train_woe, family = binomial())
summary(lr_model)

# Simple collinearity check
if (length(selected_woe_vars) > 1) {
  print(tryCatch(vif(lr_model), error = function(e) e$message))
}

# ---- 6. xgboost as benchmark ----
train_xgb <- as.matrix(as.data.frame(train_woe)[, selected_woe_vars])
test_xgb  <- as.matrix(as.data.frame(test_woe)[, selected_woe_vars])

xgb_train <- xgb.DMatrix(data = train_xgb, label = train_woe$SeriousDlqin2yrs)
xgb_test <- xgb.DMatrix(data = test_xgb, label = test_woe$SeriousDlqin2yrs)

xgb_model <- xgb.train(
  params = list(
    objective = "binary:logistic",
    eval_metric = "auc",
    max_depth = 4,
    eta = 0.1,
    subsample = 0.8,
    colsample_bytree = 0.8
  ),
  data = xgb_train,
  nrounds = 100,
  evals = list(train = xgb_train, test = xgb_test),
  verbose = 0
)

xgb_pred <- predict(xgb_model, xgb_test)
xgb_auc <- as.numeric(auc(test_woe$SeriousDlqin2yrs, xgb_pred))
print(paste("XGBoost AUC =", xgb_auc))

# ---- 7. Scorecard conversion---- 
card <- scorecard(bins, lr_model, 
                  points0 = 600,   
                  odds0 = 1/15,    
                  pdo = 60)        
print(card)

prob_to_score <- function(prob, base = 600, odds0 = 1/15, pdo = 60) {
  prob <- pmin(pmax(prob, 1e-10), 1 - 1e-10)
  b <- pdo / log(2)
  a <- base + b * log(odds0)
  as.integer(a - b * log(prob / (1 - prob)))
}

train_prob <- predict(lr_model, newdata = train_woe, type = "response")
test_prob  <- predict(lr_model, newdata = test_woe,  type = "response")

train_score <- prob_to_score(train_prob)
test_score  <- prob_to_score(test_prob)

# ---- 8. External scoring ----
ext_data <- read.csv("cs-test.csv", row.names = 1)
ext_data <- subset(ext_data, age > 0)

ext_data$MonthlyIncome[is.na(ext_data$MonthlyIncome)] <- income_med
ext_data$NumberOfDependents[is.na(ext_data$NumberOfDependents)] <- dep_med

ext_woe <- woebin_ply(ext_data, bins)
ext_prob  <- predict(lr_model, newdata = ext_woe, type = "response")
ext_score <- prob_to_score(ext_prob)

# ---- 9. Model evaluation ----

# AUC
roc_obj <- roc(response = test_woe$SeriousDlqin2yrs, predictor = test_prob, quiet = TRUE)
logit_auc <- as.numeric(auc(roc_obj))
print(paste("Logistic AUC = ", logit_auc))

# KS
pred_obj <- prediction(test_prob, test_woe$SeriousDlqin2yrs)
perf_obj <- performance(pred_obj, "tpr", "fpr")
ks_value <- max(slot(perf_obj, "y.values")[[1]] - slot(perf_obj, "x.values")[[1]])
print(paste("KS =", ks_value))

# PSI
psi_calc <- function(expected_score, actual_score, bins = 10) {
  breaks <- quantile(expected_score, probs = seq(0, 1, length.out = bins + 1), na.rm = TRUE)
  breaks <- unique(breaks)
  breaks[1] <- -Inf
  breaks[length(breaks)] <- Inf
  
  expected_bin <- cut(expected_score, breaks = breaks, include.lowest = TRUE)
  actual_bin <- cut(actual_score, breaks = breaks, include.lowest = TRUE)
  
  expected_pct <- prop.table(table(expected_bin))
  actual_pct <- prop.table(table(actual_bin))
  
  all_bins <- union(names(expected_pct), names(actual_pct))
  expected_pct <- expected_pct[all_bins]
  actual_pct <- actual_pct[all_bins]
  
  expected_pct[is.na(expected_pct)] <- 0
  actual_pct[is.na(actual_pct)] <- 0
  expected_pct[expected_pct == 0] <- 1e-6
  actual_pct[actual_pct == 0] <- 1e-6
  
  psi_each <- (expected_pct - actual_pct) * log(expected_pct / actual_pct)
  
  list(
    psi = sum(psi_each),
    data = data.frame(
      bin = all_bins,
      expected = as.numeric(expected_pct),
      actual = as.numeric(actual_pct)
    )
  )
}

psi_internal <- psi_calc(train_score, test_score)
psi_external <- psi_calc(train_score, ext_score)

print(paste("Internal PSI =", round(psi_internal$psi, 4)))
print(paste("External PSI =", round(psi_external$psi, 4)))

# ---- 10. Plots ----
# Score distribution
test_woe$Score <- test_score
p_score <- ggplot(test_woe, aes(x = Score, fill = as.factor(SeriousDlqin2yrs))) +
  geom_histogram(position = "identity", alpha = 0.5, bins = 30) +
  labs(
    title = "Score Distribution by Good/Bad",
    x = "Credit Score",
    y = "Count",
    fill = "Class"
  ) +
  theme_minimal()

print(p_score)

# ROC curve
plot(roc_obj, main = "ROC Curve")

# PSI
plot_psi <- function(psi_obj, title_text) {
  plot_df <- data.frame(
    bin = rep(psi_obj$data$bin, 2),
    pct = c(psi_obj$data$expected, psi_obj$data$actual),
    sample = rep(c("Expected", "Actual"), each = nrow(psi_obj$data))
  )
  
  ggplot(plot_df, aes(x = bin, y = pct, fill = sample)) +
    geom_col(position = "dodge") +
    labs(
      title = paste0(title_text, " (PSI = ", psi_obj$psi, ")"),
      x = "Score Bin",
      y = "Percentage",
      fill = "Sample"
    ) +
    scale_fill_manual(values = c("Expected" = "#2E86C1", "Actual" = "#E74C3C")) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

print(plot_psi(psi_internal, "PSI Internal (Train vs Test)"))
print(plot_psi(psi_external, "PSI External (Train vs External)"))

