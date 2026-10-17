library(scorecard)
library(ggplot2)
library(xgboost)
library(pROC)

set.seed(42)
data <- read.csv("cs-training.csv", row.names = 1, check.names = FALSE)
data <- subset(data, age > 0)
groups <- split(seq_len(nrow(data)), data$SeriousDlqin2yrs)
train_rows <- unlist(lapply(groups, function(x) sample(x, floor(0.7 * length(x)))))
train <- data[train_rows, ]
test <- data[-train_rows, ]
external <- read.csv("cs-test.csv", row.names = 1, check.names = FALSE)
external <- subset(external, age > 0)

# Learn missing-value replacements from the training set.
for (name in c("MonthlyIncome", "NumberOfDependents")) {
  value <- median(train[[name]], na.rm = TRUE)
  train[[name]][is.na(train[[name]])] <- value
  test[[name]][is.na(test[[name]])] <- value
  external[[name]][is.na(external[[name]])] <- value
}

features <- setdiff(names(train), "SeriousDlqin2yrs")
bins <- woebin(train, y = "SeriousDlqin2yrs", x = features,
               bin_num_limit = 5, positive = "1", no_cores = 1)
iv_values <- vapply(bins, function(x) x$total_iv[1], numeric(1))
selected <- names(iv_values)[iv_values > 0.02]
bins <- bins[selected]
woe_names <- paste0(selected, "_woe")

train_woe <- woebin_ply(train, bins, no_cores = 1)
test_woe <- woebin_ply(test, bins, no_cores = 1)
formula <- reformulate(sprintf("`%s`", woe_names), response = "SeriousDlqin2yrs")
lr <- glm(formula, data = train_woe, family = binomial())
lr_train <- predict(lr, train_woe, type = "response")
lr_test <- predict(lr, test_woe, type = "response")

# Fixed benchmark settings; the test set is used only for evaluation.
x_train <- xgb.DMatrix(as.matrix(as.data.frame(train_woe)[, woe_names]),
                       label = train$SeriousDlqin2yrs)
x_test <- xgb.DMatrix(as.matrix(as.data.frame(test_woe)[, woe_names]))
xgb <- xgb.train(
  params = list(objective = "binary:logistic", max_depth = 4, eta = 0.1,
                subsample = 0.8, colsample_bytree = 0.8, nthread = 1),
  data = x_train, nrounds = 100, verbose = 0
)
xgb_train <- predict(xgb, x_train)
xgb_test <- predict(xgb, x_test)

# odds0 is bad/good; higher default risk must give fewer points.
card <- scorecard(bins, lr, points0 = 600, odds0 = 1 / 15, pdo = 60)
train_score <- scorecard_ply(train, card)$score
test_score <- scorecard_ply(test, card)$score
external_score <- scorecard_ply(external, card)$score

y <- test$SeriousDlqin2yrs
roc_lr <- roc(y, lr_test, levels = c(0, 1), direction = "<", quiet = TRUE)
roc_xgb <- roc(y, xgb_test, levels = c(0, 1), direction = "<", quiet = TRUE)
auc_test <- roc.test(roc_xgb, roc_lr, paired = TRUE, method = "delong")
results <- data.frame(
  model = c("Logistic", "XGBoost"),
  AUC = c(as.numeric(auc(roc_lr)), as.numeric(auc(roc_xgb))),
  KS = c(max(roc_lr$sensitivities + roc_lr$specificities - 1),
         max(roc_xgb$sensitivities + roc_xgb$specificities - 1)),
  Brier = c(mean((lr_test - y)^2), mean((xgb_test - y)^2))
)

# Calibration groups use cut points from training predictions.
train_pred <- list(Logistic = lr_train, XGBoost = xgb_train)
test_pred <- list(Logistic = lr_test, XGBoost = xgb_test)
calibration <- do.call(rbind, lapply(names(test_pred), function(model) {
  breaks <- unique(quantile(train_pred[[model]], seq(0.1, 0.9, 0.1)))
  bin <- cut(test_pred[[model]], c(-Inf, breaks, Inf))
  out <- aggregate(cbind(predicted = test_pred[[model]], observed = y),
                   by = list(bin = bin), FUN = mean)
  out$model <- model
  out
}))

approval <- do.call(rbind, lapply(names(test_pred), function(model) {
  do.call(rbind, lapply(c(0.6, 0.7, 0.8), function(target) {
    cutoff <- unname(quantile(train_pred[[model]], target))
    approved <- test_pred[[model]] <= cutoff
    data.frame(model, target, cutoff, approved_rate = mean(approved),
               default_rate = mean(y[approved]))
  }))
}))

# Train-score deciles set the same PSI boundaries for both comparisons.
score_breaks <- c(-Inf, unique(quantile(train_score, seq(0.1, 0.9, 0.1))), Inf)
expected <- prop.table(table(cut(train_score, score_breaks)))
psi <- sapply(list(Test = test_score, External = external_score), function(score) {
  actual <- prop.table(table(cut(score, score_breaks)))
  p <- pmax(as.numeric(expected), 1e-6)
  q <- pmax(as.numeric(actual), 1e-6)
  sum((q - p) * log(q / p))
})

cat("Samples: train", nrow(train), "test", nrow(test),
    "external", nrow(external), "\n")
cat("Selected binned IV:\n")
print(round(iv_values[selected], 4))
cat("Test performance:\n")
print(results, row.names = FALSE, digits = 4)
cat("AUC 95% DeLong CIs: logistic", round(ci.auc(roc_lr), 4),
    "XGBoost", round(ci.auc(roc_xgb), 4), "\n")
cat("Paired AUC difference (XGBoost - logistic):",
    round(unname(auc_test$estimate[1] - auc_test$estimate[2]), 4),
    "95% CI", round(auc_test$conf.int, 4),
    "p =", signif(auc_test$p.value, 3), "\n")
cat("PSI:", round(psi, 5), "\n")
cat("Approval analysis:\n")
print(approval, row.names = FALSE, digits = 4)

dir.create("Outputs", showWarnings = FALSE)
png("Outputs/ROC_Curve.png", width = 900, height = 650)
plot(roc_lr, col = "#2369a1", lwd = 2, main = "Test ROC curves")
lines(roc_xgb, col = "#d17b29", lwd = 2)
legend("bottomright", c("Logistic", "XGBoost"),
       col = c("#2369a1", "#d17b29"), lwd = 2)
dev.off()

p_cal <- ggplot(calibration, aes(predicted, observed, color = model)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +
  geom_line() + geom_point() +
  labs(title = "Test probability calibration", x = "Mean predicted default probability",
       y = "Observed default rate", color = "Model") +
  theme_minimal()
ggsave("Outputs/Calibration.png", p_cal, width = 7, height = 5, dpi = 150)

p_approval <- ggplot(approval, aes(target, default_rate, color = model)) +
  geom_line() + geom_point(size = 2) +
  scale_x_continuous(breaks = c(0.6, 0.7, 0.8), labels = c("60%", "70%", "80%")) +
  scale_y_continuous(labels = function(x) paste0(round(100 * x, 1), "%")) +
  labs(title = "Observed default rate among approved test borrowers",
       x = "Target approval rate (training set)", y = "Approved default rate",
       color = "Model") + theme_minimal()
ggsave("Outputs/Approval_Analysis.png", p_approval, width = 7, height = 5, dpi = 150)

p_score <- ggplot(data.frame(score = test_score, default = factor(y)),
                  aes(score, fill = default)) +
  geom_density(alpha = 0.4) +
  labs(title = "Test score distribution", x = "Credit score", y = "Density",
       fill = "Default") + theme_minimal()
ggsave("Outputs/Score_Distribution.png", p_score, width = 7, height = 5, dpi = 150)
