library(tidyverse)
library(caret)
library(rpart)
library(MLmetrics)
library(randomForest)
library(gbm)

set.seed(4133) # For reproducibility of the analysis

# Setting a graphic style for the analysis
theme_set(theme_bw(base_family = "Poppins"))

# Import data
data <- read_csv("https://raw.githubusercontent.com/bbwieland/STAT-5630-Final-Project/main/data/insurance.csv")

data <- data %>%
  mutate(across(c(sex, smoker, region), factor))

## Train-Test Split ----

# We will use 70/30 train/test split size for our data

train_indices <- sample(seq_len(nrow(data)), 0.7 * nrow(data))
train <- data[train_indices, ]
test <- data[-train_indices, ]

## Response Variable Transformation ----

# Visualizing our response variable's distribution with a histogram
ggplot(data, aes(x = charges)) +
  geom_histogram(bins = round(sqrt(nrow(data))), color = "black", fill = "grey") +
  labs(x = "Medical Charges (dollars)", y = "Count", title = "Distribution of Response Variable")

# Our  methods for classification may work better with a more normally distributed response variable.
# So we explore a logarithmic transformation.
ggplot(data, aes(x = log(charges))) +
  geom_histogram(bins = round(sqrt(nrow(data))), color = "black", fill = "grey") +
  labs(x = "Medical Charges (dollars)", y = "Count", title = "Distribution of Response Variable")

# This makes our data approximately normal, which is a desirable behavior.
# We will model the log-transformed variable.
train$charges.log <- log(train$charges)
test$charges.log <- log(test$charges)

## Decision Tree ----

tree_tuned <- train(charges.log ~ age + sex + bmi + children + smoker + region,
  data = train,
  method = "rpart",
  preProcess = c("center", "scale"),
  tuneGrid = data.frame(.cp = seq(0, 0.05, 0.001)),
  trControl = trainControl(
    method = "repeatedcv",
    repeats = 5, number = 10
  )
)

best_cp <- tree_tuned$bestTune$cp

tree_best <- rpart(charges.log ~ age + sex + bmi + children + smoker + region,
  data = train,
  cp = best_cp
)

plot(tree_best)
par(mar = c(0, 4, 0, 2))
text(tree_best, cex = 0.7)

tree_preds <- predict(tree_best, newdata = test)

tree_mse <- MSE(tree_preds, test$charges.log)

## Random Forest ----

rf_best <- randomForest(charges.log ~ age + sex + bmi + children + smoker + region,
  data = train
)

rf_preds <- predict(rf_best, test)

rf_mse <- MSE(rf_preds, test$charges.log)

randomForest::varImpPlot(rf_best)

## GBM / XGBoost ----

# Note that we choose the optimal # of trees (hyperparameter)
gbm_fit <- gbm::gbm(
  formula = charges.log ~ age + sex + bmi + children + smoker + region,
  distribution = "gaussian",
  data = train,
  n.trees = 5000,
  interaction.depth = 3,
  shrinkage = 0.01,
  cv.folds = 10,
  n.cores = NULL,
  verbose = FALSE,
  train.fraction = .75
)

gbm_fit

gbm_preds <- predict(gbm_fit, test)

gbm_mse <- MSE(gbm_preds, test$charges.log)

nonlinear <- data.frame(
  model = c("Decision Tree", "Random Forest", "Gradient-Boosted Decision Tree"),
  mse = c(tree_mse, rf_mse, gbm_mse),
  type = "Non-Linear"
)

linear <- data.frame(
  model = c("Ridge Regression", "Lasso Regression", "Linear Regression (OLS)"),
  mse = c(0.188, 0.160, 0.161),
  type = "Linear"
)

baseline <- data.frame(
  model = "Sample Mean",
  mse = 0.843,
  type = "Baseline"
)

full_model_eval <- rbind(nonlinear, linear, baseline) %>%
  select(model, type, mse) %>%
  arrange(mse)


library(gt)
library(gtExtras)

full_model_eval %>%
  group_by(type) %>%
  gt() %>%
  gtExtras::gt_theme_nytimes() %>%
  fmt_number(mse, decimals = 3) %>%
  tab_header("Patient Charges: Model Evaluation",
    subtitle = "The selected metric for evaluation was mean squared error of model predictions on the testing set."
  ) %>%
  tab_style(
    locations = cells_row_groups(),
    style = cell_text(weight = "bold")
  )
