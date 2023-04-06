library(tidyverse)
library(caret)
library(MASS)
library(MLmetrics)

set.seed(4133) # For reproducibility of the analysis
theme_set(theme_bw(base_family = "Poppins")) # Setting a graphic style for the analysis
data = read_csv("https://raw.githubusercontent.com/bbwieland/STAT-5630-Final-Project/main/data/insurance.csv")

data$response = factor(data$smoker, levels = c("no","yes"))
data$response.numeric = ifelse(data$response == "no",0,1)

## Train-Test Split ----

# We will use 70/30 train/test split size for our data

train.samples = createDataPartition(data$smoker, p = 0.7, list = F)
train = data[train.samples,]
test = data[-train.samples,]

## Baseline Model ----

# for our baseline model we simply predict each observation to belong to the most prominent class

baseline.pred = length(train$response[train$response == "yes"])/length(train$response)

## Linear Discriminant Analysis ----

# We need to transform our data — specifically, center and scale it — for optimal LDA performance.

# define sampling method (downsampling to address class imbalance)
sampling.method = trainControl(method = "repeatedcv", 
                     number = 10, 
                     repeats = 10, 
                     verboseIter = FALSE,
                     sampling = "down") # note the use of downsampling to address class imbalance

# fit the model
lda.model = caret::train(response ~ age + bmi + children + region + sex + charges,
                              data = train,
                              method = "lda",
                              preProcess = c("scale", "center"),
                              trControl = sampling.method)

## Logistic Regression ----

# fit the model
lr.model = caret::train(response ~ age + bmi + children + region + sex + charges,
                         data = train,
                         method = "glmnet",
                         family = "binomial",
                         preProcess = c("scale", "center"),
                         trControl = sampling.method)

## Compare Model Performance ----

lda.pred = predict(lda.model,test, type = "prob") %>% 
  pull(yes)

lr.pred = predict(lr.model, test, type = "prob") %>%
  pull(yes)

get_metrics = function(model.type,pred,true = test$response.numeric) {
  accuracy = Accuracy(round(pred),true)
  logloss = LogLoss(pred,true)
  brier = MAE(pred,true)
  auc = AUC(pred,true)
  data.frame(Model = model.type, Accuracy = accuracy, LogLoss = logloss, Brier = brier, AUC = auc)
}

metrics = rbind(
  get_metrics("Baseline",baseline.pred),
  get_metrics("LDA",lda.pred),
  get_metrics("Logistic Regression",lr.pred)
)

metrics
