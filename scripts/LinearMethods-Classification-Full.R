library(tidyverse)
library(caret)
library(MASS)
library(MLmetrics)

# This code includes the implementation of the following models:
# 1. Baseline Highest-Frequency-Class Classification Model
# 2. Linear Discriminant Analysis for Classification
# 3. Elastic Net Logistic Regression (potentially L1/L2) for Classification
 
set.seed(4133) # For reproducibility of the analysis
theme_set(theme_bw(base_family = "Poppins")) # Setting a graphic style for the analysis
data = read_csv("https://raw.githubusercontent.com/bbwieland/STAT-5630-Final-Project/main/data/insurance.csv") # import dta

# Create response variable as factor & numeric, for different R implementations
data$response = factor(data$smoker, levels = c("no","yes"))
data$response.numeric = ifelse(data$response == "no",0,1)

## Train-Test Split ----

# We will use 70/30 train/test split size for our data

train.samples = createDataPartition(data$smoker, p = 0.7, list = F) # p = 0.7 for 70/30
train = data[train.samples,]
test = data[-train.samples,]

## Baseline Model ----

# for our baseline model we simply predict each observation to belong to the most prominent class

baseline.pred = length(train$response[train$response == "yes"])/length(train$response)

## Linear Discriminant Analysis ----

# We need to transform our data — specifically, center and scale it — for optimal LDA performance.

# define sampling method (downsampling to address class imbalance)
sampling.method = trainControl(method = "repeatedcv", # specify repeated cross-validation
                     number = 10, # we want 10-fold CV
                     repeats = 10, # we want to repeat 10 times
                     verboseIter = T, # we want to know info on each fold/repeat
                     sampling = "down") # note the use of downsampling to address class imbalance

# fit the model
lda.model = caret::train(response ~ age + bmi + children + region + sex + charges, # our model formula
                              data = train, # fit to training data, we use test data for evaluation
                              method = "lda", # specify linear discriminant analysis
                              preProcess = c("scale", "center"), # tells R to scale & center before fitting LDA
                              trControl = sampling.method) # trControl passes the settings specified above for 10-fold CV to the LDA fit

## Elastic Net Logistic Regression ----

# fit the model
# The training code is similar to that for the LDA model above
lr.model = caret::train(response ~ age + bmi + children + region + sex + charges,
                         data = train,
                         method = "glmnet", # note use of glmnet for elastic net
                         family = "binomial", # "binomial" specifies a logistic regression model
                         preProcess = c("scale", "center"),
                         trControl = sampling.method)

## Compare Model Performance ----

# create predictions from LDA model
lda.pred = predict(lda.model,test, type = "prob") %>% 
  pull(yes)

# create predictions from elastic net model
lr.pred = predict(lr.model, test, type = "prob") %>%
  pull(yes)

# function to extract common classification metrics

get_metrics = function(model.type,pred,true = test$response.numeric) {
  accuracy = Accuracy(round(pred),true) # compute accuracy
  logloss = LogLoss(pred,true) # compute log-loss
  brier = MAE(pred,true) # compute MAE, or Brier score
  data.frame(Model = model.type, Accuracy = accuracy, LogLoss = logloss, Brier = brier) # combine into tidy format
} 

# use our function to create the final table of model performance
metrics = rbind(
  get_metrics("Baseline",baseline.pred),
  get_metrics("LDA",lda.pred),
  get_metrics("Elastic Net Logistic",lr.pred)
)

# Create a pretty table for inclusion in appendinx
library(gt) # for making tables
library(gtExtras) # for styling tables

gt(metrics) %>%
  gt_theme_nytimes() %>% # use NYT theme for aesthetic reasons
  fmt_number(c(LogLoss,Brier), decimals = 3) %>% # format the log-loss and Brier scores
  fmt_percent(Accuracy, decimals = 1) %>% # format the accuracy
  tab_header("Classification Model Evaluation") # add a title to the table
