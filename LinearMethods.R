library(tidyverse) # broad tool for data cleaning and visualization
library(caret) # broad tool for model fitting
library(MASS) # for variable transformations
library(MLmetrics) # for evaluation metrics
library(glmnet) # for ridge and lasso regression
library(splines2) # for creating splines
library(e1071) # for SVM
library(gt) # for making tables
library(gtExtras) # for styling tables

# Code Block 1: Classification, Part I ----

# This code block includes the implementation of the following models:
# 1. Baseline Highest-Frequency-Class Classification Model
# 2. Linear Discriminant Analysis for Classification
# 3. Elastic Net Logistic Regression (potentially L1/L2) for Classification
 
set.seed(4133) # For reproducibility of the analysis
theme_set(theme_bw(base_family = "Poppins")) # Setting a graphic style for the analysis
data = read_csv("https://raw.githubusercontent.com/bbwieland/STAT-5630-Final-Project/main/data/insurance.csv") # import data

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

gt(metrics) %>%
  gt_theme_nytimes() %>% # use NYT theme for aesthetic reasons
  fmt_number(c(LogLoss,Brier), decimals = 3) %>% # format the log-loss and Brier scores
  fmt_percent(Accuracy, decimals = 1) %>% # format the accuracy
  tab_header("Classification Model Evaluation") # add a title to the table

# Code Block 2: Classification, Part II ----

# This section of the code implements the following models:
# 1. Support Vector Machine for Classification, Linear Kernel
# 2. Support Vector Machine for Classification, Radial Kernel

#reading in data and converting categorical binary to binary
data.svm <- data
data.svm$sex[data.svm$sex == "female"] <- as.numeric(1)
data.svm$sex[data.svm$sex == "male"] <- as.numeric(0)

data.svm$smoker[data.svm$smoker == "yes"] <- as.numeric(1)
data.svm$smoker[data.svm$smoker == "no"] <- as.numeric(0)

#creating dummy variables for regions
dummy_region <- dummyVars(" ~ .", data=data.svm[, -1:-5])
dummy_region <- data.frame(predict(dummy_region, newdata=data.svm[,-1:-5]))

#completing dataset
data.svm <- cbind(data.svm[, -6:-7], data.frame(dummy_region))
data.svm$smoker <- as.numeric(as.character(data.svm$smoker))

#randomly sampling data to testing (30%) and training (70%) sets
index <- sample(1:nrow(data.svm), 0.7 * nrow(data.svm))
train <- data.svm[index, ]
test <- data.svm[-index, ]

#creating folds for cross-validation utilizing k=15 because it gave me a stable result, wheras lower k values gave different results based on how data was randomly split
folds <- createFolds(y = (y=data.svm$smoker), k = 10, list = FALSE, returnTrain = FALSE)

#creating potential costs for cross-validation
costs <- c(0.0001, 0.001, 0.01, 0.1, 1, 10, 100)

#completing cross-validation with different cost values
meanerrors <- NULL
for(c in 1:7){
  error<-NULL
  for (k in 1:10) {
    nums <- NULL
    for(i in 1:1338){
      if(folds[i]==k){
        nums <- c(nums, i)
      }
    }
    train.svm <- data.svm[-nums, ]
    test.svm <- data.svm[nums, ]
    
    #fitting linear svm 
    svmfit = svm(train.svm$smoker ~., data = train.svm %>% select(-response.numeric), kernel = "linear", cost = costs[c], scale = FALSE)
    y.pred <- predict(svmfit, test[,-5])
    fclassification <- as.integer(y.pred > 0.5)
    
    #finding error rate
    er <- mean((test$smoker != as.numeric(fclassification)))
    error<-c(error, er)
    
  }
  meanerrors <- c(meanerrors, mean(error))
}

#selecting the cost with the minimum mean error
value <- which.min(meanerrors)

#printing the results
print(paste("The minimum error cost is c=", costs[value]))
print(paste("The mean prediction error rate from the crossvalidation based on c=", costs[value], "is", meanerrors[value]))

#creating training and testing data and training the svm with our optimal cost
index <- sample(1:nrow(data.svm), 0.7*nrow(data.svm))
trainfinal <- data.svm[index, ]
testfinal <- data.svm[-index, ]
svmfinal<- svm(trainfinal$smoker ~., data = trainfinal, kernel = "linear", cost = costs[value], scale = FALSE)

#predicting our testing data
y.pred <- predict(svmfinal, testfinal[,-5])
fclassification <- as.integer(y.pred > 0.5)

#finding error rate
er <- mean((testfinal$smoker != as.numeric(fclassification)))

#output results
print(paste("The accuracy is", 1-er))
confusionMatrix(data=as.factor(fclassification), reference=as.factor(testfinal[,5]))

#creating folds for cross-validation utilizing k=10 because it gave me a stable result, wheras lower k values gave different results based on how data was randomly split
folds <- createFolds(y = (y=health$smoker), k = 10, list = FALSE, returnTrain = FALSE)

#creating potential costs for cross-validation
costs <- c(0.0001, 0.001, 0.01, 0.1, 1, 10, 100)

#completing cross-validation with different cost values
meanerrors <- c();
for(c in 1:7){
  error<-c()
  for (k in 1:10) {
    nums <- c()
    for(i in 1:1338){
      if(folds[i]==k){
        nums <- c(nums, i)
      }
    }
    train.svm <- data.svm[-nums, ]
    test.svm <- data.svm[nums, ]
    
    #fitting nonlinear svm 
    svmfit = svm(train.svm$smoker ~., data = train.svm %>% select(-response.numeric), kernel = "radial", cost = costs[c], scale = FALSE)
    y.pred <- predict(svmfit, test[,-5])
    fclassification <- as.integer(y.pred > 0.5)
    
    #finding error rate
    er <- mean((test$smoker != as.numeric(fclassification)))
    error<-c(error, er)
    
  }
  meanerrors <- c(meanerrors, mean(error))
}

#selecting the cost with the minimum mean error
value <- which.min(meanerrors)

#printing the results
print(paste("The minimum error cost is c=", costs[value]))
print(paste("The mean prediction error rate from the crossvalidation based on c=", costs[value], "is", meanerrors[value]))

#creating training and testing data and training the svm with our optimal cost
index <- sample(1:nrow(data.svm), 0.7*nrow(data.svm))
trainfinal <- data.svm[index, ]
testfinal <- data.svm[-index, ]
svmfinal<- svm(trainfinal$smoker ~., data = trainfinal, kernel = "radial", cost = costs[value], scale = FALSE)

#predicting our testing data
y.pred <- predict(svmfinal, testfinal[,-5])
fclassification <- as.integer(y.pred > 0.5)

#finding error rate
er <- mean((testfinal$smoker != as.numeric(fclassification)))

#output results
print(paste("The accuracy is", 1-er))
confusionMatrix(data=as.factor(fclassification), reference=as.factor(testfinal[,5]))

# Code Block 3: Regression ----

# This code block contains the implementation of the following models:

# 1. Sample Mean Baseline Regression Model
# 2. Kitchen-Sink (No Variable Selection) Linear Regression
# 3. Reduced Linear Regression Model via Feature Selection
# 4. Ridge Regression
# 5. Lasso Regression

set.seed(4133)

## Train-Test Split ----

# We will use 70/30 train/test split size for our data

train.indices = sample(1:nrow(data), 0.7 * nrow(data))
train = data[train.indices,]
test = data[-train.indices,]

## Exploring a Response Variable Transformation ----

# Visualizing our response variable's distribution with a histogram
ggplot(data, aes(x = charges)) +
  geom_histogram(bins = sqrt(nrow(data)), color = "black", fill = "grey") +
  labs(x = "Medical Charges (dollars)", y = "Count", title = "Distribution of Response Variable")

# Our linear methods for classification may work better with a more normally distributed response variable.
# So we explore a logarithmic transformation.
ggplot(data, aes(x = log(charges))) +
  geom_histogram(bins = sqrt(nrow(data)), color = "black", fill = "grey") +
  labs(x = "Medical Charges (dollars)", y = "Count", title = "Distribution of Response Variable")

# This makes our data approximately normal, which is a desirable behavior.
# We will model the log-transformed variable. 
train$charges.log = log(train$charges)
test$charges.log = log(test$charges)

## EDA of Predictor Relationships with Response Variable ----

# Let us analyze each variable individually. 

# First our numeric predictors:

# Age
ggplot(train, aes(x = age, y = charges.log)) +
  geom_point() +
  labs(x = "Age", y = "log(Charges)", title = "Relationship of Age and Medical Charges")

# BMI
ggplot(train, aes(x = bmi, y = charges.log)) +
  geom_point() +
  labs(x = "BMI", y = "log(Charges)", title = "Relationship of BMI and Medical Charges")

# Children (semi-categorical, but we can treat potentially as numeric)
ggplot(train, aes(x = children, y = charges.log, group = children)) +
  geom_boxplot() +
  labs(x = "Children", y = "log(Charges)", title = "Relationship of Number of Children and Medical Charges")

# Now our categorical predictors:

# Sex
ggplot(train, aes(x = sex, y = charges.log, group = sex)) +
  geom_boxplot() +
  labs(x = "Sex", y = "log(Charges)", title = "Relationship of Sex and Medical Charges")

# Smoker
ggplot(train, aes(x = smoker, y = charges.log, group = smoker)) +
  geom_boxplot() +
  labs(x = "Smoking Status", y = "log(Charges)", title = "Relationship of Smoking Status and Medical Charges")

# Region
ggplot(train, aes(x = region, y = charges.log, group = region)) +
  geom_boxplot() +
  labs(x = "Region", y = "log(Charges)", title = "Relationship of Residence Region and Medical Charges")


#Create dummy variables and remove redundant variables
dummy_region <- model.matrix(~ region - 1, data = data)
dummy_sex <- model.matrix(~ sex - 1, data = data)
dummy_smoker <- model.matrix(~ smoker - 1, data = data)
data <- cbind(data, dummy_region)
data <- cbind(data, dummy_smoker)
data <- cbind(data, dummy_sex)
data <- subset(data, select = -c(sexmale, sex, region, regionnortheast, smoker, smokerno))
train = data[train.indices,]
test = data[-train.indices,]

#Use a log tranformation on our charges
train$charges.log = log(train$charges)
test$charges.log = log(test$charges)


#Remove redundant variables
train <- subset(train, select = -charges)
test <- subset(test, select = -charges)


#Calculate the baseline error using the mean as a predictor
mean.baseline.error <- mean((test$charges.log- mean(train$charges.log))^2)


##Ridge Regression Model ---

# Create x and y training dataset
x.train <- train[, -which(names(train) == "charges.log")]
y.train <-train$charges.log

# Fit a ridge regression model using 10-fold CV
cv_fit <- cv.glmnet(data.matrix(x.train), y.train, alpha = 0, nfolds = 10, lambda = exp(seq(-10, 10, length.out = 1000)))

# Find the best lambda
best.lambda <- cv_fit$lambda.min

# Fit the ridge model using best lambda
insurance.fit <- glmnet(x.train, y.train, alpha = 0, lambda = best.lambda)

# Use the model to generate predictions
x.test <- data.matrix(test[, -which(names(test) == "charges.log")])
y.test.pred <- predict(cv_fit, x.test)

# Calculate the MSPE
mse <- mean((test$charges.log - y.test.pred)^2)
paste("Ridge MSE = ",mse)


##Lasso Regression Model ---

# Choose the best lasso regression model using 10-fold CV
cv_fit_lasso <- cv.glmnet(data.matrix(x.train), y.train,
                          alpha = 1, 
                          nfolds = 10, 
                          lambda = exp(seq(-10, 10, length.out = 10000)))
best.lambda.lasso <- cv_fit_lasso$lambda.min
final_fit_lasso <- glmnet(x.train, y.train,
                          alpha = 1, 
                          lambda = best.lambda.lasso)

#Apply the model to create prediction using test data
y_test_pred_lasso <- predict(final_fit_lasso, x.test)

#Calculate the MSPE
mse_lasso <- mean((test$charges.log - y_test_pred_lasso)^2)


##OLS Linear Model ---

fitted.linear <- lm(charges.log ~ ., data = train)
# plug-in the estimated beta, and generate predictions of Y.
y.pred <- predict(fitted.linear, data.frame(test))
# Calculate the MPSE
mspe.lm <- mean((y.pred - test$charges.log)^2)

## Now fit the data using a reduced model

#Remove the insignificant variable
new.linear.data = subset(data, select = -regionnorthwest)
train.linear = new.linear.data[train.indices,]
test.linear = new.linear.data[-train.indices,]
train.linear$charges.log = log(train.linear$charges)
test.linear$charges.log = log(test.linear$charges)

#Fit a generic linear model using the new data
fitted.linear <- lm(charges.log ~ ., data = train.linear)
# plug-in the estimated beta, and generate predictions of Y.
y.pred <- predict(fitted.linear, data.frame(test.linear))
# Find the MPSE
mspe.lm.new <- mean((y.pred - test.linear$charges.log)^2)

#Create a dataframe containing all the errors and prediction methods
errors <- data.frame("Prediction Method" = c("Sample Mean", "Linear Regression (All variables)", "Linear Regression (Reduced Model)", "Ridge Regression", "Lasso Regression"), 
                     "Mean-squared error" = c(mean.baseline.error,mspe.lm,mspe.lm.new, mse, mse_lasso) )

#print out errors
errors
