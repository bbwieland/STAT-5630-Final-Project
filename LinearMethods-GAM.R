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