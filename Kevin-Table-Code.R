library(gt) # for making tables
library(gtExtras) # for styling tables
library(tidyverse)

metrics = read_csv("https://raw.githubusercontent.com/bbwieland/STAT-5630-Final-Project/main/data/Regression%20Model%20Evaluation.csv")[,c(2,3)] %>%
  arrange(Mean.squared.error)
gt(metrics) %>%
  gt_theme_nytimes() %>% # use NYT theme for aesthetic reasons
  fmt_number(c(Mean.squared.error), decimals = 3) %>% # format the log-loss and Brier scores
  tab_header("Regression Model Evaluation") %>% # add a title to the table
  cols_label(Prediction.Method = "Prediction Method",
             Mean.squared.error = "Mean Squared Prediction Error")
