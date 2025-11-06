# GLOBAL RF — train on 2020, test on 2023

#libraries
library(readxl)
library(dplyr)
library(caret)
library(ranger)

set.seed(123)

#reading in data
acs_2020_path <- "" #acs 5 year estimates file path (time 1)
acs_2023_path <- "" #acs 5 year estimates file path (time 2)
copd_2020_path <- "" #CDC PLACES 2023 data
copd_2023_path <- ""  #CDC PLACES 2024 data

OUTCOME <- "CASTHMA_CrudePrev" #IMPORTANT: change based on outcome you want to predict

VARS <- c("median_income","bach","high_school","poverty","over_65","median_age",
          "hispanic","less_than_9th","white","asian","under_18","black",
          "uninsured","male","female","female_over65")

#load 2023 data
x20 <- read_excel(acs_2020_path) %>%
  select(GEOID, county_fips, all_of(VARS))

y20 <- read_excel(places_2020_path) %>%
  select(FIPS, all_of(OUTCOME)) %>%
  rename(county_fips = FIPS)

df20 <- left_join(x20, y20, by = "county_fips") %>% na.omit()
mod20 <- df20[, c(OUTCOME, VARS)]

#80/20 split
set.seed(123)
idx <- createDataPartition(mod20[[OUTCOME]], p = 0.8, list = FALSE)
train_df <- mod20[idx, ]
test_df  <- mod20[-idx, ]

#train rf
ctrl <- trainControl(method = "cv", number = 5) #change nummber = 3 if you want to do 3 fold (computationally faster)

p <- length(VARS)
grid_fast <- expand.grid(
  mtry = round(sqrt(p)),
  splitrule = "variance",
  min.node.size = 3
)

fit_rf <- train(
  x = train_df[, VARS, drop = FALSE],
  y = train_df[[OUTCOME]],
  method = "ranger",
  trControl = ctrl,
  tuneGrid = grid_fast,
  num.trees = 800,
  importance = "impurity",
  metric = "Rsquared"
)

#test holdout
pred_2020 <- predict(fit_rf, newdata = test_df)
m20 <- postResample(pred_2020, test_df[[OUTCOME]])
cat("\n--- 2020 HOLDOUT ---\n")
cat("RMSE:", round(m20["RMSE"], 3), "\n")
cat("R2  :", round(m20["Rsquared"], 3), "\n")
cat("MAE :", round(m20["MAE"], 3), "\n")

#refit 2023 data to model
fit_all_2020 <- train(
  x = mod20[, VARS, drop = FALSE],
  y = mod20[[OUTCOME]],
  method = "ranger",
  trControl = trainControl(method = "none"),
  tuneGrid = grid_fast,
  num.trees = 1000
)

#load 2024 data and predict
x23 <- read_excel(acs_2023_path) %>%
  select(GEOID, county_fips, all_of(VARS))

y23 <- read_excel(places_2023_path) %>%
  select(CountyFIPS, all_of(OUTCOME)) %>%
  rename(county_fips = CountyFIPS)

df23 <- left_join(x23, y23, by = "county_fips") %>% na.omit()

pred_2023 <- predict(fit_all_2020, newdata = df23[, VARS, drop = FALSE])

out_2023 <- df23 %>%
  select(GEOID, county_fips, all_of(OUTCOME)) %>%
  mutate(pred = pred_2023)

#evaluate 2024 estimates
r <- cor(out_2023$pred, out_2023[[OUTCOME]])
r2 <- r^2
mae <- mean(abs(out_2023$pred - out_2023[[OUTCOME]]))
rmse <- sqrt(mean((out_2023$pred - out_2023[[OUTCOME]])^2))

cat("\n--- GLOBAL RF: Train 2020 → Test 2023 ---\n")
cat("RMSE:", round(rmse, 3), "\n")
cat("R2  :", round(r2, 3), "\n")
cat("MAE :", round(mae, 3), "\n")

#SAVE ESTIMATES
