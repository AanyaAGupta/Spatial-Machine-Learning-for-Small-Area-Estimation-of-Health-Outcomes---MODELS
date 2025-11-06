# ============================================================
# GLOBAL OLS WORKFLOW — Train on 2020 (80/20), Refit, Predict 2023
# ============================================================
# This script performs a cross-year regression analysis using a global Ordinary Least Squares (OLS) model. 
# It:
#   1. Loads 2020 ACS and PLACES data (health + demographics)
#   2. Trains an OLS model on 80% of 2020 counties
#   3. Evaluates model on 20% holdout
#   4. Refits on all 2020 data
#   5. Predicts outcomes for 2023 health outcome
# ============================================================

#libraries
library(readxl)
library(dplyr)
library(tidyr)

set.seed(123) #ensure reproducibility of 80/20 split

#reading in data
acs_2020_path <- "" #acs 5 year estimates file path (time 1)
acs_2023_path <- "" #acs 5 year estimates file path (time 2)
copd_2020_path <- "" #CDC PLACES 2023 data
copd_2023_path <- ""  #CDC PLACES 2024 data

#defining variables we want to use
VARS <- c(
  "under_18","over_65","male_over65",
  "white","female","bach","uninsured","poverty"
)

#helper methods to use throughout the code
std_fips <- function(x) sprintf("%05s", gsub("\\D","",as.character(x)))
to_num   <- function(x) as.numeric(as.character(x))
rmse_fun <- function(p,o) sqrt(mean((p-o)^2))
r2_fun   <- function(p,o){ sse <- sum((o-p)^2); sst <- sum((o-mean(o))^2); 1 - sse/sst }
mae_fun  <- function(p,o) mean(abs(p-o))

#step 1: load and join data
x20 <- read_excel(acs_2020_path) %>%
  mutate(county_fips = std_fips(county_fips)) %>%
  select(GEOID, county_fips, all_of(VARS))

y20 <- read_excel(copd_2020_path) %>%
  select(FIPS, CASTHMA_CrudePrev) %>%
  mutate(county_fips = std_fips(FIPS)) %>%
  select(county_fips, CASTHMA_CrudePrev)

df20 <- left_join(x20, y20, by = "county_fips")

df20[VARS] <- lapply(df20[VARS], to_num)
df20$CASTHMA_CrudePrev <- to_num(df20$CASTHMA_CrudePrev)

# drop rows missing any vals
df20 <- drop_na(df20, CASTHMA_CrudePrev, all_of(VARS))

#step 2: split 2023 Data (80/20 Train/Test)
n20  <- nrow(df20)
idx  <- sample.int(n20, size = floor(0.8*n20))
train_20 <- df20[idx, ]
test_20  <- df20[-idx, ]

form <- as.formula(paste("CASTHMA_CrudePrev ~", paste(VARS, collapse = " + ")))

fit_ols_20 <- lm(form, data = train_20)

pred_20_holdout <- predict(fit_ols_20, newdata = test_20)
cat("\n--- OLS 2020 HOLDOUT (20%) ---\n")
cat("RMSE:", round(rmse_fun(pred_20_holdout, test_20$CASTHMA_CrudePrev), 3), "\n")
cat("R2  :", round(r2_fun  (pred_20_holdout, test_20$CASTHMA_CrudePrev), 3), "\n")
cat("MAE :", round(mae_fun (pred_20_holdout, test_20$CASTHMA_CrudePrev), 3), "\n")

#refit ols and all 2023 data
final_ols_2020 <- lm(form, data = df20)

#load and join 2024 data
x23 <- read_excel(acs_2023_path) %>%
  mutate(county_fips = std_fips(county_fips)) %>%
  select(GEOID, county_fips, all_of(VARS))

y23 <- read_excel(copd_2023_path) %>%
  select(CountyFIPS, CASTHMA_CrudePrev) %>%
  mutate(county_fips = std_fips(CountyFIPS)) %>%
  select(county_fips, CASTHMA_CrudePrev)

df23 <- left_join(x23, y23, by = "county_fips")

# numeric
df23[VARS] <- lapply(df23[VARS], to_num)
df23$CASTHMA_CrudePrev <- to_num(df23$CASTHMA_CrudePrev)

# predictors must be complete
df23_pred <- drop_na(df23, all_of(VARS))

#step 5: predict 2024 outcomes using global ols (trained on 2023)
pred_2023 <- predict(final_ols_2020, newdata = df23_pred[, VARS, drop = FALSE])

out_2023 <- df23_pred %>%
  select(GEOID, county_fips, any_of("CASTHMA_CrudePrev"), all_of(VARS)) %>%
  mutate(pred_copd = pred_2023)

#step 6: evaluate 2024 predictions
eval_23 <- out_2023 %>% drop_na(CASTHMA_CrudePrev)
rmse_23 <- rmse_fun(eval_23$pred_copd, eval_23$CASTHMA_CrudePrev)
r_23    <- cor(eval_23$pred_copd,  eval_23$CASTHMA_CrudePrev)
r2_23   <- r_23^2
mae_23  <- mae_fun(eval_23$pred_copd, eval_23$CASTHMA_CrudePrev)
cat("\n--- GLOBAL OLS: Train 2020 → Test 2023 ---\n")
cat("RMSE:", round(rmse_23, 3), "\n")
cat("R2  :", round(r2_23, 3), "\n")
cat("MAE :", round(mae_23, 3), "\n")


#step 7: SAVE ESTIMATES AT THE END
