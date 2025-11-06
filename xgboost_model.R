#XGBOOST model

#load libraries
library(readxl)
library(dplyr)
library(xgboost)

set.seed(123)

#load in data
acs_2020_path <- "" #acs 5 year estimates file path (time 1)
acs_2023_path <- "" #acs 5 year estimates file path (time 2)
copd_2020_path <- "" #CDC PLACES 2023 data
copd_2023_path <- ""  #CDC PLACES 2024 data

#define variables you want to use
VARS <- c("median_income","bach","high_school","poverty","over_65","median_age",
          "hispanic","less_than_9th","white","asian","under_18","black",
          "uninsured","male","female","female_over65")

to_num <- function(z) as.numeric(as.character(z))

#load 2023 data
x20 <- read_excel(acs_2020_path) %>%
  select(GEOID, county_fips, all_of(VARS))
y20 <- read_excel(copd_2020_path) %>%
  select(FIPS, BPHIGH_CrudePrev) %>%
  rename(county_fips = FIPS)

df20 <- left_join(x20, y20, by = "county_fips")
df20[VARS] <- lapply(df20[VARS], to_num)
df20$BPHIGH_CrudePrev <- to_num(df20$BPHIGH_CrudePrev)
df20 <- df20[complete.cases(df20[, c(VARS, "BPHIGH_CrudePrev")]), ]

mod20 <- df20[, c("BPHIGH_CrudePrev", VARS)]

#80/20 split
set.seed(123)
idx_hold <- sample.int(nrow(mod20), floor(0.8 * nrow(mod20)))
train80  <- mod20[idx_hold, ]
test20   <- mod20[-idx_hold, ]

#matric for xgboost
dtrain <- xgb.DMatrix(data = data.matrix(train80[, VARS, drop = FALSE]),
                      label = train80$BPHIGH_CrudePrev)
dtestM <- xgb.DMatrix(data = data.matrix(test20[, VARS, drop = FALSE]))

#define grid
tgrid <- expand.grid(
  nrounds = c(1200, 2000, 3000, 4000),
  max_depth = c(5, 6, 7),
  eta = c(0.03, 0.05, 0.08),
  gamma = c(0, 1),
  colsample_bytree = c(0.7, 0.85),
  min_child_weight = c(1, 3, 5),
  subsample = c(0.7, 0.85)
)

#cv tuning
best <- list(score = Inf, nrounds = NA, params = NULL)
for (i in seq_len(nrow(tgrid))) {
  par <- tgrid[i, ]
  params <- list(
    objective = "reg:squarederror",
    eval_metric = "rmse",
    max_depth = par$max_depth,
    eta = par$eta,
    subsample = par$subsample,
    colsample_bytree = par$colsample_bytree,
    min_child_weight = par$min_child_weight,
    gamma = par$gamma
  )
  cv <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = par$nrounds,
    nfold = 5,
    early_stopping_rounds = 100,
    verbose = 0
  )
  sc <- cv$evaluation_log$test_rmse_mean[cv$best_iteration]
  if (!is.na(sc) && sc < best$score) {
    best$score   <- sc
    best$nrounds <- cv$best_iteration
    best$params  <- params
  }
}

cat("\nBest CV RMSE:", round(best$score, 4),
    " @ nrounds:", best$nrounds, "\n")

#train final model on best set of variables
fit_xgb <- xgb.train(
  params  = best$params,
  data    = dtrain,
  nrounds = best$nrounds,
  verbose = 0
)

#evaluate on model
pred_te <- predict(fit_xgb, dtestM)  # no ntree_limit, no warnings
rmse_20 <- sqrt(mean((pred_te - test20$BPHIGH_CrudePrev)^2))
r2_20   <- cor(pred_te, test20$BPHIGH_CrudePrev)^2
mae_20  <- mean(abs(pred_te - test20$BPHIGH_CrudePrev))

cat("\n--- XGBOOST 2020 HOLDOUT (20%) ---\n")
cat("RMSE:", round(rmse_20, 3), " R2:", round(r2_20, 3), " MAE:", round(mae_20, 3), "\n")

#refit on 2023 data
dall <- xgb.DMatrix(data = data.matrix(mod20[, VARS, drop = FALSE]),
                    label = mod20$BPHIGH_CrudePrev)
fit_all <- xgb.train(
  params  = best$params,
  data    = dall,
  nrounds = best$nrounds,
  verbose = 0
)

#load 2024 data + predict
x23 <- read_excel(acs_2023_path) %>%
  select(GEOID, county_fips, all_of(VARS))
y23 <- read_excel(copd_2023_path) %>%
  select(CountyFIPS, BPHIGH_CrudePrev) %>%
  rename(county_fips = CountyFIPS)

df23 <- left_join(x23, y23, by = "county_fips")
df23[VARS] <- lapply(df23[VARS], to_num)
df23$BPHIGH_CrudePrev <- to_num(df23$BPHIGH_CrudePrev)
df23 <- df23[complete.cases(df23[, c(VARS, "BPHIGH_CrudePrev")]), ]

d23 <- xgb.DMatrix(data = data.matrix(df23[, VARS, drop = FALSE]))
pred_2023 <- predict(fit_all, d23)

out_2023 <- df23 %>%
  select(GEOID, county_fips, BPHIGH_CrudePrev) %>%
  mutate(pred_copd = pred_2023)

#generate 2024 estimates
r <- cor(out_2023$pred_copd, out_2023$BPHIGH_CrudePrev)
r2 <- r^2
mae <- mean(abs(out_2023$pred_copd - out_2023$BPHIGH_CrudePrev))
rmse <- sqrt(mean((out_2023$pred_copd - out_2023$BPHIGH_CrudePrev)^2))

cat("\n--- XGBOOST 2023 TEMPORAL ---\n")
cat("RMSE:", round(rmse, 3), " R2:", round(r2, 3), " MAE:", round(mae, 3), "\n")

#SAVE estimates