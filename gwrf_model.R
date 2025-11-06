#gwrf model

#load libraries
library(SpatialML)
library(sf)
library(readxl)
library(dplyr)
library(caret)
library(stringr)
library(parallel)

set.seed(123)


OUTCOME <- "CASTHMA_CrudePrev"  #change for the health outcome you came to generate
CONUS   <- TRUE                

#load in data
acs_2020_path <- "" #acs 5 year estimates file path (time 1)
acs_2023_path <- "" #acs 5 year estimates file path (time 2)
copd_2020_path <- "" #CDC PLACES 2023 data
copd_2023_path <- ""  #CDC PLACES 2024 data

bnd2020_path  <- "" #2023 County Boundary File
bnd2024_path  <- "" #2024 County Boundary File
  
#define variables
VARS <- c(
  "under_18","over_65","median_age","male_over65","female_over65",
  "white","black","asian","hispanic","male","female",
  "less_than_9th","bach","high_school","uninsured","poverty","median_income"
)

#settings for speed
CRS_EPSG    <- 5070
N_THREADS   <- max(1, detectCores() - 1)
NTREE_FINAL <- 400                
KERNEL      <- "adaptive"
IMP         <- "none"

#helper methods (similar to rf model)
pad5   <- function(x) stringr::str_pad(as.character(x), 5, pad = "0")
to_num <- function(x) as.numeric(as.character(x))
rmse   <- function(p,o) sqrt(mean((p-o)^2))
r2     <- function(p,o) cor(p, o)^2
mae    <- function(p,o) mean(abs(p-o))

prep_year <- function(bnd_path, acs_path, places_path, fips_col, outcome_col, vars_keep, conus = FALSE) {
  acs      <- read_excel(acs_path)
  boundary <- st_read(bnd_path, quiet = TRUE) %>% rename(county_fips = GEOID)
  places   <- read_excel(places_path)
  
  if (conus && "STATEFP" %in% names(boundary)) {
    boundary <- boundary %>% filter(!STATEFP %in% c("02","15","72"))
  }
  
  acs      <- acs      %>% mutate(county_fips = pad5(county_fips))
  boundary <- boundary %>% mutate(county_fips = pad5(county_fips))
  places   <- places   %>% rename(county_fips = !!fips_col) %>% mutate(county_fips = pad5(county_fips))
  
  vars_final <- intersect(vars_keep, names(acs))
  keep_cols  <- c("county_fips", outcome_col)
  if (!outcome_col %in% names(places)) keep_cols <- "county_fips"
  places <- places %>% select(any_of(keep_cols))
  
  joined <- boundary %>% left_join(acs, by="county_fips") %>% left_join(places, by="county_fips")
  
  # numeric-only
  for (v in vars_final) joined[[v]] <- to_num(joined[[v]])
  if (outcome_col %in% names(joined)) joined[[outcome_col]] <- to_num(joined[[outcome_col]])
  
  joined <- joined %>% filter(if_all(all_of(vars_final), ~ is.finite(.)))
  
  # project + centroids
  joined_5070 <- st_transform(joined, CRS_EPSG)
  pts         <- st_point_on_surface(joined_5070)
  coords      <- st_coordinates(pts)
  
  df <- joined_5070 %>% st_drop_geometry() %>%
    select(any_of(c(outcome_col, vars_final))) %>%
    as.data.frame()
  df$.rid <- seq_len(nrow(df))
  list(df = df, coords = coords, vars = vars_final)
}

cat("== Prep 2020...\n")
D20 <- prep_year(bnd2020_path, acs2020_path, places2020_path, "FIPS", OUTCOME, VARS, conus = CONUS)

train_df <- D20$df %>% filter(is.finite(.data[[OUTCOME]]))
coords20 <- D20$coords[train_df$.rid, , drop = FALSE]

# drop near-zero variance + non-finite cols
nzv <- nearZeroVar(train_df[, D20$vars, drop = FALSE])
pred_vars <- if (length(nzv)) D20$vars[-nzv] else D20$vars
pred_vars <- pred_vars[vapply(train_df[, pred_vars, drop = FALSE], function(z) all(is.finite(z)), logical(1))]
stopifnot(length(pred_vars) > 1)

# 80/20 holdout
idx <- createDataPartition(train_df[[OUTCOME]], p = 0.80, list = FALSE)
df_tr <- train_df[idx, c(OUTCOME, pred_vars, ".rid"), drop = FALSE]
df_te <- train_df[-idx, c(OUTCOME, pred_vars, ".rid"), drop = FALSE]
C_tr  <- coords20[idx, , drop = FALSE]
C_te  <- coords20[-idx, , drop = FALSE]
class(df_tr) <- "data.frame"; class(df_te) <- "data.frame"

# fixed fast hyperparams
p        <- length(pred_vars)
mtry_use <- max(1, floor(sqrt(p)))            
bw_use   <- min(100, nrow(df_tr) - 1)          

form <- as.formula(paste(OUTCOME, "~", paste(pred_vars, collapse = " + ")))

cat(sprintf("== Train GWRF: n=%d, p=%d, bw=%d, mtry=%d, ntree=%d, threads=%d\n",
            nrow(df_tr), p, bw_use, mtry_use, NTREE_FINAL, N_THREADS))

gwrf_model <- SpatialML::grf(
  formula       = form,
  dframe        = df_tr[, c(OUTCOME, pred_vars), drop = FALSE],
  coords        = C_tr,
  bw            = bw_use,
  kernel        = KERNEL,
  ntree         = NTREE_FINAL,
  nthreads      = N_THREADS,
  mtry          = mtry_use,
  forests       = TRUE,
  write.forest  = TRUE,
  importance    = "impurity",
  geo.weighted  = TRUE,
  print.results = TRUE
)

#2023 holdout
pred_2020 <- SpatialML::predict.grf(
  gwrf_model,
  new.data   = as.data.frame(cbind(df_te[, pred_vars, drop = FALSE], X = C_te[,1], Y = C_te[,2])),
  x.var.name = "X",
  y.var.name = "Y",
  nthreads   = N_THREADS
)
rmse_20 <- rmse(pred_2020, df_te[[OUTCOME]])
r2_20   <- r2  (pred_2020, df_te[[OUTCOME]])
mae_20  <- mae (pred_2020, df_te[[OUTCOME]])
cat("\n--- GWRF 2020 HOLDOUT (", OUTCOME, ") ---\n", sep = "")
cat("RMSE:", round(rmse_20, 3), " R2:", round(r2_20, 3), " MAE:", round(mae_20, 3), "\n")

#2024 predictions
cat("== Prep 2023...\n")
D23 <- prep_year(bnd2024_path, acs2023_path, places2023_path, "CountyFIPS", OUTCOME, VARS, conus = CONUS)

pred_keep <- intersect(pred_vars, names(D23$df))
df23 <- D23$df %>% filter(if_all(all_of(pred_keep), ~ is.finite(.)))
coords23 <- D23$coords[df23$.rid, , drop = FALSE]

cat(sprintf("== Predict 2023: n=%d\n", nrow(df23)))
pred_23 <- SpatialML::predict.grf(
  gwrf_model,
  new.data   = as.data.frame(cbind(df23[, pred_keep, drop = FALSE], X = coords23[,1], Y = coords23[,2])),
  x.var.name = "X",
  y.var.name = "Y",
  nthreads   = N_THREADS
)

if (OUTCOME %in% names(df23)) {
  eval_idx <- which(is.finite(df23[[OUTCOME]]))
  if (length(eval_idx) > 0) {
    rmse_23 <- rmse(pred_23[eval_idx], df23[[OUTCOME]][eval_idx])
    r2_23   <- r2  (pred_23[eval_idx], df23[[OUTCOME]][eval_idx])
    mae_23  <- mae (pred_23[eval_idx], df23[[OUTCOME]][eval_idx])
    cat("\n--- 2023 TRANSFERABILITY (", OUTCOME, ") ---\n", sep = "")
    cat("RMSE:", round(rmse_23, 3), " R2:", round(r2_23, 3), " MAE:", round(mae_23, 3), "\n")
  } else {
    cat("\nNo non-missing observed ", OUTCOME, " in 2023 after filtering — only predictions produced.\n", sep = "")
  }
} else {
  cat("\nOutcome ", OUTCOME, " not found in 2023 PLACES — only predictions produced.\n", sep = "")
}

#SAVE estimates
