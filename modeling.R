library(dplyr)
library(tidymodels)
library(doFuture)
library(themis) # for SMOTE

############## 1. Data preprocessing before modeling ###############
# Factorize categorical vars
df_imputed2 <- df_imputed %>%
  mutate(
    Gender = as.factor(Gender),
    ICUType = as.factor(ICUType),
    across(matches("_missing(_[0-9]+)?$"), as.factor),
    `In-hospital_death` = as.factor(`In-hospital_death`)
  ) %>%
  rename(Death = `In-hospital_death`)

# Remove outcome variables for other projects 
data <- df_imputed2 %>%
  select(-c("SAPS-I", "SOFA", "Length_of_stay", "Survival")) %>%
  rename(ID=recordid)

# Split the data
set.seed(123)
data_split <- initial_split(data, prop = 0.7, strata = Death)
train_data <- training(data_split)
test_data  <- testing(data_split)

# Set up parallel backend
registerDoFuture()
plan(multisession, workers = 7)

################### Modeling #####################
############## 2. Elastic net ###############
set.seed(123)

rec_en <- recipe(Death ~ ., data = train_data) %>%
  step_indicate_na(all_of(c("Age", "Gender", "Height", "ICUType", "Weight")), prefix = "missing") %>% 
  step_impute_median(all_numeric_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_smote(Death, over_ratio = 1)

spec_en <- logistic_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet")
wf_en <- workflow() %>%
  add_recipe(rec_en) %>%
  add_model(spec_en)
cv_folds <- vfold_cv(train_data, v=5, strata=Death)
grid_en <- expand_grid(
  penalty = 10^seq(-4, -2, length.out = 100),  
  mixture = c(0.4, 0.6, 0.8) 
)
tune_en <- tune_grid(wf_en,
                     resamples = cv_folds,
                     grid = grid_en,
                     metrics = metric_set(mn_log_loss))
best_en <- select_best(tune_en, metric="mn_log_loss")
final_en_wf <- finalize_workflow(wf_en, best_en)
final_en_fit <- fit(final_en_wf, data = train_data)

# Predict and compute evaluation metrics
pred_en <- predict(final_en_fit, new_data = test_data, type = "prob") %>%
  bind_cols(test_data)
auc_en <- roc_auc(pred_en, truth = Death, .pred_1, event_level = "second")
class_pred_en <- predict(final_en_fit, new_data = test_data, type = "class") %>%
  bind_cols(test_data)
accuracy_en <- accuracy(class_pred_en, truth = Death, estimate = .pred_class)
sensitivity_en <- sens(class_pred_en, truth = Death, estimate = .pred_class, event_level = "second")
ppv_en <- ppv(class_pred_en, truth = Death, estimate = .pred_class, event_level = "second")
f1_en <- f_meas(class_pred_en, truth = Death, estimate = .pred_class, event_level = "second")

# Check the best hyperparameters
best_en
autoplot(tune_en)
# U-shaped curves for all mixture values, which is expected as too little regularization can lead to overfitting (left) and too much underfitting (right).
# The best hyperparamters are penalty = 0.00869 and mixture = 0.6, which lie in the middle of search grid.

############## 3. Lasso ###############
set.seed(123)

rec_lasso <- rec_en
spec_lasso <- logistic_reg(penalty = tune(), mixture = 1) %>%
  set_engine("glmnet")
wf_lasso <- workflow() %>%
  add_recipe(rec_lasso) %>%
  add_model(spec_lasso)
cv_folds <- vfold_cv(train_data, v=5, strata=Death)
grid_lasso <- expand_grid(
  penalty = 10^seq(-4, -2, length.out = 100)
)
tune_lasso <- tune_grid(wf_lasso,
                        resamples = cv_folds,
                        grid = grid_lasso,
                        metrics = metric_set(mn_log_loss))
best_lasso <- select_best(tune_lasso, metric="mn_log_loss")
final_lasso_wf <- finalize_workflow(wf_lasso, best_lasso)
final_lasso_fit <- fit(final_lasso_wf, data = train_data)

# Predict and compute evaluation metrics
pred_lasso <- predict(final_lasso_fit, new_data = test_data, type = "prob") %>%
  bind_cols(test_data)
auc_lasso <- roc_auc(pred_lasso, truth = Death, .pred_1, event_level = "second")
class_pred_lasso <- predict(final_lasso_fit, new_data = test_data, type = "class") %>%
  bind_cols(test_data)
accuracy_lasso <- accuracy(class_pred_lasso, truth = Death, estimate = .pred_class)
sensitivity_lasso <- sens(class_pred_lasso, truth = Death, estimate = .pred_class, event_level = "second")
ppv_lasso <- ppv(class_pred_lasso, truth = Death, estimate = .pred_class, event_level = "second")
f1_lasso <- f_meas(class_pred_lasso, truth = Death, estimate = .pred_class, event_level = "second")

############## 4. Random Forest ###############
## 
1. trees = 500: f1=0.145, auc=0.848



# time 39min

rec_rf <- rec_lgbm 

spec_rf <- rand_forest(
  trees = 500,
  mtry = tune(),
  # sample_size = tune(), # # not tunable with the ranger engine
  min_n = tune(),
  # tree_depth = tune() # not tunable with the ranger engine
) %>%
  set_engine("ranger", 
             importance="impurity",
             max.depth = 10) %>% 
  # early_stopping_rounds & eval_metric aren’t valid for ranger
  set_mode("classification")

wf_rf <- workflow() %>%
  add_recipe(rec_rf) %>%
  add_model(spec_rf)

rf_params <- extract_parameter_set_dials(spec_rf)
rf_params <- rf_params %>%
  update(
    min_n = min_n(range = c(20, 200)), 
    mtry = mtry(range = c(10,50)) #sqrt(p)=21
    # sample_size = sample_prop(range = c(0.1, 1)),
    # tree_depth = tree_depth(range = c(5,15))
  )

tune_rf <- tune_bayes(
  wf_rf,                
  resamples = cv_folds,
  param_info = rf_params,
  initial = 10,
  iter = 20,
  metrics = metric_set(mn_log_loss),
  control = control_bayes(no_improve = 10)
)

best_rf <- select_best(tune_rf, metric = "mn_log_loss")

final_rf_wf <- finalize_workflow(wf_rf, best_rf)
final_rf_fit <- fit(final_rf_wf, data = train_data)

# Predict on testing data
pred_rf <- predict(final_rf_fit, new_data = test_data, type = "prob") %>%
  bind_cols(test_data)
auc_rf <- roc_auc(pred_rf, truth = Death, .pred_1, event_level = "second")


class_pred_rf <- predict(final_rf_fit, new_data = test_data, type = "class") %>%
  bind_cols(test_data)

accuracy_rf <- accuracy(class_pred_rf, truth = Death, estimate = .pred_class)
sensitivity_rf <- sens(class_pred_rf, truth = Death, estimate = .pred_class, event_level = "second")
ppv_rf <- ppv(class_pred_rf, truth = Death, estimate = .pred_class, event_level = "second")
f1_rf <- f_meas(class_pred_rf, truth = Death, estimate = .pred_class, event_level = "second")


############## 5. LightGBM ###############
library(bonsai) # for lightGBM
library(dials) # for sample_prop() & col_sample_prop()
set.seed(123)

rec_lgbm <- recipe(Death ~., data=train_data) %>%
  step_indicate_na(all_of(c("Age", "Gender", "Height", "ICUType", "Weight")), prefix = "missing") %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors())

spec_lgbm <- boost_tree(
  trees = 1000,
  learn_rate = tune(),
  tree_depth = tune(),
  sample_size = tune(),
  mtry = tune(), # feature_fraction
  min_n = tune(),
) %>%
  set_engine(
    "lightgbm",
    objective = "binary",
    num_leaves = tune(), # engine‐specific
    lambda_l1 = tune(),
    lambda_l2 = tune(),
    bagging_freq = 5,
    max_bin = 255,
    counts = FALSE # interpret mtry as a proportion
  ) %>%
  set_mode("classification")

wf_lgbm <- workflow() %>%
  add_recipe(rec_lgbm) %>%
  add_model(spec_lgbm)

lgbm_params <- extract_parameter_set_dials(spec_lgbm)
lgbm_params <- lgbm_params %>%
  update(
    learn_rate = learn_rate(range = c(0.01, 0.5), trans = log10_trans()),
    tree_depth = tree_depth(range = c(4, 15)),
    num_leaves = num_leaves(range = c(10, 200)),
    min_n = min_n(range = c(10, 100)),
    sample_size = sample_prop(range = c(0.5, 1)),
    mtry = sample_prop(range = c(0.5, 1.0)),
    # feature_fraction = sample_prop(range=c(0.5,1)),
    lambda_l1 = penalty(range=c(1e-2,10), 
                        trans=log10_trans()),
    lambda_l2 = penalty(range=c(1e-2,10),
                        trans=log10_trans())
  )
tune_lgbm <- tune_bayes(
  wf_lgbm,                
  resamples = cv_folds,
  param_info = lgbm_params,
  initial = 24, # Question: ~ 2/4–10 × (# of parameters) (min 5–10)
  iter = 60, # ~ 2-4x initial /10–20 × (# of parameters), or until your budget in minutes
  metrics = metric_set(mn_log_loss),
  control = control_bayes(no_improve = 30) # ~ half of iter, or something like 10–20
)

best_lgbm <- select_best(tune_lgbm, metric = "mn_log_loss")

final_lgbm_wf <- finalize_workflow(wf_lgbm, best_lgbm)
final_lgbm_fit <- fit(final_lgbm_wf, data = train_data)

# Predict on testing data
pred_lgbm <- predict(final_lgbm_fit, new_data = test_data, type = "prob") %>%
  bind_cols(test_data)
auc_lgbm <- roc_auc(pred_lgbm, truth = Death, .pred_1, event_level = "second")


class_pred_lgbm <- predict(final_lgbm_fit, new_data = test_data, type = "class") %>%
  bind_cols(test_data)
accuracy_lgbm <- accuracy(class_pred_lgbm, truth = Death, estimate = .pred_class)
sensitivity_lgbm <- sens(class_pred_lgbm, truth = Death, estimate = .pred_class, event_level = "second")
ppv_lgbm <- ppv(class_pred_lgbm, truth = Death, estimate = .pred_class, event_level = "second")
f1_lgbm <- f_meas(class_pred_lgbm, truth = Death, estimate = .pred_class, event_level = "second")


############## 6. XGboost ###############
set.seed(123)

# THM: pure XGBoost can handle NAs, but not XGboost in tidymodels
rec_xgb <- rec_lgbm

spec_xgb <- boost_tree(
  trees = 1000,
  tree_depth = tune(),
  min_n = tune(),
  sample_size = tune(),
  mtry = tune(), # feature_fraction
  learn_rate = tune(),
  loss_reduction = tune()
) %>%
  set_engine("xgboost", 
             early_stopping_rounds = 30, 
             eval_metric = "logloss",
             counts = FALSE) %>%
  set_mode("classification")

wf_xgb <- workflow() %>%
  add_recipe(rec_xgb) %>%
  add_model(spec_xgb)

xgb_params <- extract_parameter_set_dials(spec_xgb)
xgb_params <- xgb_params %>%
  update(
    learn_rate = learn_rate(range = c(0.001, 0.3), trans = log10_trans()), 
    # XBboost is a slower learner than lightGBM
    tree_depth = tree_depth(range = c(4,20)), # XBboost is usu deeper than lightGBM
    min_n = min_n(range = c(10, 100)),
    sample_size = sample_prop(range = c(0.1, 1)),
    mtry = sample_prop(range = c(0.1, 1.0)),
    loss_reduction = loss_reduction(range = c(1e-3, 10), trans = log10_trans())
  )

tune_xgb <- tune_bayes(
  wf_xgb,
  resamples = cv_folds,
  param_info = xgb_params,
  initial = 10,
  iter = 20,
  metrics =  metric_set(mn_log_loss),
  control = control_bayes(no_improve = 10)
)

best_xgb <- select_best(tune_xgb, metric = "mn_log_loss")

final_xgb_wf <- finalize_workflow(wf_xgb, best_xgb)
final_xgb_fit <- fit(final_xgb_wf, data = train_data)

# Predict on testing data
pred_xgb <- predict(final_xgb_fit, new_data = test_data, type = "prob") %>%
  bind_cols(test_data)
auc_xgb <- roc_auc(pred_xgb, truth = Death, .pred_1, event_level = "second")
class_pred_xgb <- predict(final_xgb_fit, new_data = test_data, type = "class") %>%
  bind_cols(test_data)
accuracy_xgb <- accuracy(class_pred_xgb, truth = Death, estimate = .pred_class)
sensitivity_xgb <- sens(class_pred_xgb, truth = Death, estimate = .pred_class, event_level = "second")
ppv_xgb <- ppv(class_pred_xgb, truth = Death, estimate = .pred_class, event_level = "second")
f1_xgb <- f_meas(class_pred_xgb, truth = Death, estimate = .pred_class, event_level = "second")

# Check
extract_spec_parsnip(final_xgb_wf)




############## 7. Conclusion ###############
# F-1 score is chosen as the major evaluation metric because the test data is highly unbalanced.
# Three tree based models have higher accuracy and PPV, while Lasso and elastic net have higher F-1 score and sensitivity. 
# Lasso is the overall best model.

### Compare the metrics of five models
model_compare <- tibble(
  Model = c("Lasso","Elastic Net", "LightGBM", "XGBoost", "Random Forest"),
  AUC = c(auc_lasso$.estimate, auc_en$.estimate, auc_lgbm$.estimate, auc_xgb$.estimate, auc_rf$.estimate),
  Accuracy = c(accuracy_lasso$.estimate,accuracy_en$.estimate, accuracy_lgbm$.estimate, accuracy_xgb$.estimate, accuracy_rf$.estimate),
  Sensitivity = c(sensitivity_lasso$.estimate, sensitivity_en$.estimate, sensitivity_lgbm$.estimate, sensitivity_xgb$.estimate, sensitivity_rf$.estimate),
  PPV = c(ppv_lasso$.estimate, ppv_en$.estimate, ppv_lgbm$.estimate, ppv_xgb$.estimate, ppv_rf$.estimate),
  F1_Score = c(f1_lasso$.estimate,f1_en$.estimate, f1_lgbm$.estimate, f1_xgb$.estimate, f1_rf$.estimate)
)

model_compare
# Model           AUC Accuracy Sensitivity   PPV F1_Score
# 1 Lasso         0.863    0.799      0.751  0.389    0.513
# 2 Elastic Net   0.868    0.791      0.760  0.379    0.506
# 3 LightGBM      0.875    0.883      0.297  0.699    0.417
# 4 XGBoost       0.806    0.855      0.181  0.462    0.260
# 5 Random Forest 0.848    0.867      0.0801 0.771    0.145

# Copy to excel
model_compare %>%
  mutate(across(AUC:F1_Score, ~ round(.x, 3))) %>%
  clipr::write_clip()

### Check the top predictors in the overall best model (Lasso)
# Among the top predictors selected by the Lasso model
# - GCS_max_24 had the strongest negative association with mortality risk. The Glasgow Coma Scale (GCS) is a widely used tool to assess a patient's level of consciousness, and lower maximum GCS scores indicate severe neurological impairment, consistent with higher mortality risk. 
# - Age and BUN_min were positively associated with death, reflecting known risks, as older age and elevated blood urea nitrogen levels suggest impaired renal function and worse overall prognosis. 
# - Missingness indicators such as RespRate_missing between the first 6 hours and NISysABP_missing at the last 24 hours were important predictors, suggesting that absent or infrequent monitoring of respiratory rate and non-invasive systolic blood pressure may signal clinical instability or low healthcare intervention, both of which could contribute to poor outcomes. Other selected variables included markers of vital sign trends (e.g., GCS_trend_24, Na_trend, HCO3_trend) and laboratory measurements (Bilirubin_median, Creatinine_median, Lactate_median) that reflect underlying organ dysfunction. Overall, the model emphasized the critical role of neurologic status, age, kidney and liver function in predicting ICU mortality.

final_lasso_fit %>%
  extract_fit_parsnip() %>%
  tidy() %>%
  filter(term !="(Intercept)") %>%
  mutate(abs_estimate=abs(estimate)) %>%
  arrange(desc(abs_estimate)) %>%
  select(term, estimate) %>%
  slice_head(n=20) %>%
  mutate(estimate = round(estimate, 3)) 
#    term                   estimate
#    <chr>                     <dbl>
#  1 Age                       0.466
#  2 GCS_max_24               -0.39 
#  3 BUN_min                   0.339
#  4 GCS_trend_24             -0.296
#  5 Weight                   -0.287
#  6 RespRate_missing_0_X1     0.255
#  7 NIMAP_max_6              -0.233
#  8 NIMAP_missing_0_X1       -0.223
#  9 Bilirubin_median          0.213
# 10 HR_n_obs_0               -0.213
# 11 RespRate_median_0         0.211
# 12 PaCO2_n_obs              -0.206
# 13 Temp_median_24           -0.203
# 14 NIMAP_first_6             0.201
# 15 GCS_median_24            -0.201
# 16 Na_trend                  0.193
# 17 HCO3_trend               -0.193
# 18 NISysABP_missing_24_X1    0.193
# 19 Creatinine_median        -0.186
# 20 MechVent_first           -0.166

plan(sequential)

