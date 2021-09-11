# https://hansjoerg.me/2020/02/09/tidymodels-for-machine-learning/
rm(list=ls())  # remove all variables
cat("\014")    # clear Console
if (dev.cur()!=1) {dev.off()} # clear R plots if exists

packages = c('corrplot', 'skimr', 'tidymodels', 'tidyverse')
for (p in packages){
    if(!require(p, character.only = T)){
        install.packages(p)
    }
    library(p,character.only = T)
}

# Data Set: Diamonds ----
data(diamonds)
glimpse(diamonds)
summary(diamonds)
skim(diamonds)

corrplot(cor(diamonds[, c(1,5:6, 8:10)]), 
         diag = FALSE, 
         tl.pos = "td", 
         tl.cex = 1, 
         method = "number", 
         type = "upper",
         mar = c(0, 0, 1.5, 0),
         title = "Correlations between predictors")

# Separating Testing and Training Data: rsample ----
set.seed(1243)

dia_split <- initial_split(diamonds, prop = .1, strata = price)

dia_train <- training(dia_split)
dia_test  <- testing(dia_split)

dim(dia_train)
dim(dia_test)

dia_vfold <- vfold_cv(dia_train, v = 3, repeats = 1, strata = price)
dia_vfold %>% 
    mutate(df_ana = map(splits, analysis),
           df_ass = map(splits, assessment))

# Data Pre-Processing and Feature Engineering: recipes ----
qplot(carat, price, data = dia_train) +
    scale_y_continuous(trans = log_trans(), labels = function(x) round(x, -2)) +
    geom_smooth(method = "lm", formula = "y ~ poly(x, 4)") +
    labs(title = "Nonlinear relationship between price and carat of diamonds",
         subtitle = "The degree of the polynomial is a potential tuning parameter")

dia_rec <-
    recipe(price ~ ., data = dia_train) %>%
    step_log(all_outcomes()) %>%
    step_normalize(all_predictors(), -all_nominal()) %>%
    step_dummy(all_nominal()) %>%
    step_poly(carat, degree = 2)

juiced_data <- juice(prep(dia_rec))
names(juiced_data)

# Defining and Fitting Models: parsnip ----
lm_model <-
    linear_reg() %>%
    set_mode("regression") %>%
    set_engine("lm")

# fit on train data
lm_fit1 <- fit(lm_model, price ~ ., juiced_data)
lm_fit1

# Summarizing Fitted Models: broom ----
glance(lm_fit1$fit)
tidy(lm_fit1) %>% 
    arrange(desc(abs(statistic)))

lm_predicted <- augment(lm_fit1$fit, data = juiced_data) %>% 
    rowid_to_column()
select(lm_predicted, rowid, price, .fitted:.std.resid)

# In the following pipeline, the model is fit() separately to the three analysis data sets, 
# and then the fitted models are used to predict() on the three corresponding assessment data sets 
# (i.e., 3-fold cross-validation).

# Before that, analysis() and assessment() are used to extract the respective folds from dia_vfold. 
# Furthermore, the recipe dia_rec is prepped (i.e., trained) using the analysis data of each fold, 
# and this prepped recipe is then applied to the assessment data of each fold using bake(). 
# Preparing the recipe separately for each fold (rather than once for the whole training data set 
# dia_train) guards against data leakage.

# The code in the following chunk makes use of list columns to store all information about the 
# three folds in a single tibble lm_fit2, and a combination of dplyr::mutate() and purrr::map() 
# is used to “loop” across the three rows of the tibble.

# Evaluating Model Performance: yardstick ----
dia_vfold

# map(YOUR_LIST, YOUR_FUNCTION)
# Extract analysis/training and assessment/testing data
lm_fit2 <- mutate(dia_vfold,
                  df_ana = map (splits,  analysis),
                  df_ass = map (splits,  assessment))
lm_fit2

lm_fit2 <- 
    lm_fit2 %>% 
    # prep, juice, bake
    mutate(
        recipe = map (df_ana, ~prep(dia_rec, training = .x)),
        df_ana = map (recipe,  juice),
        df_ass = map2(recipe, 
                      df_ass, ~bake(.x, new_data = .y))) %>% 
    # fit
    mutate(
        model_fit  = map(df_ana, ~fit(lm_model, price ~ ., data = .x))) %>% 
    # predict
    mutate(
        model_pred = map2(model_fit, df_ass, ~predict(.x, new_data = .y)))

select(lm_fit2, id, recipe:model_pred)

lm_preds <- 
    lm_fit2 %>% 
    mutate(res = map2(df_ass, model_pred, ~data.frame(price = .x$price,
                                                      .pred = .y$.pred))) %>% 
    select(id, res) %>% 
    tidyr::unnest(res) %>% 
    group_by(id)
lm_preds

metrics(lm_preds, truth = price, estimate = .pred)

# Tuning Model Parameters: tune and dials ----
# Preparing a parsnip Model for Tuning

rf_model <- 
    rand_forest(mtry = tune()) %>%
    set_mode("regression") %>%
    set_engine("ranger")

parameters(rf_model)
mtry()

rf_model %>% 
    parameters() %>% 
    update(mtry = mtry(c(1L, 5L)))

rf_model %>% 
    parameters() %>% 
    # Here, the maximum of mtry equals the number of predictors, i.e., 24.
    finalize(x = select(juice(prep(dia_rec)), -price)) %>% 
    pull("object")

# Preparing Data for Tuning: recipes ----
# Note that this recipe cannot be prepped (and juiced), since "degree" is a
# tuning parameter
dia_rec2 <-
    recipe(price ~ ., data = dia_train) %>%
    step_log(all_outcomes()) %>%
    step_normalize(all_predictors(), -all_nominal()) %>%
    step_dummy(all_nominal()) %>%
    step_poly(carat, degree = tune())

dia_rec2 %>% 
    parameters() %>% 
    pull("object")

# Combine Everything: workflows ----
rf_wflow <-
    workflow() %>%
    add_model(rf_model) %>%
    add_recipe(dia_rec2)
rf_wflow

rf_param <-
    rf_wflow %>%
    parameters() %>%
    update(mtry = mtry(range = c(3L, 5L)),
           degree = degree_int(range = c(2L, 4L)))
rf_param$object

# use cross-validation for tuning, that is, 
# to select the best combination of the hyperparameters.
rf_grid <- grid_regular(rf_param, levels = 3)
rf_grid

# Cross-validation and hyperparameter tuning can involve fitting many models. 
# Herein, for example, we have to fit 3 x 9 models (folds x parameter combinations).

library("doFuture")
all_cores <- parallel::detectCores(logical = FALSE) - 1

registerDoFuture()
cl <- parallel::makeCluster(all_cores)
plan(future::cluster, workers = cl)

rf_search <- tune_grid(rf_wflow, grid = rf_grid, resamples = dia_vfold,
                       param_info = rf_param)

autoplot(rf_search, metric = "rmse") +
    labs(title = "Results of Grid Search for Two Tuning Parameters of a Random Forest")

show_best(rf_search, "rmse", n = 9)
select_best(rf_search, metric = "rmse")
select_by_one_std_err(rf_search, mtry, degree, metric = "rmse")

# Selecting the Best Model to Make the Final Predictions ----
rf_param_final <- select_by_one_std_err(rf_search, mtry, degree,
                                        metric = "rmse")

rf_wflow_final <- finalize_workflow(rf_wflow, rf_param_final)

rf_wflow_final_fit <- fit(rf_wflow_final, data = dia_train)

# Unfortunately, predict(rf_wflow_final_fit, new_data = dia_test) does not work 
# in the present case, because the outcome is modified in the recipe via step_log()

dia_rec3     <- workflows::extract_recipe(rf_wflow_final_fit)
rf_final_fit <- workflows::extract_fit_parsnip(rf_wflow_final_fit)

dia_test$.pred <- predict(rf_final_fit, 
                          new_data = bake(dia_rec3, dia_test))$.pred
dia_test$logprice <- log(dia_test$price)

metrics(dia_test, truth = logprice, estimate = .pred)
