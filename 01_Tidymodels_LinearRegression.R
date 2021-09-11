rm(list=ls())  # remove all variables
cat("\014")    # clear Console
if (dev.cur()!=1) {dev.off()} # clear R plots if exists

library(tidymodels)
data(ames)
ames <- mutate(ames, Sale_Price = log10(Sale_Price))

set.seed(123)
ames_split <- initial_split(ames, prop = 0.80, strata = Sale_Price)
ames_train <- training(ames_split)
ames_test  <-  testing(ames_split)

ames_rec <- 
    recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + 
               Latitude + Longitude, data = ames_train) %>%
    step_log(Gr_Liv_Area, base = 10) %>% 
    step_other(Neighborhood, threshold = 0.01) %>% 
    step_dummy(all_nominal_predictors()) %>% 
    step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") ) %>% 
    step_ns(Latitude, Longitude, deg_free = 20)

lm_model <- 
    linear_reg() %>% 
    set_engine("lm")

lm_wflow <- 
    workflow() %>% 
    add_model(lm_model) %>% 
    add_recipe(ames_rec)

# lm_fit <- fit(lm_wflow, ames_train)
# a=predict(lm_fit, ames_test)

lm_fit <- lm_wflow %>%
    # fit on the training set and evaluate on test set
    last_fit(ames_split, metrics = metric_set(rmse,mae,rsq))

collect_metrics(lm_fit)

# generate predictions from the test set
test_predictions <- lm_fit %>% collect_predictions()
test_predictions

lm_fit %>% 
    workflows::extract_fit_parsnip() %>% 
    broom::tidy()

SalePrice <- test_predictions %>% dplyr::select(Sale_Price,.pred) %>% 
                                  dplyr::mutate(Sale_Price = 10 ^ Sale_Price,
                                                Predicted = 10 ^ .pred) %>% 
                                  dplyr::select(-.pred)
