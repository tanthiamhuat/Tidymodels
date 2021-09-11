# https://towardsdatascience.com/modelling-with-tidymodels-and-parsnip-bae2c01c131c

rm(list=ls())  # remove all variables
cat("\014")    # clear Console
if (dev.cur()!=1) {dev.off()} # clear R plots if exists

# load the relevant tidymodels libraries
library(tidymodels)
library(skimr)
library(tibble)

# Data preparation ----
telco <- readr::read_csv(url("https://raw.githubusercontent.com/DiegoUsaiUK/Classification_Churn_with_Parsnip/master/00_Data/WA_Fn-UseC_-Telco-Customer-Churn.csv"))
telco %>% 
    skimr::skim()

telco <- 
    telco %>%
    select(-customerID) %>%
    drop_na()

# Train and test split ----
set.seed(seed = 1972) 
train_test_split <-
    rsample::initial_split(
        data = telco,     
        prop = 0.80   
    ) 

train_tbl <- train_test_split %>% training() 
test_tbl  <- train_test_split %>% testing()

# A simple recipe ----
recipe_simple <- function(dataset) {
    recipe(Churn ~ ., data = dataset) %>%
        step_string2factor(all_nominal(), -all_outcomes()) %>%
        prep(data = dataset)
}

recipe_prepped <- recipe_simple(dataset = train_tbl)

train_baked <- bake(recipe_prepped, new_data = train_tbl)
test_baked  <- bake(recipe_prepped, new_data = test_tbl)

# Fit the model ----
logistic_glm <-
    logistic_reg(mode = "classification") %>%
    set_engine("glm") %>%
    fit(Churn ~ ., data = train_baked)

# Performance assessment ----
predictions_glm <- logistic_glm %>%
    predict(new_data = test_baked) %>%
    bind_cols(test_baked %>% select(Churn))
predictions_glm

predictions_glm %>%
    conf_mat(Churn, .pred_class) %>%
    pluck(1) %>%
    as_tibble() %>%
    ggplot(aes(Prediction, Truth, alpha = n)) +
    geom_tile(show.legend = FALSE) +
    geom_text(aes(label = n), colour = "white", alpha = 1, size = 8)

predictions_glm %>%
    metrics(Churn, .pred_class) %>%
    select(-.estimator) %>%
    filter(.metric == "accuracy") 

tibble(
    "precision" = 
        precision(predictions_glm, Churn, .pred_class) %>%
        select(.estimate),
    "recall" = 
        recall(predictions_glm, Churn, .pred_class) %>%
        select(.estimate)
) %>%
    unnest(cols = c(precision, recall)) 

predictions_glm %>%
    f_meas(Churn, .pred_class) %>%
    select(-.estimator) 

# A Random Forest ----
# Cross-validation set up ----
cross_val_tbl <- vfold_cv(train_tbl, v = 10)
cross_val_tbl

cross_val_tbl$splits %>%
    pluck(1)

# Update the recipe ----
recipe_rf <- function(dataset) {
    recipe(Churn ~ ., data = dataset) %>%
        step_string2factor(all_nominal(), -all_outcomes()) %>%
        step_dummy(all_nominal(), -all_outcomes()) %>%
        step_center(all_numeric()) %>%
        step_scale(all_numeric()) %>%
        prep(data = dataset)
}

# Estimate the model ----
rf_fun <- function(split, id, try, tree) {
    
    analysis_set <- split %>% analysis()
    analysis_prepped <- analysis_set %>% recipe_rf()
    analysis_baked <- analysis_prepped %>% bake(new_data = analysis_set)
    model_rf <-
        rand_forest(
            mode = "classification",
            mtry = try,
            trees = tree
        ) %>%
        set_engine("ranger",
                   importance = "impurity"
        ) %>%
        fit(Churn ~ ., data = analysis_baked)
    assessment_set <- split %>% assessment()
    assessment_prepped <- assessment_set %>% recipe_rf()
    assessment_baked <- assessment_prepped %>% bake(new_data = assessment_set)
    tibble(
        "id" = id,
        "truth" = assessment_baked$Churn,
        "prediction" = model_rf %>%
            predict(new_data = assessment_baked) %>%
            unlist()
    )
    
}

# Performance assessment ----
pred_rf <- map2_df(
    .x = cross_val_tbl$splits,
    .y = cross_val_tbl$id,
    ~ rf_fun(split = .x, id = .y, try = 3, tree = 200)
)
head(pred_rf)

pred_rf %>%
    conf_mat(truth, prediction) %>%
    summary() %>%
    select(-.estimator) %>%
    filter(.metric %in%
               c("accuracy", "precision", "recall", "f_meas"))
