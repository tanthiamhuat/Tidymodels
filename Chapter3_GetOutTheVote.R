# https://github.com/juliasilge/supervised-ML-case-studies-course
# https://supervised-ml-course.netlify.app/chapter3

rm(list=ls())  # remove all variables
cat("\014")    # clear Console
if (dev.cur()!=1) {dev.off()} # clear R plots if exists

library(tidyverse)
library(tidymodels)
library(themis)
voters <- read_csv("https://raw.githubusercontent.com/juliasilge/supervised-ML-case-studies-course/master/data/voters.csv")

# Take a look at voters
glimpse(voters)

# How many people voted?
voters %>%
    count(turnout16_2016)

# How do the responses on the survey vary with voting behavior?
voters %>%
    group_by(turnout16_2016) %>%
    summarise(`Elections don't matter` = mean(RIGGED_SYSTEM_1_2016 <= 2),
              `Economy is getting better` = mean(econtrend_2016 == 1),
              `Crime is very important` = mean(imiss_a_2016 == 2))

voters <- voters %>% mutate(turnout16_2016 = factor(turnout16_2016))

## Visualize difference by voter turnout
voters %>%
    ggplot(aes(econtrend_2016, after_stat(density), fill = turnout16_2016)) +
    geom_histogram(alpha = 0.5, position = "identity", binwidth = 1) +
    labs(title = "Overall, is the economy getting better or worse?")

voters_select <- voters %>%
    mutate(turnout16_2016 = factor(turnout16_2016)) %>% 
    select(-case_identifier)

# Split data into training and testing sets
set.seed(1234)
vote_split <- voters_select %>%  initial_split(prop = 0.8, strata = turnout16_2016)
vote_train <- training(vote_split)
vote_test <- testing(vote_split)

glimpse(vote_train)
glimpse(vote_test)

vote_recipe <- recipe(turnout16_2016 ~ ., data = vote_train) %>% 
    step_upsample(turnout16_2016)

## Specify a ranger model
rf_spec <- rand_forest() %>%
    set_engine("ranger") %>%
    set_mode("classification")

## Add the recipe + model to a workflow
vote_wf <- workflow() %>%
    add_recipe(vote_recipe) %>%
    add_model(rf_spec)

vote_wf

# Cross-validation
# Partitioning your data into subsets and using one subset for validation
# Cross-validation means taking your training set and randomly dividing it up evenly into subsets, 
# sometimes called "folds". A fold here means a group or subset or partition.

# You use one of the folds for validation and the rest for training, then you repeat these steps with all the subsets 
# and combine the results, usually by taking the mean. 
# The reason we do this is the same reason we would use bootstrap resampling; 
# cross-validation allows you to get a more accurate estimate of how your model will perform on new data.

vote_folds <- vfold_cv(vote_train, v = 10, repeats = 5)

# Evaluate models with resampling
vote_wf %>%
    fit_resamples(
        vote_folds,
        metrics = metric_set(roc_auc, sensitivity, specificity),
        control = control_resamples(save_pred = TRUE)
    )

# The fitted models themselves are not kept or stored because they are only used for computing performance metrics. 
# However, we are saving the predictions with save_pred = TRUE so we can build a confusion matrix, 
# and we have also set specific performance metrics to be computed (instead of the defaults) 
# with metric_set(roc_auc, sensitivity, specificity)

glm_spec <- logistic_reg() %>%
    set_engine("glm")

vote_wf <- workflow() %>%
    add_recipe(vote_recipe) %>%
    add_model(glm_spec)

set.seed(234)
glm_res <- vote_wf %>%
    fit_resamples(
        vote_folds,
        metrics = metric_set(roc_auc, sensitivity, specificity),
        control = control_resamples(save_pred = TRUE)
    )
glimpse(glm_res)
collect_metrics(glm_res)

rf_spec <- rand_forest() %>%
    set_engine("ranger") %>%
    set_mode("classification")

vote_wf <- workflow() %>%
    add_recipe(vote_recipe) %>%
    add_model(rf_spec)

set.seed(234)
rf_res <- vote_wf %>%
    fit_resamples(
        vote_folds,
        metrics = metric_set(roc_auc,sensitivity,specificity),
        control = control_resamples(save_pred = TRUE)
    )

glimpse(rf_res)

# Performance metrics from resampling
collect_metrics(glm_res)
collect_metrics(rf_res)

# Back to testing data
# When we used resampling to evaluate model performance with the training set, 
# the logistic regression model performed better. Now, let’s put this model to the test! 
# Let’s use the last_fit() function to fit to the entire training set one time and evaluate one time on the testing set.

## Model specification
glm_spec <- logistic_reg() %>%
    set_engine("glm")

## Combine in workflow
vote_wf <- workflow() %>%
    add_recipe(vote_recipe) %>%
    add_model(glm_spec)

## Final fit
# last_fit() function to fit to the entire training set one time and evaluate one time on the testing set
vote_final <- vote_wf %>%
    last_fit(vote_split)

## Confusion matrix
vote_final %>% 
    collect_predictions() %>% 
    conf_mat(turnout16_2016, .pred_class)
