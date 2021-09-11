# https://www.rebeccabarter.com/blog/2020-03-25_machine_learning/

rm(list=ls())  # remove all variables
cat("\014")    # clear Console
if (dev.cur()!=1) {dev.off()} # clear R plots if exists

# load the relevant tidymodels libraries
library(tidymodels)
library(tidyverse)
library(workflows)
library(tune)

# load the Pima Indians dataset from the mlbench dataset
library(mlbench)
data(PimaIndiansDiabetes)
# rename dataset to have shorter name because lazy
diabetes_orig <- PimaIndiansDiabetes

ggplot(diabetes_orig) +
    geom_histogram(aes(x = triceps))

diabetes_clean <- diabetes_orig %>%
    mutate_at(vars(triceps, glucose, pressure, insulin, mass), 
              function(.var) { 
                  if_else(condition = (.var == 0), # if true (i.e. the entry is 0)
                          true = as.numeric(NA),  # replace the value with NA
                          false = .var # otherwise leave it as it is
                  )
              })

set.seed(234589)
# split the data into trainng (75%) and testing (25%)
diabetes_split <- initial_split(diabetes_clean, 
                                prop = 3/4)
diabetes_split

# extract training and testing sets
diabetes_train <- training(diabetes_split)
diabetes_test <- testing(diabetes_split)

# create CV object from training data
diabetes_cv <- vfold_cv(diabetes_train)

# define the recipe
diabetes_recipe <- 
    # which consists of the formula (outcome ~ predictors)
    recipe(diabetes ~ pregnant + glucose + pressure + triceps + 
               insulin + mass + pedigree + age, 
           data = diabetes_clean) %>%
    # and some pre-processing steps
    step_normalize(all_numeric()) %>%
    step_impute_knn(all_predictors())

diabetes_recipe

diabetes_train_preprocessed <- diabetes_recipe %>%
    # apply the recipe to the training data
    prep(diabetes_train) %>%
    # extract the pre-processed training dataset
    juice()
diabetes_train_preprocessed

rf_model <- 
    # specify that the model is a random forest
    rand_forest() %>%
    # specify that the `mtry` parameter needs to be tuned
    set_args(mtry = tune()) %>%
    # select the engine/package that underlies the model
    set_engine("ranger", importance = "impurity") %>%
    # choose either the continuous regression or binary classification mode
    set_mode("classification")

lr_model <- 
    # specify that the model is a random forest
    logistic_reg() %>%
    # select the engine/package that underlies the model
    set_engine("glm") %>%
    # choose either the continuous regression or binary classification mode
    set_mode("classification") 

fitted_logistic_model<- logistic_reg() %>%
    # Set the engine
    set_engine("glm") %>%
    # Set the mode
    set_mode("classification") %>%
    # Fit the model
    fit(diabetes~., data = diabetes_train)
tidy(fitted_logistic_model)    # Generate Summary Table

# Model Prediction
# (a) Class prediction
pred_class <- predict(fitted_logistic_model,
                      new_data = diabetes_test,
                      type = "class")
# (b) Test Data Class Probabilities
pred_proba <- predict(fitted_logistic_model,
                      new_data = diabetes_test,
                      type = "prob")

diabetes_results <- diabetes_test %>%
    select(diabetes) %>%
    bind_cols(pred_class, pred_proba)

conf_mat(diabetes_results, truth = diabetes,
         estimate = .pred_class)

accuracy(diabetes_results, truth = diabetes,
         estimate = .pred_class)

roc_auc(diabetes_results, truth = diabetes, .pred_neg)

diabetes_results %>%
    roc_curve(truth = diabetes, .pred_neg) %>%
    autoplot()

# set the workflow
rf_workflow <- workflow() %>%
    # add the recipe
    add_recipe(diabetes_recipe) %>%
    # add the model
    add_model(rf_model)

# specify which values eant to try
rf_grid <- expand.grid(mtry = c(3, 4, 5))
# extract results
rf_tune_results <- rf_workflow %>%
    tune_grid(resamples = diabetes_cv, #CV object
              grid = rf_grid, # grid of values to try
              metrics = metric_set(accuracy, roc_auc) # metrics we care about
    )

# print results
rf_tune_results %>%
    collect_metrics()

param_final <- rf_tune_results %>%
    select_best(metric = "accuracy")
param_final

rf_workflow <- rf_workflow %>%
    finalize_workflow(param_final)

rf_fit <- rf_workflow %>%
    # fit on the training set and evaluate on test set
    last_fit(diabetes_split, metrics = metric_set(recall,precision,accuracy,roc_auc))

test_performance <- rf_fit %>% collect_metrics()
test_performance

# generate predictions from the test set
test_predictions <- rf_fit %>% collect_predictions()
test_predictions

# generate a confusion matrix
test_predictions %>% 
    conf_mat(truth = diabetes, estimate = .pred_class)

## ROC Curve
rf_fit %>% collect_predictions() %>% 
    roc_curve(truth  = diabetes, estimate = .pred_neg) %>% 
    autoplot()

test_predictions %>%
    ggplot() +
    geom_density(aes(x = .pred_pos, fill = diabetes), 
                 alpha = 0.5)

test_predictions <- rf_fit %>% pull(.predictions)
test_predictions

final_model <- fit(rf_workflow, diabetes_clean)

new_woman <- tribble(~pregnant, ~glucose, ~pressure, ~triceps, ~insulin, ~mass, ~pedigree, ~age,
                     2, 95, 70, 31, 102, 28.2, 0.67, 47)
new_woman

predict(final_model, new_data = new_woman)

ranger_obj <- extract_fit_parsnip(final_model)$fit
ranger_obj

ranger_obj$variable.importance