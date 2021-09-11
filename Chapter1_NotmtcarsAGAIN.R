# https://github.com/juliasilge/supervised-ML-case-studies-course
# https://github.com/juliasilge/supervised-ML-case-studies-course/blob/master/chapters/chapter1.md
# https://github.com/juliasilge/supervised-ML-case-studies-course/issues/47

rm(list=ls())  # remove all variables
cat("\014")    # clear Console
if (dev.cur()!=1) {dev.off()} # clear R plots if exists

library(tidyverse)
library(tidymodels)
cars2018 <- read_csv("https://raw.githubusercontent.com/juliasilge/supervised-ML-case-studies-course/master/data/cars2018.csv")

# a) Visualization data ----
# Print the cars2018 object
glimpse(cars2018)

# Plot the histogram
ggplot(cars2018, aes(x = mpg)) +
    geom_histogram(bins = 25) +
    labs(x = "Fuel efficiency (mpg)",
         y = "Number of cars")

# Deselect the 2 columns to create cars_vars
car_vars <- cars2018 %>%
    select(-model, -model_index) %>%
    #janitor::clean_names() %>%
    mutate_if(is.character, factor)

# b) Build Simple Linear Model ----
# Fit a linear model
fit_all <- lm(mpg ~ ., data = car_vars)

# Print the summary of the model
summary(fit_all)

# c) Training and testing data ----
# Split the data into training and test sets
set.seed(1234)
car_split <- car_vars %>%
    initial_split(prop = 0.8, strata = transmission)

car_train <- training(car_split)
car_test <- testing(car_split)

glimpse(car_train)
glimpse(car_test)

# d) Train models with tidymodels ----
# Build a linear regression model specification
lm_mod <- linear_reg() %>%
    set_engine("lm")

# Train a linear regression model
fit_lm <- lm_mod %>%
    fit(log(mpg) ~ ., 
        data = car_train)

# Print the model object
fit_lm

# Build a random forest model specification
rf_mod <- rand_forest() %>%
    set_engine("randomForest") %>%
    set_mode("regression")

# Train a random forest model
fit_rf <- rf_mod %>%
    fit(log(mpg) ~ ., 
        data = car_train)

# Print the model object
fit_rf

## need to change type character to factor
## car_train <- car_train %>% mutate_if(is.character,as.factor)
# e) Evaluate Model Performance ----
# Create the new columns
results <- car_train %>%
    mutate(mpg = log(mpg)) %>%
    bind_cols(predict(fit_lm, car_train) %>%
                  rename(.pred_lm = .pred)) %>%
    bind_cols(predict(fit_rf, car_train) %>%
                  rename(.pred_rf = .pred))

# Evaluate the performance
metrics(results, truth = mpg, estimate = .pred_lm)
metrics(results, truth = mpg, estimate = .pred_rf)

# f) Using Testing Data ----
# Create the new columns
results <- car_test %>%
    mutate(mpg = log(mpg)) %>%
    bind_cols(predict(fit_lm, car_test) %>%
                  rename(.pred_lm = .pred)) %>%
    bind_cols(predict(fit_rf, car_test) %>%
                  rename(.pred_rf = .pred))

# Evaluate the performance
metrics(results, truth = mpg, estimate = .pred_lm)
metrics(results, truth = mpg, estimate = .pred_rf)

# g) Bootstrap Resampling ----
## Create bootstrap resamples
car_boot <- bootstraps(car_train)

# Evaluate the models with bootstrap resampling
lm_res <- lm_mod %>%
    fit_resamples(
        log(mpg) ~ .,
        resamples = car_boot,
        control = control_resamples(save_pred = TRUE)
    )

rf_res <- rf_mod %>%
    fit_resamples(
        log(mpg) ~ .,
        resamples = car_boot,
        control = control_resamples(save_pred = TRUE)
    )

glimpse(rf_res)

# h) Plot modeling results ----
results <-  bind_rows(lm_res %>%
                          collect_predictions() %>%
                          mutate(model = "lm"),
                      rf_res %>%
                          collect_predictions() %>%
                          mutate(model = "rf"))

glimpse(results)

results %>%
    ggplot(aes(`log(mpg)`, .pred)) +
    geom_abline(lty = 2, color = "gray50") +
    geom_point(aes(color = id), size = 1.5, alpha = 0.3, show.legend = FALSE) +
    geom_smooth(method = "lm") +
    facet_wrap(~ model)
