# https://oliviergimenez.github.io/blog/learning-machine-learning/

rm(list=ls())  # remove all variables
cat("\014")    # clear Console
if (dev.cur()!=1) {dev.off()} # clear R plots if exists

# load the relevant tidymodels libraries
library(tidymodels)
library(skimr)
library(tibble)
library(naniar)

# Load Libraries ----
library(tidymodels) # metapackage for ML 
library(tidyverse) # metapackage for data manipulation and visulaisation
library(stacks) # stack ML models for better perfomance
theme_set(theme_light())
doParallel::registerDoParallel(cores = 4) # parallel computations

# Load Data
rawdata <- read_csv("E://Learning//data//titanic//train.csv")
glimpse(rawdata)

naniar::miss_var_summary(rawdata)

# Get most frequent port of embarkation
uniqx <- unique(na.omit(rawdata$Embarked))
mode_embarked <- as.character(fct_drop(uniqx[which.max(tabulate(match(rawdata$Embarked, uniqx)))]))

# Build function for data cleaning and handling NAs
process_data <- function(tbl){
    
    tbl %>%
        mutate(class = case_when(Pclass == 1 ~ "first",
                                 Pclass == 2 ~ "second",
                                 Pclass == 3 ~ "third"),
               class = as_factor(class),
               gender = factor(Sex),
               fare = Fare,
               age = Age,
               ticket = Ticket,
               alone = if_else(SibSp + Parch == 0, "yes", "no"), # alone variable
               alone = as_factor(alone),
               port = factor(Embarked), # rename embarked as port
               title = str_extract(Name, "[A-Za-z]+\\."), # title variable
               title = fct_lump(title, 4)) %>% # keep only most frequent levels of title
        mutate(port = ifelse(is.na(port), mode_embarked, port), # deal w/ NAs in port (replace by mode)
               port = as_factor(port)) %>%
        group_by(title) %>%
        mutate(median_age_title = median(age, na.rm = T)) %>%
        ungroup() %>%
        mutate(age = if_else(is.na(age), median_age_title, age)) %>% # deal w/ NAs in age (replace by median in title)
        mutate(ticketfreq = ave(1:nrow(.), FUN = length),
               fareadjusted = fare / ticketfreq) %>%
        mutate(familyage = SibSp + Parch + 1 + age/70)
    
}

# Process the data
dataset <- rawdata %>%
    process_data() %>%
    mutate(survived = as_factor(if_else(Survived == 1, "yes", "no"))) %>%
    mutate(survived = relevel(survived, ref = "yes")) %>% # first event is survived = yes
    select(survived, class, gender, age, alone, port, title, fareadjusted, familyage) 

# Have a look again
glimpse(dataset)

naniar::miss_var_summary(dataset)

rawdata <- read_csv("E://Learning//data//titanic//test.csv") 
holdout <- rawdata %>%
    process_data() %>%
    select(PassengerId, class, gender, age, alone, port, title, fareadjusted, familyage) 

glimpse(holdout)

naniar::miss_var_summary(holdout)

skimr::skim(dataset)
dataset %>%
    count(survived)

dataset %>%
    group_by(gender) %>%
    summarize(n = n(),
              n_surv = sum(survived == "yes"),
              pct_surv = n_surv / n)

dataset %>%
    group_by(title) %>%
    summarize(n = n(),
              n_surv = sum(survived == "yes"),
              pct_surv = n_surv / n) %>%
    arrange(desc(pct_surv))

dataset %>%
    group_by(class, gender) %>%
    summarize(n = n(),
              n_surv = sum(survived == "yes"),
              pct_surv = n_surv / n) %>%
    arrange(desc(pct_surv))

dataset %>%
    group_by(class, gender) %>%
    summarize(n = n(),
              n_surv = sum(survived == "yes"),
              pct_surv = n_surv / n) %>%
    mutate(class = fct_reorder(class, pct_surv)) %>%
    ggplot(aes(pct_surv, class, fill = class, color = class)) +
    geom_col(position = position_dodge()) +
    scale_x_continuous(labels = percent) +
    labs(x = "% in category that survived", fill = NULL, color = NULL, y = NULL) +
    facet_wrap(~gender)

dataset %>%
    mutate(age = cut(age, breaks = c(0, 20, 40, 60, 80))) %>%
    group_by(age, gender) %>%
    summarize(n = n(),
              n_surv = sum(survived == "yes"),
              pct_surv = n_surv / n) %>%
    mutate(age = fct_reorder(age, pct_surv)) %>%
    ggplot(aes(pct_surv, age, fill = age, color = age)) +
    geom_col(position = position_dodge()) +
    scale_x_continuous(labels = percent) +
    labs(x = "% in category that survived", fill = NULL, color = NULL, y = NULL) +
    facet_wrap(~gender)

dataset %>%
    ggplot(aes(fareadjusted, group = survived, color = survived, fill = survived)) +
    geom_histogram(alpha = .4, position = position_dodge()) +
    labs(x = "fare", y = NULL, color = "survived?", fill = "survived?")

dataset %>%
    ggplot(aes(familyage, group = survived, color = survived, fill = survived)) +
    geom_histogram(alpha = .4, position = position_dodge()) +
    labs(x = "family aged", y = NULL, color = "survived?", fill = "survived?")

set.seed(2021)
spl <- initial_split(dataset, strata = "survived")
train <- training(spl)
test <- testing(spl)

train_5fold <- train %>%
    vfold_cv(5)

# Gradient boosting algorithms - xgboost
mset <- metric_set(accuracy) # metric is accuracy
control <- control_grid(save_workflow = TRUE,
                        save_pred = TRUE,
                        extract = extract_model) # grid for tuning

xg_rec <- recipe(survived ~ ., data = train) %>%
    step_impute_median(all_numeric()) %>% # replace missing value by median
    step_dummy(all_nominal_predictors()) # all factors var are split into binary terms (factor disj coding)

xg_model <- boost_tree(mode = "classification", # binary response
                       trees = tune(),
                       mtry = tune(),
                       tree_depth = tune(),
                       learn_rate = tune(),
                       loss_reduction = tune(),
                       min_n = tune()) # parameters to be tuned

xg_wf <- 
    workflow() %>% 
    add_model(xg_model) %>% 
    add_recipe(xg_rec)

xg_tune <- xg_wf %>%
    tune_grid(train_5fold,
              metrics = mset,
              control = control,
              grid = crossing(trees = 1000,
                              mtry = c(3, 5, 8), # finalize(mtry(), train)
                              tree_depth = c(5, 10, 15),
                              learn_rate = c(0.01, 0.005),
                              loss_reduction = c(0.01, 0.1, 1),
                              min_n = c(2, 10, 25)))

# Visualize the results.
autoplot(xg_tune) + theme_light()

# Collect metrics.
xg_tune %>%
    collect_metrics() %>%
    arrange(desc(mean))

# Fit model
xg_fit <- xg_wf %>%
    finalize_workflow(select_best(xg_tune)) %>%
    fit(train)

xg_fit %>%
    augment(test, type.predict = "response") %>%
    accuracy(survived, .pred_class)

# Check out important features (aka predictors).
importances <- xgboost::xgb.importance(model = extract_fit_engine(xg_fit))
importances %>%
    mutate(Feature = fct_reorder(Feature, Gain)) %>%
    ggplot(aes(Gain, Feature)) +
    geom_col()

# Make predictions
xg_wf %>%
    finalize_workflow(select_best(xg_tune)) %>%
    fit(dataset) %>%
    augment(holdout) %>%
    select(PassengerId, Survived = .pred_class) %>%
    mutate(Survived = if_else(Survived == "yes", 1, 0)) %>%
    write_csv("E://Learning//data//titanic//xgboost.csv")
