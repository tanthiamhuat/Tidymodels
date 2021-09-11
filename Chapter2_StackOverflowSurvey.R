# https://github.com/juliasilge/supervised-ML-case-studies-course
# https://supervised-ml-course.netlify.app/chapter2

rm(list=ls())  # remove all variables
cat("\014")    # clear Console
if (dev.cur()!=1) {dev.off()} # clear R plots if exists

library(tidyverse)
library(tidymodels)
library(themis)
stack_overflow <- read_csv("https://raw.githubusercontent.com/juliasilge/supervised-ML-case-studies-course/master/data/stack_overflow.csv")

# a) Explore Stack Overflow Survey ----
# Take a look at stack_overflow
glimpse(stack_overflow)

# First count for `remote`
stack_overflow %>% 
    count(remote, sort = TRUE)

# then count for `country`
stack_overflow %>% 
    count(country, sort = TRUE)

ggplot(stack_overflow, aes(remote, years_coded_job)) +
    geom_boxplot() +
    labs(x = NULL,
         y = "Years of professional coding experience")

# b) Training and Testing data ----
stack_overflow <- stack_overflow %>%
    mutate(remote = factor(remote, levels = c("Remote", "Not remote"))) %>%
    mutate_if(is.character, factor)

# Create stack_select dataset
stack_select <- stack_overflow %>%
    select(-respondent)

# Split the data into training and testing sets
set.seed(1234)
stack_split <- stack_select %>%
    initial_split(prop=0.8,
                  strata = remote)

stack_train <- training(stack_split)
stack_test <- testing(stack_split)

glimpse(stack_train)
glimpse(stack_test)

dim(stack_train)
# c) Dealing with Imbalanced data, and preprocess with a recipe ----
stack_recipe <- recipe(remote ~ ., data = stack_train) %>% 
    step_downsample(remote)

stack_recipe

stack_prep <- prep(stack_recipe )
stack_down <- bake(stack_prep,new_data = NULL)

stack_down %>%
    count(remote)

# d) Train models ----
## Build a logistic regression model
glm_spec <- logistic_reg() %>%
    set_engine("glm")

## Start a workflow (recipe only)
stack_wf <- workflow() %>%
    add_recipe(stack_recipe)

## Add the model and fit the workflow
stack_glm <- stack_wf %>%
    add_model(glm_spec) %>%
    fit(data = stack_train)

# Print the fitted model
stack_glm

## Build a decision tree model
tree_spec <- decision_tree() %>%         
    set_engine("rpart") %>%      
    set_mode("classification") 

## Start a workflow (recipe only)
stack_wf <- workflow() %>%
    add_recipe(stack_recipe)

## Add the model and fit the workflow
stack_tree <- stack_wf %>%
    add_model(tree_spec) %>%
    fit(data = stack_train)

# Print the fitted model
stack_tree

# e) Confusion Matrix ----
results_glm <- stack_test %>%
     bind_cols(predict(stack_glm, stack_test) %>%
        rename(.pred_glm = .pred_class))

# Confusion matrix for logistic regression model
results_glm %>%
    conf_mat(truth = remote, estimate = .pred_glm)

results_tree <- stack_test %>%
    bind_cols(predict(stack_tree,stack_test) %>%
         rename(.pred_tree = .pred_class))

# Confusion matrix for decision tree model
results_tree %>%
    conf_mat(truth = remote, estimate = .pred_tree)

# f) Classification Model metrics ----
results <- stack_test %>%
    bind_cols(predict(stack_glm, stack_test) %>%
                  rename(.pred_glm = .pred_class)) %>%
    bind_cols(predict(stack_tree, stack_test) %>%
                  rename(.pred_tree = .pred_class))

## Calculate accuracy
accuracy(results, truth = remote, estimate = .pred_glm)
accuracy(results, truth = remote, estimate = .pred_tree)

## Calculate positive predict value
ppv(results, truth = remote, estimate = .pred_glm)
ppv(results, truth = remote, estimate = .pred_tree)

# ROC ----
roc_glm <- stack_test %>%
    bind_cols(predict(stack_glm, stack_test, type = 'prob') %>%
                  rename(.pred_Remote_glm = .pred_Remote)) %>% 
    roc_curve(truth  = remote, estimate = .pred_Remote_glm) %>% 
    mutate(model = "GLM (Logistic Regression)") 

roc_tree <- stack_test %>%
    bind_cols(predict(stack_tree, stack_test, type = 'prob') %>%
                  rename(.pred_Remote_glm = .pred_Remote)) %>% 
    roc_curve(truth  = remote, estimate = .pred_Remote_glm) %>% 
    mutate(model = "Decision Tree") 

full_plot <- bind_rows(roc_glm, roc_tree) %>% 
    ggplot(aes(x = 1 - specificity, 
               y = sensitivity, 
               color = model)) + 
    geom_path(lwd = 1, alpha = 0.5) +
    geom_abline(lty = 3) + 
    scale_color_manual(
        values = c("#374785", "#E98074")
    ) +
    theme_minimal() +
    theme(legend.position = "top",
          legend.title = element_blank()) 

roc_auc_glm <- stack_test %>%
    bind_cols(predict(stack_glm, stack_test, type = 'prob') %>%
                  rename(.pred_Remote_glm = .pred_Remote)) %>% 
    roc_auc(truth  = remote, estimate = .pred_Remote_glm) %>% 
    mutate(model = "GLM (Logistic Regression)") 

roc_auc_tree <- stack_test %>%
    bind_cols(predict(stack_tree, stack_test, type = 'prob') %>%
                  rename(.pred_Remote_tree = .pred_Remote)) %>% 
    roc_auc(truth  = remote, estimate = .pred_Remote_tree) %>% 
    mutate(model = "Decision Tree") 

label_DT <- paste0("AUC(DecisionTree) = ",round(roc_auc_tree$.estimate,2))
label_GLM <- paste0("AUC(GLM) = ",round(roc_auc_glm$.estimate,2))
label_DTGLM <- paste(label_DT,label_GLM,sep="\n")
full_plot <- full_plot + annotate("text", x = 0.5, y = 0.95, label = label_DTGLM) + coord_fixed()
