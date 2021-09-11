# https://github.com/juliasilge/supervised-ML-case-studies-course
# https://supervised-ml-course.netlify.app/chapter4

rm(list=ls())  # remove all variables
cat("\014")    # clear Console
if (dev.cur()!=1) {dev.off()} # clear R plots if exists

library(tidyverse)
library(tidymodels)
library(themis)
sisters67 <- read_csv("https://raw.githubusercontent.com/juliasilge/supervised-ML-case-studies-course/master/data/sisters_select.csv")

# Visualization Age distribution ----

glimpse(sisters67)

glimpse(sisters67)

# Plot the histogram
ggplot(sisters67, aes(x = age)) +
    geom_histogram(binwidth = 10)

# Tidy the data set
tidy_sisters <- sisters67 %>%
    select(-sister) %>%
    pivot_longer(-age, names_to = "question", values_to = "rating")

# Print the structure of tidy_sisters
glimpse(tidy_sisters)

# Overall agreement with all questions by age
tidy_sisters %>%
    group_by(age) %>%
    summarize(rating = mean(rating, na.rm = TRUE))

# Number of respondents agreed or disagreed overall
tidy_sisters %>%
    count(rating)

tidy_sisters %>%
    filter(question %in% paste0("v", 153:170)) %>%
    group_by(question, rating) %>%
    summarise(age = mean(age)) %>%
    ggplot(aes(rating, age, color = question)) +
    geom_line(alpha = 0.5, size = 1.5) +
    geom_point(size = 2) +
    facet_wrap(~question)

# Training, validation and testing data ----
set.seed(123)
sisters_select <- sisters67
sisters_splits <- initial_split(sisters_select, strata = age)

sisters_other <- training(sisters_splits)
sisters_test <- testing(sisters_splits)
# The final evaluation is done with the testing set. 
# It is important to do this final evaluation with a separate data set so that you do not underestimate your error on new data.

set.seed(123)
sisters_val <- validation_split(sisters_other, strata = age)
# A validation test is used to compare models or tune hyperparameters.

sisters_val

# Hyperparameter Tuning
# Some model parameters cannot be learned directly from a dataset during model training; 
# These kinds of parameters are called hyperparameters. 
# Some examples of hyperparameters include the number of predictors that are sampled at splits in a tree-based model 
# (we call this mtry in tidymodels) or the learning rate in a boosted tree model (we call this learn_rate).
# Instead of learning these kinds of hyperparameters during model training, 
# we can estimate the best values for these values by training many models on a resampled data set 
# (like the validation set you just created) and measuring how well all these models perform. This process is called tuning.

tree_spec <- decision_tree(
    cost_complexity = tune(),
    tree_depth = tune()
) %>% 
    set_engine("rpart") %>% 
    set_mode("regression")

# Preprocessing and tuning
# Model hyperparameters aren't the only things you can tune. 
# You can also tune steps in your preprocessing pipeline. This recipe has two steps:
# First, this recipe centers and scales all those numeric predictors we have in this dataset, 
# cataloging the nuns' responses to the survey questions.
# Second, this recipe implements principal component analysis on these same predictors. 
# Except... this recipe identifies that we want to implement PCA and we aren't sure how many predictors we should use. 
# We want to choose the best number of predictors.

# Identify tuning parameters ----
sisters_recipe <- recipe(age ~ ., data = sisters_other) %>% 
    step_normalize(all_predictors()) %>%
    step_pca(all_predictors(), num_comp = tune())

# Create a tuning grid ----
# Grid of tuning parameters
tree_grid <- grid_regular(num_comp(c(3, 12)),
             cost_complexity(),
             tree_depth(),
             levels = 5)

# we are using default ranges for cost complexity and tree depth, and we are going to try 3 to 12 principal components. 
# When we set levels = 5, we are saying we want five levels for each parameter, 
# which means there will be 125 (5 * 5 * 5) total models to try.

# You can use the function tune_grid() to fit all these models; you can tune either a workflow or a model specification 
# with a set of resampled data, such as the validation set you created.

# You train these 125 possible models on the training data and use the validation data to compare all the results in terms of
# performance. We won't use the testing data until the very end of our modeling process, 
# when we use it to estimate how our model will perform on new data.

# For some modeling use cases, an approach with three data partitions is overkill, perhaps a bit too much, 
# but if you have enough data that you can use some of these powerful machine learning algorithms or techniques, 
# the danger you face is underestimating your uncertainty for new data if you estimate it with data that you used 
# to pick a model.

# To get a reliable estimate from tuning, for example, you need to use another heldout dataset for assessing the models, 
# either a validation set or a set of simulated datasets created through resampling.

# Use your validation set to find which values of the parameters 
# (cost complexity, tree depth, and number of principal components) result in the highest R-squared and lowest RMSE.

# Time to tune ----
tree_wf <- workflow() %>%
    add_recipe(sisters_recipe) %>%
    add_model(tree_spec)

tree_wf

set.seed(123)
tree_res <- tune_grid(
    tree_wf,
    resamples = sisters_val,
    grid = tree_grid
)

glimpse(tree_res)

# Visualize tuning results ----
tree_metrics <- tree_res %>%
    collect_metrics()

glimpse(tree_metrics)

tree_metrics %>%
    mutate(tree_depth = factor(tree_depth),
           num_comp = paste("num_comp =", num_comp),
           num_comp = fct_inorder(num_comp)) %>%
    ggplot(aes(cost_complexity, mean, color = tree_depth)) +
    geom_line(size = 1.5, alpha = 0.6) +
    geom_point(size = 2) +
    scale_x_log10() +
    facet_grid(.metric ~ num_comp, scales = "free")

# Find the best parameters ----
best_tree <- tree_res %>%
    select_best("rmse")

best_tree

final_wf <- tree_wf %>% 
    finalize_workflow(best_tree)

final_wf

# Using testing data ----
# Fit to the training set and evaluate on the testing set using last_fit()
# Using the testing data ----
final_tree <- final_wf %>%
    last_fit(sisters_splits) 

final_tree %>%
    collect_metrics()
