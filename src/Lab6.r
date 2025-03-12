############################################################
##  Lab 6
##  Super Learning with Economic Data
##  Adapted from Hands-on Machine Learning with R
##  Ames Iowa Housing Data from De Cock (2011)
############################################################

############################################################
##  Load Libraries
##  
############################################################

library(rsample)
library(recipes)
library(h2o)
library(AmesHousing)

h2o.init()
## If this gives an error you may need to install Java from here:
#https://www.oracle.com/java/technologies/downloads/

###################################
##	Data Pre-processing:
###################################

# Load and split the Ames housing data into training and testing
ames <- AmesHousing::make_ames()
set.seed(123)  # for reproducibility
# Splits dataset stratified by sale price so that training and testing have
# even mix of prices
split <- initial_split(ames, strata = "Sale_Price")
ames_train <- training(split)
ames_test <- testing(split)

blueprint <- recipe(Sale_Price ~ ., data = ames_train) %>%
  step_other(all_nominal(), threshold = 0.005)

# Apply preprocessing to training and test sets
ames_train_preprocessed <- prep(blueprint, training = ames_train, retain = TRUE) %>%
  bake(new_data = ames_train)

ames_test_preprocessed <- prep(blueprint, training = ames_train, retain = TRUE) %>%
  bake(new_data = ames_test)

# Combine train and test data into one dataset for saving
ames_combined <- rbind(ames_train_preprocessed, ames_test_preprocessed)

# Save the same training data to train DL in python bc i bad at R
write.csv(ames_combined, "ames_housing.csv", row.names = FALSE)


# Make sure we have consistent categorical levels
# pull out any categories that have low frequency into "other" category
blueprint <- recipe(Sale_Price ~ ., data = ames_train) %>%
  step_other(all_nominal(), threshold = 0.005)

# Create training & test sets for h2o modeling
train_h2o <- prep(blueprint, training = ames_train, retain = TRUE) %>%
  bake(new_data = ames_train) %>%
  as.h2o()
test_h2o <- prep(blueprint, training = ames_train) %>%
  bake(new_data = ames_test) %>%
  as.h2o()

# Get response and feature names
Y <- "Sale_Price"
X <- setdiff(names(ames_train), Y)


###################################
##	Model Building:
###################################

# Below we build a series of base learners that are then fed into
# an ensemble (stacked) learner, to do this all base learners must:
# 1. trained on the same training set.
# 2. trained with the same number of CV folds.
# 3. use the same fold assignment to ensure the same observations are used 
#    (we can do this by using fold_assignment = "Modulo").
# 4. the cross-validated predictions from all of the models must be preserved 
#    by setting keep_cross_validation_predictions = TRUE. This is the data which 
#    is used to train the meta learner algorithm in the ensemble.

# Train & cross-validate a GLM model
best_glm <- h2o.glm(
  x = X, y = Y, training_frame = train_h2o, alpha = 0.1,
  remove_collinear_columns = TRUE, nfolds = 10, fold_assignment = "Modulo",
  keep_cross_validation_predictions = TRUE, seed = 123
)

# Train & cross-validate a RF model
best_rf <- h2o.randomForest(
  x = X, y = Y, training_frame = train_h2o, ntrees = 1000, mtries = 20,
  max_depth = 30, min_rows = 1, sample_rate = 0.8, nfolds = 10,
  fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE,
  seed = 123, stopping_rounds = 50, stopping_metric = "RMSE",
  stopping_tolerance = 0
)

# Train & cross-validate a GBM model
best_gbm <- h2o.gbm(
  x = X, y = Y, training_frame = train_h2o, ntrees = 1000, learn_rate = 0.01,
  max_depth = 7, min_rows = 5, sample_rate = 0.8, nfolds = 10,
  fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE,
  seed = 123, stopping_rounds = 50, stopping_metric = "RMSE",
  stopping_tolerance = 0
)

# Train & cross-validate a deep learning model (feedforward neural network)
#best_dl <- h2o.deeplearning(
 # x = X, y = Y, training_frame = train_h2o, nfolds = 10, fold_assignment = "Modulo", 
  #epochs = 100, hidden = c(500,500),
  #keep_cross_validation_predictions = TRUE, seed = 123, stopping_rounds = 50,
  #stopping_metric = "RMSE", stopping_tolerance = 0
#)
# could not for the life of me get this model to train in R using h2o so I did 
# in python and saved the model and loaded it here
best_dl <- h2o.loadModel("DeepLearning_model_python_1741804463053_1")
print(best_dl)


#####
## Question #1 - Briefly describe how each of the above models (GLM, RF, GBM, DL) operates, 
##               which do you expect to have the best performance? (30 points)
#####
## GLM 


# Train a stacked tree ensemble
# Here we supply a list of the base learners trained above
# We use a random forest model as the metalearner but could use other model types
# See ?h2o.stackedEnsemble
ensemble_tree <- h2o.stackedEnsemble(
  x = X, y = Y, training_frame = train_h2o, model_id = "my_tree_ensemble",
  base_models = list(best_glm, best_rf, best_gbm, best_dl),
  metalearner_algorithm = "drf"
)

#####
## Question #2 - What is the advantage of using an ensemble learner? 
##               (10 points)
#####



###################################
##	Model Output:
###################################

# Get results from base learners
get_rmse <- function(model) {
  results <- h2o.performance(model, newdata = test_h2o)
  results@metrics$RMSE
}
list(best_glm, best_rf, best_gbm, best_dl) %>%
  purrr::map_dbl(get_rmse)


# Stacked results
h2o.performance(ensemble_tree, newdata = test_h2o)@metrics$RMSE


# Check out the correlation among predictions across all models
data.frame(
  GLM_pred = as.vector(h2o.getFrame(best_glm@model$cross_validation_holdout_predictions_frame_id$name)),
  RF_pred = as.vector(h2o.getFrame(best_rf@model$cross_validation_holdout_predictions_frame_id$name)),
  GBM_pred = as.vector(h2o.getFrame(best_gbm@model$cross_validation_holdout_predictions_frame_id$name)),
  DL_pred = as.vector(h2o.getFrame(best_dl@model$cross_validation_holdout_predictions_frame_id$name))
) %>% cor()


#####
## Question #3 - Look at the correlations among model predictions from the above code, 
##               why do you think there is not a huge improvement when using the ensemble 
##               learner versus the individual models and how could ensemble performance 
##               be improved? (20 points)
#####


