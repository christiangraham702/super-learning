import h2o
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
import os

# Initialize H2O (make sure this connects to your H2O cluster)
h2o.init(max_mem_size="2G", port=54321)

# Load the Ames Housing data.
# Option 1: Load from a CSV file (ensure this file matches the data used in R)
data = h2o.import_file("ames_housing.csv")

# Option 2: If you already pre-processed the data in R and saved it as CSV,
# use that file here so that the data is identical.
#
# Identify the response and predictor variables.
response = "Sale_Price"
predictors = data.col_names
predictors.remove(response)

# Split the data into training and test sets (approximate to R’s 80/20 split)
train, test = data.split_frame(ratios=[0.8], seed=123)

# Build the Deep Learning model.
# These parameters mirror your R code:
# - Two hidden layers with 500 neurons each
# - 100 epochs
# - 10-fold cross-validation with fold_assignment "Modulo"
# - Early stopping parameters as in the R script
best_dl = H2ODeepLearningEstimator(
    epochs=100,
    hidden=[500, 500],
    nfolds=10,
    fold_assignment="Modulo",
    keep_cross_validation_predictions=True,
    seed=123,
    stopping_rounds=50,
    stopping_metric="RMSE",
    stopping_tolerance=0
)

# Train the model on the training set
best_dl.train(x=predictors, y=response, training_frame=train)

# Evaluate performance on the test set and print RMSE
perf = best_dl.model_performance(test_data=test)
print("Test RMSE:", perf.rmse())

# Save the trained model to disk
model_path = h2o.save_model(model=best_dl, path=os.getcwd(), force=True)
print("Model saved to:", model_path)