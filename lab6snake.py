import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
import pandas as pd

# Initialize H2O
h2o.init(max_mem_size="4G", port=54321)

# Load the preprocessed Ames Housing data from CSV.
# (Ensure that you have saved "ames_housing.csv" using your R script.)
data = h2o.import_file("ames_housing.csv")

# Define the response and predictor variables.
response = "Sale_Price"
predictors = data.col_names
predictors.remove(response)

# Split the data into training (80%) and testing (20%) sets.
train, test = data.split_frame(ratios=[0.8], seed=123)

# Train a GLM model with 10-fold cross-validation.
best_glm = H2OGeneralizedLinearEstimator(
    family="gaussian",
    alpha=0.1,
    nfolds=10,
    fold_assignment="Modulo",
    keep_cross_validation_predictions=True,
    seed=123
)
best_glm.train(x=predictors, y=response, training_frame=train)

# Train a Random Forest model.
best_rf = H2ORandomForestEstimator(
    ntrees=1000,
    mtries=20,
    max_depth=30,
    min_rows=1,
    sample_rate=0.8,
    nfolds=10,
    fold_assignment="Modulo",
    keep_cross_validation_predictions=True,
    seed=123,
    stopping_rounds=50,
    stopping_metric="RMSE",
    stopping_tolerance=0
)
best_rf.train(x=predictors, y=response, training_frame=train)

# Train a GBM model.
best_gbm = H2OGradientBoostingEstimator(
    ntrees=1000,
    learn_rate=0.01,
    max_depth=7,
    min_rows=5,
    sample_rate=0.8,
    nfolds=10,
    fold_assignment="Modulo",
    keep_cross_validation_predictions=True,
    seed=123,
    stopping_rounds=50,
    stopping_metric="RMSE",
    stopping_tolerance=0
)
best_gbm.train(x=predictors, y=response, training_frame=train)

# Train a Deep Learning model (feedforward neural network).
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
best_dl.train(x=predictors, y=response, training_frame=train)

# Build a stacked ensemble using a DRF metalearner.
ensemble = H2OStackedEnsembleEstimator(
    metalearner_algorithm="drf",
    base_models=[best_glm.model_id, best_rf.model_id, best_gbm.model_id, best_dl.model_id]
)
ensemble.train(x=predictors, y=response, training_frame=train)

# Define a function to compute RMSE for a given model on the test set.
def get_rmse(model, test_frame):
    perf = model.model_performance(test_data=test_frame)
    return perf.rmse()

# Evaluate each base model and the ensemble on the test set.
rmse_glm = get_rmse(best_glm, test)
rmse_rf = get_rmse(best_rf, test)
rmse_gbm = get_rmse(best_gbm, test)
rmse_dl  = get_rmse(best_dl, test)
rmse_ensemble = get_rmse(ensemble, test)

print("GLM RMSE: ", rmse_glm)
print("RF RMSE: ", rmse_rf)
print("GBM RMSE: ", rmse_gbm)
print("Deep Learning RMSE: ", rmse_dl)
print("Stacked Ensemble RMSE: ", rmse_ensemble)

# Extract cross-validation predictions from each base model.
def get_cv_preds(model):
    # Returns a pandas Series containing the CV predictions.
    cv_frame = h2o.get_frame(model.cross_validation_holdout_predictions())
    return cv_frame['predict']

cv_glm = get_cv_preds(best_glm)
cv_rf = get_cv_preds(best_rf)
cv_gbm = get_cv_preds(best_gbm)
cv_dl = get_cv_preds(best_dl)

# Convert the H2OFrames to pandas DataFrames and combine them.
cv_df = pd.DataFrame({
    'GLM_pred': cv_glm.as_data_frame().values.flatten(),
    'RF_pred': cv_rf.as_data_frame().values.flatten(),
    'GBM_pred': cv_gbm.as_data_frame().values.flatten(),
    'DL_pred': cv_dl.as_data_frame().values.flatten()
})

print("Correlation Matrix among Cross-Validation Predictions:")
print(cv_df.corr())