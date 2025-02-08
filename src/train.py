from datetime import datetime
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import numpy as np

def create_pipeline():

    # Define numerical and categorical columns
    numerical_cols = ["trip_seconds", "trip_miles", "avg_tips"]
    categorical_cols = ["company", "payment_type"]

    # Preprocessing pipeline
    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ]), numerical_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_cols)
        ],
        remainder="drop"
    )

    # Define the model with initial/default parameters (to be tuned)
    model = XGBRegressor(
        n_estimators=1000,  # Default; will be tuned
        learning_rate=0.1,  # Default; will be tuned
        max_depth=6,  # Default; will be tuned
        random_state=42,
        objective="reg:squarederror"
    )

    # Create a pipeline that chains preprocessing and modeling
    pipeline = Pipeline(steps=[
        ("preprocessing", preprocess),
        ("model", model)
    ])

    return pipeline


def train_pipeline(pipeline, data, experiment, cv=5 ):

    # Split data into features and target
    X = data[["trip_seconds", "trip_miles", "company", "payment_type", "avg_tips"]]
    y = data["trip_total"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Define parameter grid for hyperparameter search.
    # Note: Parameters for the XGBRegressor are referenced as "model__<param_name>".
    param_grid = {
        "model__n_estimators": [100, 300, 500],
        "model__max_depth": [3, 6, 9],
        "model__learning_rate": [0.01, 0.1, 0.2]
    }

    run_name = datetime.now().strftime("%Y-%m-%d_%H:%M")
    tags = {
        "env": "Production",
        "model_type": "XGB Regressor",
        "experiment_description": "Taxi Regressor"
    }
    # Start a parent MLflow run for the grid search
    with mlflow.start_run(run_name=run_name, experiment_id=experiment.experiment_id, tags=tags) as parent_run:
        # Log the parameter grid (optional)
        mlflow.log_param("param_grid", param_grid)

        # Create and fit GridSearchCV
        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            return_train_score=True
        )
        grid_search.fit(X_train, y_train)

        # Extract best parameters and best score
        best_params = grid_search.best_params_

        # Log best parameters and best cross-validation MSE in the parent run
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

        # Log each individual run from the grid search as a nested MLflow run
        cv_results = grid_search.cv_results_
        for i, params in enumerate(cv_results["params"]):
            with mlflow.start_run(run_name=f"Nested_Run_{i}", nested=True, experiment_id=experiment.experiment_id, tags=tags):
                # Log parameters for this run
                mlflow.log_params(params)
                # Log mean and standard deviation of the test scores
                mlflow.log_metric("mean_test_score", cv_results["mean_test_score"][i])
                mlflow.log_metric("std_test_score", cv_results["std_test_score"][i])
                # Optionally log additional metrics from the training scores:
                mlflow.log_metric("mean_train_score", cv_results["mean_train_score"][i])

        # Retrieve the best pipeline (model) from grid search
        best_pipeline = grid_search.best_estimator_

        # Evaluate the best estimator on the test set
        y_pred = best_pipeline.predict(X_test)

        metrics = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2": r2_score(y_test, y_pred)
        }


    run_id = parent_run.info.run_id

    return best_pipeline, metrics, best_params, X_train, y_train, X_test, y_test, run_id

