import logging
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Get the global logger (inherits from main.py)
logger = logging.getLogger(__name__)

def create_pipeline() -> Pipeline:
    """
    Creates a machine learning pipeline for preprocessing and training an XGBoost regression model.

    Returns:
        Pipeline: A scikit-learn Pipeline that includes preprocessing and an XGBoost regressor.
    """
    logger.info("Creating ML pipeline with preprocessing and XGBoost model.")

    # Define numerical and categorical columns
    categorical_cols = ["payment_type", "company", "day_type"]
    numerical_cols = ["trip_miles", "tolls", "extras", "daytime", "month", "day_of_week", "day_of_month",
                      "avg_tips", "pickup_latitude", "pickup_longitude", "pickup_community_area"]

    # Define the column transformer for preprocessing
    column_transformer = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", RobustScaler())
            ]), numerical_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), categorical_cols)
        ],
        remainder="drop"
    )

    # Wrap preprocessing in a pipeline
    preprocess = Pipeline([
        ("column_transformer", column_transformer)
    ])

    # Define the XGBoost model
    model = XGBRegressor(
        subsample=0.8,  # Train on 80% of data each iteration
        colsample_bytree=0.8,  # Use 80% of features per tree
        n_estimators=500,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
        objective="reg:squarederror"
    )

    # Create final pipeline
    pipeline = Pipeline(steps=[
        ("preprocessing", preprocess),
        ("model", model)
    ])

    logger.info("Pipeline created successfully.")
    return pipeline


def train_pipeline(pipeline: Pipeline, data: pd.DataFrame):
    """
    Trains the machine learning pipeline and evaluates its performance.

    Args:
        pipeline (Pipeline): The preprocessing and model pipeline.
        data (pd.DataFrame): The input dataset containing features and target variable.

    Returns:
        tuple:
            - pipeline (Pipeline): The trained pipeline.
            - metrics (dict): A dictionary containing evaluation metrics.
            - X_train (pd.DataFrame): Training feature set.
            - y_train (pd.Series): Training target values.
            - X_test (pd.DataFrame): Test feature set.
            - y_test (pd.Series): Test target values.
            - params (dict): The model parameters.

    Raises:
        ValueError: If `data` is not a DataFrame, is empty, or lacks required columns.
    """
    logger.info("Starting pipeline training.")

    # Validate input data
    if not isinstance(data, pd.DataFrame):
        logger.error("Invalid input: Data is not a pandas DataFrame.")
        raise ValueError("Input data must be a pandas DataFrame.")
    if data.empty:
        logger.error("Input data is empty.")
        raise ValueError("Input data is empty.")


    # Define required columns
    required_columns = {
        "trip_total", "trip_miles", "tolls", "extras", "avg_tips",
        "daytime", "day_of_week", "day_of_month", "month", "payment_type",
        "company", "day_type", "pickup_latitude", "pickup_longitude", "pickup_community_area"
    }
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        raise ValueError(f"Missing required columns: {missing_columns}")

    logger.info("Splitting data into features (X) and target (y).")
    X = data[[
        "trip_miles", "tolls", "extras", "avg_tips",
        "daytime", "day_of_week", "day_of_month", "month", "payment_type",
        "company", "day_type", "pickup_latitude", "pickup_longitude", "pickup_community_area"]]
    y = data["trip_total"]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    logger.info(f"Data split into training (size: {len(X_train)}) and test (size: {len(X_test)}) sets.")

    # Train the pipeline
    logger.info("Training the pipeline...")
    pipeline.fit(X_train, y_train)
    logger.info("Pipeline training completed.")

    # Evaluate the model
    logger.info("Evaluating model performance on test set.")
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Compute standardized metrics
    mean_actual = np.mean(y_test)
    actual_range = np.ptp(y_test)  # Max - Min

    mae_standardized_mean = mae / mean_actual if mean_actual != 0 else np.nan
    rmse_standardized_mean = rmse / mean_actual if mean_actual != 0 else np.nan

    mae_standardized_range = mae / actual_range if actual_range != 0 else np.nan
    rmse_standardized_range = rmse / actual_range if actual_range != 0 else np.nan

    # Collect metrics
    metrics = {
        "MAE": mae,
        "MAE_standardized_by_mean": mae_standardized_mean,
        "MAE_standardized_by_range": mae_standardized_range,
        "MSE": mse,
        "RMSE": rmse,
        "RMSE_standardized_by_mean": rmse_standardized_mean,
        "RMSE_standardized_by_range": rmse_standardized_range,
        "R2": r2
    }

    # Log model performance
    logger.info(f"Model performance: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

    # Extract model parameters
    params = pipeline.named_steps["model"].get_params()
    logger.info("Pipeline training completed successfully.")

    return pipeline, metrics, X_train, y_train, X_test, y_test, params