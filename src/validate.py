import logging
import numpy as np
import pandas as pd
import mlflow
from mlflow.models import validate_serving_input, make_metric
from mlflow.models import convert_input_example_to_serving_input
from mlflow.models import MetricThreshold
from mlflow.models.evaluation.validation import ModelValidationFailedException

# Get the global logger (inherits from main.py)
logger = logging.getLogger(__name__)

def validate_serving(input_example: pd.DataFrame, model_uri: str):
    """
    Validates that the model at the given URI can successfully generate predictions
    for the provided input example.

    Args:
        input_example (pd.DataFrame): The input data example for validation.
        model_uri (str): The URI of the model to validate.

    Returns:
        dict: The result of the serving validation.

    Raises:
        ValueError: If `model_uri` is empty or None, or `input_example` is empty.
        TypeError: If `input_example` is not a pandas DataFrame.
    """
    logger.info(f"Validating serving for model URI: {model_uri}")

    if not model_uri:
        logger.error("Model URI is missing or empty.")
        raise ValueError("Model URI cannot be empty or None.")

    if not isinstance(input_example, pd.DataFrame):
        logger.error("Input example is not a pandas DataFrame.")
        raise TypeError("Input example must be a pandas DataFrame.")

    if input_example.empty:
        logger.error("Input example is empty.")
        raise ValueError("Input example cannot be empty.")

    # Convert input example to the serving format
    try:
        serving_payload = convert_input_example_to_serving_input(input_example)
        logger.info("Successfully converted input example for serving validation.")

        # Validate model prediction
        output = validate_serving_input(model_uri, serving_payload)
        logger.info("Model serving validation completed successfully.")
        return output
    except Exception as e:
        logger.error(f"Error during model serving validation: {e}", exc_info=True)
        raise RuntimeError(f"Model serving validation failed: {e}")


def validate_model(candidate_model_uri: str, X_test, y_test, parent_run_id: str, experiment):
    """
    Validates the candidate model against evaluation criteria and the last running model.

    Args:
        candidate_model_uri (str): URI of the candidate model.
        X_test (pd.DataFrame or np.ndarray): Test feature set.
        y_test (pd.Series, np.ndarray, or list): Test target values.
        parent_run_id (str): Parent MLflow run ID.
        experiment (mlflow.Experiment): MLflow experiment object.

    Raises:
        TypeError: If `X_test` or `y_test` is not in the expected format.
        ValueError: If `X_test` and `y_test` have mismatched lengths.
        ModelValidationFailedException: If the model does not meet validation thresholds.
    """
    logger.info(f"Starting model validation for candidate model: {candidate_model_uri}")

    # Validate input types
    if not isinstance(X_test, (pd.DataFrame, np.ndarray)):
        logger.error("X_test must be a pandas DataFrame or NumPy array.")
        raise TypeError("X_test must be a pandas DataFrame or NumPy array.")
    if not isinstance(y_test, (pd.Series, np.ndarray, list)):
        logger.error("y_test must be a pandas Series, NumPy array, or list.")
        raise TypeError("y_test must be a pandas Series, NumPy array, or list.")
    if len(X_test) != len(y_test):
        logger.error("Mismatch: X_test and y_test have different lengths.")
        raise ValueError("X_test and y_test must have the same number of samples.")

    # Convert X_test into a DataFrame if it's a NumPy array
    if isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test)

    eval_data = X_test.copy()
    eval_data["target"] = y_test

    logger.info("Beginning MLflow evaluation process.")

    # Start a new validation run under the experiment
    with mlflow.start_run(run_name="validation_run", parent_run_id=parent_run_id,
                          experiment_id=experiment.experiment_id) as run:

        def root_mean_squared_error_by_mean(eval_df, _builtin_metrics):
            """Computes RMSE standardized by mean of target values."""
            try:
                mse = np.mean((eval_df["prediction"] - eval_df["target"]) ** 2)
                rmse = np.sqrt(mse)
                target_mean = np.mean(eval_df["target"])
                return rmse / target_mean if target_mean != 0 else np.nan  # Avoid division by zero
            except Exception as e:
                logger.error(f"Error in root_mean_squared_error_by_mean: {e}")
                return np.nan

        def root_mean_squared_error_by_range(eval_df, _builtin_metrics):
            """Computes RMSE standardized by range of target values."""
            try:
                mse = np.mean((eval_df["prediction"] - eval_df["target"]) ** 2)
                rmse = np.sqrt(mse)
                target_range = np.max(eval_df["target"]) - np.min(eval_df["target"])
                return rmse / target_range if target_range != 0 else np.nan  # Avoid division by zero
            except Exception as e:
                logger.error(f"Error in root_mean_squared_error_by_range: {e}")
                return np.nan

        candidate_result = mlflow.evaluate(
            model=candidate_model_uri,
            data=eval_data,
            targets="target",
            model_type="regressor",
            evaluators=["default"],
            extra_metrics=[
                make_metric(
                    eval_fn=root_mean_squared_error_by_mean,
                    greater_is_better=False,
                ),
                make_metric(
                    eval_fn=root_mean_squared_error_by_range,
                    greater_is_better=True,
                ),
            ])

    logger.info("Evaluation completed. Checking validation thresholds.")

    # Define validation criteria
    thresholds = {
        "mean_absolute_error": MetricThreshold(threshold=5, greater_is_better=False),
        "root_mean_squared_error": MetricThreshold(threshold=8, greater_is_better=False),
        "r2_score": MetricThreshold(threshold=0.8, greater_is_better=True),
        "root_mean_squared_error_by_mean": MetricThreshold(threshold=0.3, greater_is_better=False),
        "root_mean_squared_error_by_range": MetricThreshold(threshold=0.1, greater_is_better=False),
    }

    # Validate the candidate model against baseline
    mlflow.validate_evaluation_results(
        candidate_result=candidate_result,
        validation_thresholds=thresholds,
    )

    logger.info("Model validation successful. The model meets the required thresholds.")
