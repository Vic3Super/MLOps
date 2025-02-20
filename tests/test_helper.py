import json
import os
import uuid
from unittest.mock import MagicMock, patch, mock_open

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowException
from sklearn.pipeline import Pipeline

from src.helper import setup_mlflow, log_to_mlflow, log_config


def test_mlflow_connection(experiment_name="Test_Experiment"):
    """
    Tests the connection to the MLflow tracking server by:
    1. Checking if MLflow has a tracking URI.
    2. Logging a test parameter and retrieving it (ensuring full tracking functionality).

    Args:
        experiment_name (str): Name of the experiment to use for testing.

    Returns:
        bool: True if MLflow connection is fully functional, False otherwise.
    """
    try:
        setup_mlflow()
        # Step 1: Check if MLflow has a tracking URI (basic check)
        tracking_uri = mlflow.get_tracking_uri()
        print(f"MLflow Tracking URI: {tracking_uri}")

        # Step 2: Ensure logging works
        with mlflow.start_run(run_name="mlflow_test_connection") as run:
            run_id = run.info.run_id
            mlflow.log_param("test_param", 42)

            # Retrieve the run to verify the parameter was logged
            client = mlflow.MlflowClient()
            params = client.get_run(run_id).data.params

            if params.get("test_param") == "42":
                print(f"MLflow connection verified! Run ID: {run_id}")
                return True

        print(f"MLflow connection failed: Could not retrieve logged parameter.")
        return False

    except MlflowException as e:
        print(f"MLflow connection error: {e}")
        return False


@pytest.fixture
def mock_pipeline():
    """Creates a mock sklearn pipeline with a predict method."""
    pipeline = MagicMock(spec=Pipeline)
    pipeline.predict = MagicMock(return_value=np.array([1.0, 2.0, 3.0]))
    return pipeline

@pytest.fixture
def mock_experiment():
    """Creates a mock MLflow experiment object."""
    experiment = MagicMock()
    experiment.experiment_id = "1234"
    return experiment

@pytest.fixture
def sample_data():
    """Returns sample test data."""
    X_test = pd.DataFrame(np.random.rand(10, 3), columns=["feat1", "feat2", "feat3"])
    Y_test = pd.Series(np.random.rand(10))
    return X_test, Y_test

@pytest.fixture
def valid_inputs(mock_pipeline, mock_experiment, sample_data):
    """Returns valid input data for testing."""
    X_test, Y_test = sample_data
    return {
        "pipeline": mock_pipeline,
        "X_test": X_test,
        "Y_test": Y_test,
        "signature": None,  # Can be mocked if needed
        "experiment": mock_experiment,
        "metrics": {"rmse": 0.5},
        "params": {"n_estimators": 100}
    }

# -------------------------------
# Input Validation Tests
# -------------------------------
def test_invalid_pipeline(sample_data, mock_experiment):
    X_test, Y_test = sample_data
    with pytest.raises(TypeError):
        log_to_mlflow("not_a_pipeline", X_test, Y_test, None, mock_experiment, {}, {})


def test_invalid_X_test(sample_data, mock_pipeline, mock_experiment):
    X_test, Y_test = sample_data
    with pytest.raises(TypeError):
        log_to_mlflow(mock_pipeline, "invalid_X_test", Y_test, None, mock_experiment, {}, {})

def test_invalid_Y_test(sample_data, mock_pipeline, mock_experiment):
    X_test, _ = sample_data
    with pytest.raises(TypeError):
        log_to_mlflow(mock_pipeline, X_test, "invalid_Y_test", None, mock_experiment, {}, {})

def test_X_Y_length_mismatch(mock_pipeline, mock_experiment):
    X_test = pd.DataFrame(np.random.rand(10, 3))
    Y_test = pd.Series(np.random.rand(5))  # Mismatched length
    with pytest.raises(ValueError, match="X_test and Y_test must have the same number of samples"):
        log_to_mlflow(mock_pipeline, X_test, Y_test, None, mock_experiment, {}, {})

def test_invalid_experiment():
    invalid_experiment = MagicMock()
    del invalid_experiment.experiment_id  # Remove experiment_id
    with pytest.raises(ValueError, match="Invalid MLflow experiment object"):
        log_to_mlflow(MagicMock(spec=Pipeline), pd.DataFrame(), pd.Series(), None, invalid_experiment, {}, {})

# -------------------------------
# Successful Execution Tests
# -------------------------------
@patch("mlflow.start_run")
@patch("mlflow.sklearn.log_model")
@patch("mlflow.log_metrics")
@patch("mlflow.log_params")
@patch("mlflow.log_artifact")
def test_successful_logging(mock_log_artifact, mock_log_params, mock_log_metrics, mock_log_model, mock_start_run, valid_inputs):
    """Test the function runs successfully and logs correct components."""
    mock_start_run.return_value.__enter__.return_value.info.run_id = "run_123"

    logged_model_uri, run_id = log_to_mlflow(**valid_inputs)

    assert logged_model_uri is not None
    assert run_id == "run_123"

    mock_start_run.assert_called_once()
    mock_log_model.assert_called_once()
    mock_log_metrics.assert_called_once()
    mock_log_params.assert_called()
    mock_log_artifact.assert_called()

# -------------------------------
# Error Handling Tests
# -------------------------------
@patch("mlflow.sklearn.log_model", side_effect=MlflowException("Logging failed"))
@patch("mlflow.start_run")
def test_model_logging_failure(mock_start_run, mock_log_model, valid_inputs):
    """Test failure when logging model to MLflow."""
    # Mock the MLflow run object to prevent actual connection
    mock_start_run.return_value.__enter__.return_value.info.run_id = "run_123"

    # Expect RuntimeError due to model logging failure
    with pytest.raises(RuntimeError, match="Failed to log model to MLflow"):
        log_to_mlflow(**valid_inputs)

    # Ensure the start_run was called
    mock_start_run.assert_called_once()
    mock_log_model.assert_called_once()



# Sample experiment_id and run_id
experiment_id = "test_experiment"
run_id = "test_run"

# Define the expected file path (should match your function logic)
json_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs/config.json"))
@pytest.mark.parametrize("exp_id, run_id", [
    ("", "valid"),  # Missing experiment_id
    ("valid", ""),  # Missing run_id
    ("", ""),       # Both missing
])
def test_log_config_missing_arguments(exp_id, run_id):
    """Test if the function raises ValueError when required arguments are missing."""
    with pytest.raises(ValueError, match="Both 'experiment_id' and 'run_id' must be provided and non-empty."):
        log_config(exp_id, run_id)

@patch("os.path.exists", side_effect=lambda path: False if path == os.path.dirname(json_file_path) else True)
def test_log_config_directory_missing(mock_exists):
    """Test if the function raises FileNotFoundError when the config directory does not exist."""
    with pytest.raises(FileNotFoundError, match="Config directory does not exist"):
        log_config(experiment_id, run_id)

