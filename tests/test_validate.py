import pytest
import pandas as pd
import numpy as np
import mlflow
from unittest.mock import patch, MagicMock
from src.validate import validate_serving, validate_model  # Replace `my_module` with the actual module name


def test_validate_serving_empty_dataframe():
    input_example = pd.DataFrame()
    model_uri = "models:/test-model/1"

    with pytest.raises(ValueError, match="Input example cannot be empty."):
        validate_serving(input_example, model_uri)

def test_validate_serving_invalid_input_type():
    input_example = [[1, 2, 3], [4, 5, 6]]  # Not a DataFrame
    model_uri = "models:/test-model/1"

    with pytest.raises(TypeError, match="Input example must be a pandas DataFrame"):
        validate_serving(input_example, model_uri)

def test_validate_serving_invalid_model_uri():
    input_example = pd.DataFrame({"feature1": [1, 2, 3]})
    model_uri = None

    with patch("mlflow.models.convert_input_example_to_serving_input", side_effect=mlflow.exceptions.MlflowException("Invalid URI")):
        with pytest.raises(ValueError, match="Model URI cannot be empty or None."):
            validate_serving(input_example, model_uri)

def test_validate_model_invalid_X_test_type():
    X_test = [[1, 2, 3], [4, 5, 6]]  # Not a DataFrame or NumPy array
    y_test = pd.Series([0.1, 0.2, 0.3])
    candidate_model_uri = "models:/test-model/1"
    parent_run_id = "1234"
    experiment = MagicMock()

    with pytest.raises(TypeError, match="X_test must be a pandas DataFrame or NumPy array."):
        validate_model(candidate_model_uri, X_test, y_test, parent_run_id, experiment)


def test_validate_model_invalid_y_test_type():
    X_test = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    y_test = "not a valid type"  # Invalid
    candidate_model_uri = "models:/test-model/1"
    parent_run_id = "1234"
    experiment = MagicMock()

    with pytest.raises(TypeError, match="y_test must be a pandas Series, NumPy array, or list."):
        validate_model(candidate_model_uri, X_test, y_test, parent_run_id, experiment)


def test_validate_model_mismatched_data_lengths():
    X_test = pd.DataFrame({"feature1": [1, 2], "feature2": [4, 5]})
    y_test = pd.Series([0.1, 0.2, 0.3])  # More elements than X_test
    candidate_model_uri = "models:/test-model/1"
    parent_run_id = "1234"
    experiment = MagicMock()

    with pytest.raises(ValueError, match="X_test and y_test must have the same number of samples."):
        validate_model(candidate_model_uri, X_test, y_test, parent_run_id, experiment)
