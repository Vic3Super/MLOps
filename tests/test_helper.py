
from unittest.mock import MagicMock, patch, mock_open

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowException
from sklearn.pipeline import Pipeline

from src.helper import setup_mlflow, log_to_mlflow


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






