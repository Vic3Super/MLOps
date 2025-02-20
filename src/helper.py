import json
import os
import logging
from datetime import datetime
import mlflow
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mlflow import MlflowException
from sklearn.pipeline import Pipeline
from sklearn.utils import estimator_html_repr

# Get the global logger (inherits from main.py)
logger = logging.getLogger(__name__)


def setup_mlflow() -> mlflow.entities.Experiment:
    """
    Configures MLflow tracking settings and retrieves the experiment object.

    Returns:
        mlflow.entities.Experiment: The MLflow experiment object.

    Raises:
        MlflowException: If there is an issue setting up the MLflow experiment.
    """
    logger.info("Setting up MLflow.")

    os.environ["GIT_PYTHON_REFRESH"] = "quiet"
    TRACKING_URI = "https://mlflow-service-974726646619.us-central1.run.app"
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.autolog(disable=True)

    experiment_name = "Chicago Taxi Regressor"

    try:
        if not mlflow.get_experiment_by_name(experiment_name):
            mlflow.create_experiment(name=experiment_name)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        logger.info(f"MLflow experiment '{experiment_name}' set up successfully.")
        return experiment
    except MlflowException as e:
        logger.critical(f"Failed to set up MLflow: {e}", exc_info=True)
        raise


def feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """
    Extracts feature importances from a trained pipeline.

    Args:
        pipeline (Pipeline): The trained pipeline containing a model.

    Returns:
        pd.DataFrame: A DataFrame containing feature importance scores.

    Raises:
        ValueError: If the pipeline is None.
        TypeError: If the pipeline is not an instance of sklearn Pipeline.
        KeyError: If the pipeline does not contain a 'model' step.
    """
    logger.info("Extracting feature importance from pipeline.")

    if pipeline is None:
        logger.error("Pipeline is None.")
        raise ValueError("Pipeline cannot be None.")

    if not isinstance(pipeline, Pipeline):
        logger.error("Invalid pipeline type.")
        raise TypeError("Expected a sklearn.pipeline.Pipeline object.")

    if 'model' not in pipeline.named_steps:
        logger.error("Pipeline does not contain a 'model' step.")
        raise KeyError("Pipeline does not contain a 'model' step. Please check your pipeline structure.")

    model = pipeline.named_steps['model']

    # Extract feature importances
    try:
        if hasattr(model, "feature_importances_"):  # Tree-based models
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):  # Linear models
            importances = np.abs(model.coef_)
        else:
            logger.warning("Model does not have feature importances.")
            raise ValueError("The model does not have feature importances.")
    except Exception as e:
        logger.error(f"Failed to retrieve feature importances: {e}", exc_info=True)
        raise

    # Get feature names from preprocessing step
    feature_names = None
    if 'preprocessing' in pipeline.named_steps:
        preprocessor = pipeline.named_steps['preprocessing']
        if hasattr(preprocessor, "get_feature_names_out"):
            try:
                feature_names = preprocessor.get_feature_names_out()
            except Exception as e:
                logger.warning(f"Failed to extract feature names from preprocessing: {e}")

    # Ensure feature_names and importances match
    num_features = len(importances)
    if feature_names is not None and len(feature_names) != num_features:
        logger.warning("Feature names and importances do not match in length. Using indices instead.")
        feature_names = range(num_features)

    # Create DataFrame for visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names if feature_names is not None else range(num_features),
        'Importance': importances
    }).sort_values(by="Importance", ascending=False)

    logger.info("Feature importance extraction completed.")
    return importance_df


def create_plots(pipeline: Pipeline, X_test, y_test) -> list:
    """
    Generates diagnostic plots for model evaluation.

    Args:
        pipeline (Pipeline): Trained machine learning pipeline.
        X_test (pd.DataFrame or np.ndarray): Test feature set.
        y_test (pd.Series or np.ndarray): Test target values.

    Returns:
        list: A list of Matplotlib figures.

    Raises:
        TypeError: If input types are incorrect.
        ValueError: If X_test and y_test have mismatched lengths.
        RuntimeError: If no plots are generated.
    """
    logger.info("Generating diagnostic plots.")

    # Validate inputs
    if not isinstance(pipeline, Pipeline):
        raise TypeError("Expected 'pipeline' to be a sklearn.pipeline.Pipeline object.")
    if not hasattr(pipeline, "predict"):
        raise ValueError("Pipeline does not have a predict method. Ensure it's a trained model.")
    if not isinstance(X_test, (pd.DataFrame, np.ndarray)):
        raise TypeError("X_test must be a pandas DataFrame or NumPy array.")
    if not isinstance(y_test, (pd.Series, np.ndarray, list)):
        raise TypeError("y_test must be a pandas Series, NumPy array, or list.")
    if len(X_test) != len(y_test):
        raise ValueError("X_test and y_test must have the same number of samples.")

    try:
        y_pred = pipeline.predict(X_test)

        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.scatter(y_test, y_pred, alpha=0.7)
        ax1.plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], 'r--', lw=2)
        ax1.set_title("Predicted vs Actual Values")
        ax1.set_xlabel("Actual Values")
        ax1.set_ylabel("Predicted Values")

        logger.info("Generated 'Predicted vs Actual' plot.")

    except Exception as e:
        logger.error(f"Failed to generate 'Predicted vs Actual' plot: {e}", exc_info=True)
        fig1 = None

    try:
        importance_df = feature_importance(pipeline)
        if importance_df.empty:
            raise ValueError("Feature importance data is empty.")

        fig2, ax2 = plt.subplots(figsize=(12, 8))
        ax2.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
        ax2.set_title("Feature Importances", fontsize=14)
        ax2.set_xlabel("Importance Score", fontsize=12)
        ax2.set_ylabel("Features", fontsize=12)
        ax2.invert_yaxis()
        plt.tight_layout()

        logger.info("Generated 'Feature Importances' plot.")

    except ValueError as ve:
        logger.warning(f"{ve}")
        fig2 = None
    except Exception as e:
        logger.error(f"Failed to generate 'Feature Importances' plot: {e}", exc_info=True)
        fig2 = None

    figs = [fig for fig in [fig1, fig2] if fig is not None]

    if not figs:
        logger.critical("No plots were generated due to errors.")
        raise RuntimeError("No plots were generated due to errors.")

    logger.info("Plot generation completed successfully.")
    return figs


def log_config(experiment_id: str, run_id: str):
    """
    Logs the configuration details of a model run into a JSON file.

    Args:
        experiment_id (str): The experiment ID in MLflow.
        run_id (str): The run ID in MLflow.

    Raises:
        ValueError: If `experiment_id` or `run_id` is empty.
        FileNotFoundError: If the configuration directory does not exist.
        RuntimeError: If unable to update the JSON configuration due to file write errors.
    """
    logger.info("Starting configuration logging.")

    # Validate input arguments
    if not experiment_id or not run_id:
        logger.error("Missing experiment_id or run_id.")
        raise ValueError("Both 'experiment_id' and 'run_id' must be provided and non-empty.")

    # Define the path to the JSON file (in the configs directory)
    json_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs/config.json"))

    # Ensure the config directory exists
    config_dir = os.path.dirname(json_file_path)
    if not os.path.exists(config_dir):
        logger.error(f"Configuration directory does not exist: {config_dir}")
        raise FileNotFoundError(f"Config directory does not exist: {config_dir}")

    # Construct new configuration data
    new_config = {
        "RUN_ID": f"gs://mlflow-bucket-1998/mlruns/{experiment_id}/{run_id}",
    }

    existing_data = {}

    # Read existing content if the JSON file exists
    if os.path.exists(json_file_path):
        try:
            with open(json_file_path, "r", encoding="utf-8") as file:
                existing_data = json.load(file)
                if not isinstance(existing_data, dict):
                    logger.warning("JSON file does not contain a valid dictionary. Overwriting...")
                    existing_data = {}  # Reset to empty dict
                else:
                    logger.info("Existing configuration file loaded successfully.")
        except json.JSONDecodeError:
            logger.warning("JSON file is corrupted. Overwriting with new configuration...")
            existing_data = {}  # Reset to empty dict
        except Exception as e:
            logger.error(f"Error reading JSON config file: {e}", exc_info=True)
            raise RuntimeError(f"Error reading JSON config file: {e}")

    # Merge new configuration with existing content
    existing_data.update(new_config)

    # Write updated content back to the file
    try:
        with open(json_file_path, "w", encoding="utf-8") as file:
            json.dump(existing_data, file, indent=4)
        logger.info(f"Configuration successfully updated in {json_file_path}.")
    except (PermissionError, IOError) as e:
        logger.critical(f"Error writing to {json_file_path}: {e}", exc_info=True)
        raise RuntimeError(f"Error writing to {json_file_path}: {e}")


def log_to_mlflow(
        pipeline: Pipeline,
        X_test: pd.DataFrame or np.ndarray,
        Y_test: pd.Series or np.ndarray or list,
        signature,
        experiment: mlflow.entities.Experiment,
        metrics: dict,
        params: dict
) -> tuple:
    """
    Logs a trained pipeline, plots, and relevant metadata to MLflow.

    Args:
        pipeline (Pipeline): Trained Scikit-learn pipeline.
        X_test (pd.DataFrame or np.ndarray): Test feature set.
        Y_test (pd.Series, np.ndarray, or list): Test target values.
        signature (mlflow.models.Signature): MLflow model signature.
        experiment (mlflow.entities.Experiment): MLflow experiment object.
        metrics (dict): Dictionary of evaluation metrics.
        params (dict): Dictionary of model parameters.

    Returns:
        tuple: (logged_model_uri, run_id) if successful.

    Raises:
        TypeError: If input data types are incorrect.
        ValueError: If input data has mismatched dimensions or missing attributes.
        RuntimeError: If MLflow operations fail.
    """

    logger.info("Starting MLflow logging process.")

    # -------------------------------
    # Validate Inputs
    # -------------------------------
    if not isinstance(pipeline, Pipeline):
        logger.error("Invalid pipeline type.")
        raise TypeError("Expected 'pipeline' to be a trained sklearn.pipeline.Pipeline object.")

    if not hasattr(pipeline, "predict"):
        logger.error("Pipeline does not have a predict method.")
        raise ValueError("Pipeline does not have a predict method. Ensure it's a trained model.")

    if not isinstance(X_test, (pd.DataFrame, np.ndarray)):
        logger.error("Invalid type for X_test.")
        raise TypeError("X_test must be a pandas DataFrame or NumPy array.")

    if not isinstance(Y_test, (pd.Series, np.ndarray, list)):
        logger.error("Invalid type for Y_test.")
        raise TypeError("Y_test must be a pandas Series, NumPy array, or list.")

    if len(X_test) != len(Y_test):
        logger.error("X_test and Y_test have mismatched sample sizes.")
        raise ValueError("X_test and Y_test must have the same number of samples.")

    if not isinstance(metrics, dict) or not isinstance(params, dict):
        logger.error("Metrics or parameters are not dictionaries.")
        raise TypeError("Both 'metrics' and 'params' must be dictionaries.")

    if not hasattr(experiment, "experiment_id"):
        logger.error("Invalid MLflow experiment object.")
        raise ValueError("Invalid MLflow experiment object: missing experiment_id.")

    run_name = datetime.now().strftime("%Y-%m-%d_%H:%M")
    tags = {
        "env": "Production",
        "model_type": "XGB Regressor",
        "experiment_description": "Taxi Regressor"
    }

    try:
        with mlflow.start_run(run_name=run_name, experiment_id=experiment.experiment_id, tags=tags) as run:
            logger.info(f"MLflow run started: {run.info.run_id}")

            # -------------------------------
            # Log Model to MLflow
            # -------------------------------
            try:
                logged_model_uri = mlflow.sklearn.log_model(
                    sk_model=pipeline,
                    artifact_path="model",
                    registered_model_name="xgb_pipeline_taxi_regressor",
                    signature=signature,
                ).model_uri
                logger.info("Model successfully logged to MLflow.")
            except MlflowException as e:
                logger.critical(f"Failed to log model to MLflow: {e}", exc_info=True)
                raise RuntimeError(f"Failed to log model to MLflow: {e}")

            # -------------------------------
            # Generate and Log Estimator HTML
            # -------------------------------
            try:
                estimator_html = estimator_html_repr(pipeline)
                html_path = "estimator.html"

                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(estimator_html)

                mlflow.log_artifact(html_path, artifact_path="artifacts")
                logger.info("Estimator HTML successfully logged.")
            except Exception as e:
                logger.warning(f"Failed to generate or log estimator HTML. Error: {e}")

            # -------------------------------
            # Create and Log Config
            # -------------------------------
            try:
                TRAINING_SIZE = os.getenv("TRAINING_SIZE")
                TEST_RUN = os.getenv("TEST_RUN")
                PROJECT_ID = os.getenv("PROJECT_ID")
                DATASET_NAME = os.getenv("DATASET_NAME")

                config = {
                    "TRAINING_SIZE": TRAINING_SIZE,
                    "TEST_RUN": TEST_RUN,
                    "PROJECT_ID": PROJECT_ID,
                    "DATASET_NAME": DATASET_NAME,
                }

                mlflow.log_params(config)

            except Exception as e:
                logger.warning(f"Failed to generate or log config. Error: {e}")

            # -------------------------------
            # Generate and Log Plots
            # -------------------------------
            try:
                figs = create_plots(pipeline, X_test, Y_test)

                for i, fig in enumerate(figs):
                    try:
                        mlflow.log_figure(fig, f'plots/{i}.png')
                        logger.info(f"Plot {i} successfully logged to MLflow.")
                    except MlflowException as e:
                        logger.warning(f"Failed to log figure {i} to MLflow. Error: {e}")

            except Exception as e:
                logger.warning(f"Failed to generate plots. Error: {e}")

            # -------------------------------
            # Log Metrics and Parameters
            # -------------------------------
            try:
                mlflow.log_metrics(metrics)
                logger.info("Metrics successfully logged to MLflow.")
            except Exception as e:
                logger.warning(f"Failed to log metrics. Ensure 'metrics' is a valid dictionary. Error: {e}")

            try:
                mlflow.log_params(params)
                logger.info("Parameters successfully logged to MLflow.")
            except Exception as e:
                logger.warning(f"Failed to log parameters. Ensure 'params' is a valid dictionary. Error: {e}")

            # -------------------------------
            # Retrieve Run ID
            # -------------------------------
            run_id = run.info.run_id
            logger.info(f"MLflow logging process completed. Run ID: {run_id}")

    except MlflowException as e:
        logger.critical(f"MLflow error encountered: {e}", exc_info=True)
        raise RuntimeError(f"MLflow error: {e}")

    return logged_model_uri, run_id