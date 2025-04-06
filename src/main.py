import os
import sys
import logging
from datetime import datetime
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
import mlflow.data
from sklearn.utils import estimator_html_repr
import pandas as pd
from mlflow.data.pandas_dataset import PandasDataset
from extract_data import extract_data
from load_data import load_data_from_feature_store
from extract_data import upload_training_data_to_bigquery
from train import create_pipeline, train_pipeline
from helper import setup_mlflow, create_plots
from validate import validate_serving, validate_model

# Setup Logging
LOG_FILE = "logs/app.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,  # Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Timestamp and level
    handlers=[
        logging.FileHandler(LOG_FILE),  # Save logs to file
        logging.StreamHandler(sys.stdout)  # Print logs to console
    ]
)

PROJECT_ID = os.getenv("PROJECT_ID")
DATASET_NAME = os.getenv("DATASET_NAME")

logger = logging.getLogger(__name__)  # Global logger

# Load and preprocess data

def main():
    DATA_SIZE = int(os.getenv("TRAINING_SIZE", 100000))
    TEST_RUN = bool(os.getenv("TEST_RUN", "True"))

    experiment = setup_mlflow()
    run_name = datetime.now().strftime("%Y-%m-%d_%H:%M")
    tags = {
        "model_type": "XGB Regressor",
        "experiment_description": "Taxi Regressor"
    }
    with mlflow.start_run(run_name=run_name, experiment_id=experiment.experiment_id, tags=tags) as main_run:
        config = {
            "DATA_SIZE": DATA_SIZE,
            "TEST_RUN": TEST_RUN,
            "PROJECT_ID": PROJECT_ID,
            "DATASET_NAME": DATASET_NAME,
        }
        mlflow.log_params(config)

        with mlflow.start_run(nested=True, experiment_id=experiment.experiment_id, parent_run_id=main_run.info.run_id, tags=tags, run_name="data_load") as data_load_run:
            data = load_data_from_feature_store(size=DATA_SIZE)
            dataset = mlflow.data.pandas_dataset.from_pandas(data)
            logger.info(dataset)
            mlflow.log_input(dataset, "loaded_data")

        with mlflow.start_run(nested=True, experiment_id=experiment.experiment_id, parent_run_id=main_run.info.run_id, tags=tags, run_name="data_extract") as data_extract_run:
            data = extract_data(data)
            dataset = mlflow.data.pandas_dataset.from_pandas(data)
            logger.info(dataset)
            mlflow.log_input(dataset, "extracted_data")


        with mlflow.start_run(nested=True, experiment_id=experiment.experiment_id, parent_run_id=main_run.info.run_id, tags=tags, run_name="model_training") as model_training_run:
            mlflow.log_input(dataset, "training")
            pipeline = create_pipeline()
            pipeline, metrics, X_train, y_train, X_test, y_test, params = train_pipeline(pipeline, data)
            mlflow.log_metrics(metrics)
            mlflow.log_params(params)
            example_input = X_test.iloc[:5]
            signature = infer_signature(example_input, pipeline.predict(example_input))

            logged_model_uri = mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                registered_model_name="xgb_pipeline_taxi_regressor",
                signature=signature,
            ).model_uri

            client = MlflowClient()
            model_version = client.get_latest_versions("xgb_pipeline_taxi_regressor")[0].version
            estimator_html = estimator_html_repr(pipeline)
            html_path = "estimator.html"

            with open(html_path, "w", encoding="utf-8") as f:
                f.write(estimator_html)

            mlflow.log_artifact(html_path, artifact_path="artifacts")

            mlflow.log_param("TRAINING_SIZE", len(X_train))

            figs = create_plots(pipeline, X_test, y_test)

            for i, fig in enumerate(figs):
                mlflow.log_figure(fig, f'plots/{i}.png')
                logger.info(f"Plot {i} successfully logged to MLflow.")

        with mlflow.start_run(nested=True, experiment_id=experiment.experiment_id, parent_run_id=main_run.info.run_id, tags=tags, run_name="model_validation") as model_validation_run:
            validated_serving = validate_serving(example_input, logged_model_uri)
            validate_model(logged_model_uri, X_test, y_test, model_validation_run.info.run_id, experiment, model_version)
            client = MlflowClient()
            # Set registered model tag
            client.set_registered_model_tag("xgb_pipeline_taxi_regressor", "task", "regressor")
            # Set model version tag
            client.set_model_version_tag("xgb_pipeline_taxi_regressor", model_version, "validation_status", "approved")
            if not TEST_RUN:
                client.set_registered_model_alias("xgb_pipeline_taxi_regressor", "challenger", model_version)

        if not TEST_RUN:
            upload_training_data_to_bigquery(data, model_training_run.info.run_id)

        mlflow.log_artifact(LOG_FILE, artifact_path="logs")


if __name__ == "__main__":
    main()