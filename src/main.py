import os
import sys
import logging
from mlflow import MlflowClient
from mlflow.models import infer_signature

from extract_data import extract_data
from load_data import load_data_from_feature_store
from extract_data import upload_training_data_to_bigquery
from train import create_pipeline, train_pipeline
from helper import log_to_mlflow, setup_mlflow
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

logger = logging.getLogger(__name__)  # Global logger


#from google.auth import default

#credentials, project = default()
#logger.info(f"Authenticated with project: {project}")



def main():
    try:
        logger.info("Starting main pipeline execution...")

        # Load and preprocess data
        TRAINING_SIZE = int(os.getenv("TRAINING_SIZE", 100000))
        try:
            data = load_data_from_feature_store(TRAINING_SIZE)
            logger.info("Data loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading data from feature store: {e}")
            sys.exit(1)

        try:
            data = extract_data(data)
            logger.info("Data extraction completed.")
        except Exception as e:
            logger.error(f"Error extracting data: {e}")
            sys.exit(1)

        TEST_RUN = os.getenv("TEST_RUN", "False").lower() == "true"
        if not TEST_RUN:
            try:
                upload_training_data_to_bigquery(data)
                logger.info("Training data uploaded to BigQuery.")
            except Exception as e:
                logger.error(f"Error uploading training data to BigQuery: {e}")
                sys.exit(1)

        try:
            pipeline = create_pipeline()
            logger.info("Pipeline created successfully.")
        except Exception as e:
            logger.error(f"Error creating pipeline: {e}")
            sys.exit(1)

        try:
            experiment = setup_mlflow()
            logger.info(f"MLflow experiment with id {experiment.experiment_id} used.")
        except Exception as e:
            logger.error(f"Error setting up MLflow experiment: {e}")
            sys.exit(1)

        # Train pipeline
        try:
            pipeline, metrics, X_train, y_train, X_test, y_test, params = train_pipeline(pipeline, data)
            logger.info("Pipeline training completed.")
        except Exception as e:
            logger.error(f"Error training pipeline: {e}")
            sys.exit(1)

        try:
            example_input = X_test.iloc[:5]
            signature = infer_signature(example_input, pipeline.predict(example_input))
        except Exception as e:
            logger.error(f"Error inferring model signature: {e}")
            sys.exit(1)

        try:
            model_uri, run_id, model_version = log_to_mlflow(pipeline, X_test, y_test, signature, experiment, metrics, params)
            logger.info(f"Model logged to MLflow: {model_uri}")
        except Exception as e:
            logger.error(f"Error logging to MLflow: {e}")
            sys.exit(1)


        try:
            validated_serving = validate_serving(example_input, model_uri)
            logger.info(f"Validated serving: {validated_serving}")
        except Exception as e:
            logger.warning(f"Model serving validation failed: {e}")  # Non-critical

        try:
            validate_model(model_uri, X_test, y_test, run_id, experiment, model_version)
            logger.info("Model validation completed.")

            client = MlflowClient()
            # Set registered model tag
            client.set_registered_model_tag("xgb_pipeline_taxi_regressor", "task", "regressor")
            # Set model version tag
            client.set_model_version_tag("xgb_pipeline_taxi_regressor", model_version, "validation_status", "approved")
            client.set_registered_model_alias("xgb_pipeline_taxi_regressor", "champion", model_version)

        except Exception as e:
            logger.warning(f"Model validation failed: {e}")  # Non-critical

        logger.info("Pipeline execution completed successfully.")

    except Exception as e:
        logger.critical("Unexpected error occurred!", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()