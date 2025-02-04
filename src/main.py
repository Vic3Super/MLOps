# Main function
import pandas as pd
from mlflow.models import infer_signature

from deploy import setup_mlflow, log_to_mlflow
from extract_data import extract_data
from load_data import load_data_from_feature_store
from train import train_pipeline, create_pipeline


def main():
    # Load and preprocess data
    data = load_data_from_feature_store(size=1000)
    data = extract_data(data)

    # Create pipeline
    pipeline = create_pipeline()

    print("Pipeline created")

    # Train pipeline
    pipeline, metrics, X_train, y_train, X_test, y_test = train_pipeline(pipeline, data)
    example_input = X_train.iloc[:5]  # Select a small sample (e.g., first 5 rows)


    signature = infer_signature(example_input, pipeline.predict(example_input))

    # Set up MLflow
    experiment = setup_mlflow()
    print("Experiment created.")
    log_to_mlflow(pipeline, X_train, X_test, y_test, signature, experiment, metrics)
    print("Logged to MLFlow.")


if __name__ == "__main__":
    main()