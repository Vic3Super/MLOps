# Main function
import pandas as pd
from mlflow.models import infer_signature
from plotly.data import experiment

from extract_data import extract_data
from load_data import load_data_from_feature_store
from train import  create_pipeline, train_pipeline
from helper import log_to_mlflow, setup_mlflow
from validate import validate_serving, validate_model


def main():
    # Load and preprocess data
    data = load_data_from_feature_store(size=100000)
    print("Data loaded.")
    data = extract_data(data)
    print("Data extracted.")
    # Create pipeline
    pipeline = create_pipeline()
    print("Pipeline created")

    experiment = setup_mlflow()
    print("Experiment created.")

    # Train pipeline
    pipeline, metrics, best_params, X_train, y_train, X_test, y_test, run_id = train_pipeline(pipeline, data, experiment)
    print("Pipeline trained")

    example_input = X_test.iloc[:5]  # Select a small sample (e.g., first 5 rows)


    signature = infer_signature(example_input, pipeline.predict(example_input))


    model_uri = log_to_mlflow(pipeline, X_test, y_test, signature, experiment, metrics, run_id)
    print("Logged to MLFlow.")


    validated_serving = validate_serving(example_input, model_uri)
    print(f"Validated serving with {validated_serving}")
    try:
        validate_model(model_uri, X_test, y_test, run_id, experiment)
    except Exception as e:
        print(e)

    print("Model validated")

if __name__ == "__main__":
    main()