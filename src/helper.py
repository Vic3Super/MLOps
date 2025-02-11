# Define MLflow setup
import json
import os
from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.utils import estimator_html_repr


def setup_mlflow():

    os.environ["GIT_PYTHON_REFRESH"] = "quiet"

    TRACKING_URI = "https://mlflow-service-974726646619.us-central1.run.app"
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.autolog(disable=True)
    experiment_name = "Chicago Taxi Regressor"
    if not mlflow.get_experiment_by_name(name=experiment_name):
        mlflow.create_experiment(name=experiment_name)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    return experiment

def feature_importance(pipeline):
    # Example: Assuming `pipeline` is your trained pipeline
    model = pipeline.named_steps['model']  # Replace 'model' with the actual step name in your pipeline

    # Extract feature importances
    if hasattr(model, "feature_importances_"):  # Tree-based models (RandomForest, XGBoost, etc.)
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):  # Linear models
        importances = np.abs(model.coef_)
    else:
        raise ValueError("The model does not have feature importances.")

    # Get feature names from preprocessing step (if any)
    feature_names = None
    if 'preprocessing' in pipeline.named_steps:  # Replace 'preprocessor' with your actual step name
        preprocessor = pipeline.named_steps['preprocessing']
        if hasattr(preprocessor, "get_feature_names_out"):
            feature_names = preprocessor.get_feature_names_out()

    # Create DataFrame for visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names if feature_names is not None else range(len(importances)),
        'Importance': importances
    }).sort_values(by="Importance", ascending=False)

    return importance_df


def create_plots(pipeline, X_test, y_test):
    # -------------------------------
    # Figure 1: Predicted vs Actual Values
    # -------------------------------
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    y_pred = pipeline.predict(X_test)
    ax1.scatter(y_test, y_pred, alpha=0.7)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # Perfect prediction line
    ax1.set_title("Predicted vs Actual Values")
    ax1.set_xlabel("Actual Values")
    ax1.set_ylabel("Predicted Values")

    # -------------------------------
    # Figure 2: Feature Importances
    # -------------------------------
    importance_df = feature_importance(pipeline)

    fig2, ax2 = plt.subplots(figsize=(12, 8))  # Increase figure size for more space
    ax2.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
    ax2.set_title("Feature Importances", fontsize=14)
    ax2.set_xlabel("Importance Score", fontsize=12)
    ax2.set_ylabel("Features", fontsize=12)
    ax2.invert_yaxis()  # Highest importance on top
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    # Rotate x-axis labels if needed (in case of overlapping text)
    ax2.tick_params(axis="x", labelrotation=45)

    figs = [fig1, fig2]

    return figs



def log_config(experiment_id, run_id):
    # Define the path to the JSON file (same directory as the script)
    json_file_path = os.path.join(os.path.dirname(__file__), "../configs/config.json")
    def update_json_file(data):
        """
        Update the JSON file with new data.

        Args:
            data (dict): The data to write into the JSON file.
        """
        try:
            # Read existing content
            if os.path.exists(json_file_path):
                with open(json_file_path, "r") as file:
                    existing_data = json.load(file)
            else:
                existing_data = {}

            # Merge new data with existing content
            existing_data.update(data)

            # Write updated content back to the file
            with open(json_file_path, "w") as file:
                json.dump(existing_data, file, indent=4)

            print(f"Updated {json_file_path} successfully.")
        except Exception as e:
            print(f"An error occurred: {e}")
    new_config = {
        "GCS_REQUIREMENTS_PATH":f"gs://mlflow-bucket-1998/mlruns/{experiment_id}/{run_id}/artifacts/model/requirements.txt",
        "MODEL_PATH": f"gs://mlflow-bucket-1998/mlruns/{experiment_id}/{run_id}/artifacts/model",
        "RUN_ID": f"gs://mlflow-bucket-1998/mlruns/{experiment_id}/{run_id}",
    }
    update_json_file(new_config)


def log_to_mlflow(pipeline, X_test, Y_test, signature, experiment, metrics, params):

    run_name = datetime.now().strftime("%Y-%m-%d_%H:%M")
    tags = {
        "env": "Production",
        "model_type": "XGB Regressor",
        "experiment_description": "Taxi Regressor"
    }

    with mlflow.start_run(
            run_name=run_name, experiment_id=experiment.experiment_id, tags=tags
    ) as run:
        logged_model_uri = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name="xgb_pipeline_taxi_regressor",
            signature=signature,
        ).model_uri

        # Generate estimator HTML
        estimator_html = estimator_html_repr(pipeline)

        # Save the HTML to a file
        html_path = "estimator.html"
        with open("estimator.html", "w", encoding="utf-8") as f:
            f.write(estimator_html)
        mlflow.log_artifact("estimator.html", artifact_path="artifacts")

        figs = create_plots(pipeline, X_test, Y_test)

        for i, fig in enumerate(figs):  # Use enumerate to get index and figure
            mlflow.log_figure(fig, f'plots/{i}.png')

        mlflow.log_metrics(metrics)
        mlflow.log_params(params)

        run_id = run.info.run_id
        log_config(experiment_id=experiment.experiment_id, run_id=run_id)

    print("Pipeline logged successfully!")

    return logged_model_uri, run_id