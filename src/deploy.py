# Define MLflow setup
import json
import os

from datetime import datetime
import mlflow
import matplotlib.pyplot as plt

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

# Define MLflow setup
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

def create_plots(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, alpha=0.7)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # Line of perfect predictions
    ax.set_title("Predicted vs Actual Values")
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    return fig


def log_to_mlflow(pipeline, X_train, X_test, y_test, signature, experiment, metrics):
    run_name = datetime.now().strftime("%Y-%m-%d_%H:%M")
    tags = {
        "env": "test",
        "model_type": "XGB Regressor",
        "experiment_description": "Taxi Regressor"
    }

    with mlflow.start_run(
            experiment_id=experiment.experiment_id,
            run_name=run_name,
            tags=tags
    ) as run:
        logged_model_uri = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name="xgb_pipeline_taxi_regressor",
            signature=signature,
        ).model_uri

        from sklearn.utils import estimator_html_repr

        # Generate estimator HTML
        estimator_html = estimator_html_repr(pipeline)

        # Save the HTML to a file
        html_path = "estimator.html"
        with open("estimator.html", "w", encoding="utf-8") as f:
            f.write(estimator_html)
        mlflow.log_artifact("estimator.html", artifact_path="artifacts")

        fig = create_plots(pipeline, X_test, y_test)

        mlflow.log_figure(fig, 'plots/predicted_vs_actual.png')
        mlflow.log_metrics(metrics)

        run_id = run.info.run_id
        log_config(experiment_id=experiment.experiment_id, run_id=run_id)

    print("Pipeline logged successfully!")

    return logged_model_uri

