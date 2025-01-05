import json

from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd
import os
from google.cloud import bigquery
from datetime import datetime
from src.extract_data import extract_data_for_predictions


# Check if GOOGLE_APPLICATION_CREDENTIALS is set
credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if credentials_path and os.path.exists(credentials_path):
    print(f"Using Google Cloud credentials from: {credentials_path}")
else:
    print("GOOGLE_APPLICATION_CREDENTIALS is not set or the file does not exist!")

# Load the MLflow model when the application starts
RUN_ID = os.getenv("RUN_ID", "gs://mlflow-bucket-1998/mlruns/2/0d73288a96d84fe5b8f4ca6219bb3df9/artifacts/model")  # Use environment variable for flexibility
print(f"Loading model at {RUN_ID}/artifacts/model")
model = mlflow.pyfunc.load_model(f"{RUN_ID}/artifacts/model")
print(f"Loaded model at {RUN_ID}/artifacts/model")

# Initialize Flask app
app = Flask(__name__)
print(f"App initialized.")


# Initialize BigQuery client
bq_client = bigquery.Client()
project_id = os.getenv("PROJECT_ID", "carbon-relic-439014-t0")  # Replace with your GCP project ID
dataset_name = os.getenv("BQ_DATASET_NAME", "chicago_taxi")  # Replace with your BigQuery dataset name
table_name = os.getenv("BQ_TABLE_NAME", "prediction")  # Replace with your BigQuery table name
table_id = f"{project_id}.{dataset_name}.{table_name}"

# Ensure table schema (create if not exists)
def create_table_if_not_exists():
    schema = [
        bigquery.SchemaField("timestamp", "TIMESTAMP"),
        bigquery.SchemaField("input_data", "STRING"),
        bigquery.SchemaField("prediction", "STRING")
    ]
    try:
        bq_client.get_table(table_id)  # Check if table exists
        print(f"BigQuery table {table_id} already exists.")
    except Exception as e:
        print(f"Creating BigQuery table {table_id}.")
        table = bigquery.Table(table_id, schema=schema)
        bq_client.create_table(table)
        print(f"Table {table_id} created.")

create_table_if_not_exists()


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse the JSON input
        input_data = request.get_json()

        # Convert input to Pandas DataFrame
        data = pd.DataFrame(input_data['data'], columns=input_data['columns'])

        data = extract_data_for_predictions(data)

        # Make predictions
        predictions = model.predict(data)

        input_data_serialized = json.dumps(input_data)
        # Store input and predictions in BigQuery
        rows_to_insert = [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "input_data": input_data_serialized,
                "prediction": str(predictions.tolist())
            }
        ]
        errors = bq_client.insert_rows_json(table_id, rows_to_insert)
        if errors:
            print(f"BigQuery insertion errors: {errors}")
            return jsonify({"error": "Failed to store predictions in BigQuery"}), 500

        # Return predictions as JSON
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        print(f"Exception: {e}")
        return jsonify({"error": str(e)}), 400


# Health check endpoint
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    # Use the PORT environment variable or default to 8080
  #  port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=8080, debug=False)