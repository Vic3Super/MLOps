import json

import numpy as np
from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd
import os
from google.cloud import bigquery
from datetime import datetime
from src.extract_data import extract_data_for_predictions

"""
# Check if GOOGLE_APPLICATION_CREDENTIALS is set
credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if credentials_path and os.path.exists(credentials_path):
    print(f"Using Google Cloud credentials from: {credentials_path}")
else:
    print("GOOGLE_APPLICATION_CREDENTIALS is not set or the file does not exist!")
"""
from google.auth import default
from google.auth.exceptions import DefaultCredentialsError

try:
    # Get default credentials
    credentials, project = default()
    print(f"Using Google Cloud credentials for project: {project}")

except DefaultCredentialsError:
    print("No Google Cloud credentials found. Ensure the service account has the correct IAM permissions.")


# Load the MLflow model when the application starts
RUN_ID = os.getenv("RUN_ID", "gs://mlflow-bucket-1998/mlruns/2/a741f99410cc4e7d8cf8dc6eb0922ffb")  # Use environment variable for flexibility
print(f"Loading model at {RUN_ID}/artifacts/model")
model = mlflow.pyfunc.load_model(f"{RUN_ID}/artifacts/model")
print(f"Loaded model at {RUN_ID}/artifacts/model")

# Initialize Flask app
app = Flask(__name__)
print(f"App initialized.")


# Initialize BigQuery client
bq_client = bigquery.Client()
project_id = os.getenv("PROJECT_ID", "carbon-relic-439014-t0")
dataset_name = os.getenv("BQ_DATASET_NAME", "chicago_taxi")
table_name = os.getenv("BQ_TABLE_NAME", "prediction")
table_id = f"{project_id}.{dataset_name}.{table_name}"

# Ensure table schema (create if not exists)
def create_table_if_not_exists():
    schema = [
        bigquery.SchemaField("Trip ID", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("Taxi ID", "BYTES", mode="NULLABLE"),
        bigquery.SchemaField("Trip Start Timestamp", "TIMESTAMP", mode="NULLABLE"),
        bigquery.SchemaField("Trip End Timestamp", "TIMESTAMP", mode="NULLABLE"),
        bigquery.SchemaField("Trip Seconds", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("Trip Miles", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("Pickup Census Tract", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("Dropoff Census Tract", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("Pickup Community Area", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("Dropoff Community Area", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("Fare", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("Tips", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("Tolls", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("Extras", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("Trip Total", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("Payment Type", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("Company", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("Pickup Centroid Latitude", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("Pickup Centroid Longitude", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("Pickup Centroid Location", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("Dropoff Centroid Latitude", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("Dropoff Centroid Longitude", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("Dropoff Centroid Location", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("timestamp", "TIMESTAMP"),
        bigquery.SchemaField("prediction", "FLOAT"),
    ]
    """
    schema = [
        bigquery.SchemaField("timestamp", "TIMESTAMP"),
        bigquery.SchemaField("input_data", "STRING"),
        bigquery.SchemaField("prediction", "STRING"),
        bigquery.SchemaField("truth", field_type="STRING")
    ]
    """
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

        print(f"Data loaded")

        data_extracted = extract_data_for_predictions(data)

        # Pretend we also get the ground truth some way
        truth = data["Trip Total"]
        data_extracted = data.drop(columns=['Trip Total'])

        # Make predictions
        predictions = model.predict(data_extracted)

        input_data_serialized = json.dumps(input_data)
        # Store input and predictions in BigQuery

        rows_to_insert = []

        print(f"Begin iteration")
        # Iterate over the rows in the DataFrame
        for index, row in data.iterrows():
            rows_to_insert.append({
                "Trip ID": row.get("Trip ID", None),
                "Taxi ID": row.get("Taxi ID", None),
                "Trip Start Timestamp": row.get("Trip Start Timestamp", None),
                "Trip End Timestamp": row.get("Trip End Timestamp", None),
                "Trip Seconds": row.get("Trip Seconds", None),
                "Trip Miles": row.get("Trip Miles", None),
                "Pickup Census Tract": row.get("Pickup Census Tract", None),
                "Dropoff Census Tract": row.get("Dropoff Census Tract", None),
                "Pickup Community Area": row.get("Pickup Community Area", None),
                "Dropoff Community Area": row.get("Dropoff Community Area", None),
                "Fare": row.get("Fare", None),
                "Tips": row.get("Tips", None),
                "Tolls": row.get("Tolls", None),
                "Extras": row.get("Extras", None),
                "Trip Total": row.get("Trip Total", None),
                "Payment Type": row.get("Payment Type", None),
                "Company": row.get("Company", None),
                "Pickup Centroid Latitude": row.get("Pickup Centroid Latitude", None),
                "Pickup Centroid Longitude": row.get("Pickup Centroid Longitude", None),
                "Pickup Centroid Location": row.get("Pickup Centroid Location", None),
                "Dropoff Centroid Latitude": row.get("Dropoff Centroid Latitude", None),
                "Dropoff Centroid Longitude": row.get("Dropoff Centroid Longitude", None),
                "Dropoff Centroid Location": row.get("Dropoff Centroid Location", None),
                "timestamp": datetime.utcnow().isoformat(),
                "prediction": float(predictions[index])
            })

        print(f"Iteration over")
        print(f"rows_to_insert: {rows_to_insert}")
        errors = bq_client.insert_rows_json(table_id, rows_to_insert)
        print(f"Error while inserting rows: {errors}")
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