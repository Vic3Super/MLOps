import numpy as np
from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd
import os
import logging
from google.cloud import bigquery
from datetime import datetime


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
RUN_ID = os.getenv("RUN_ID", "gs://mlflow-bucket-1998/mlruns/2/80162f97da5d4f5fa928bf1f386e1dd4")  # Use environment variable for flexibility
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
table_name_prediction = os.getenv("BQ_TABLE_NAME_PREDICTIONS", "trip_prediction")
table_name_data = os.getenv("BQ_TABLE_NAME_DATA", "data")
table_name_drivers = os.getenv("BQ_TABLE_NAME_DRIVERS", "driver_aggregates")

table_id_prediction = f"{project_id}.{dataset_name}.{table_name_prediction}"
table_id_data = f"{project_id}.{dataset_name}.{table_name_data}"
table_id_drivers = f"{project_id}.{dataset_name}.{table_name_drivers}"

# Configure logging
logging.basicConfig(level=logging.INFO)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse the JSON input
        input_data = request.get_json()

        # Convert input to Pandas DataFrame
        data = pd.DataFrame(input_data['data'], columns=input_data['columns'])
        logging.info("Input Data loaded")

        taxi_ids = data['taxi_id'].unique().tolist()

        query = f"""
        SELECT taxi_id, avg_tips
        FROM `{table_id_drivers}`
        WHERE taxi_id IN UNNEST(@taxi_ids)
        """


        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("taxi_ids", "STRING", taxi_ids)
            ]
        )

        query_job = bq_client.query(query, job_config=job_config)
        avg_tips_df = query_job.to_dataframe()

        data = data.merge(avg_tips_df, on='taxi_id', how='left')

        data = data.replace(np.nan, None)

        #prediction_data = data_extracted.drop(columns=["unique_key", "taxi_id"])
        predictions = model.predict(data)

        # Prepare data for BigQuery
        timestamp_now = datetime.utcnow().isoformat()

        data.drop(inplace=True, columns=['avg_tips'])
        # Convert data to dictionary format
        rows_to_insert_data = []
        rows_to_insert_prediction = []

        for row, pred in zip(data.to_dict(orient="records"), predictions):
            # Add timestamp to row data
            row["timestamp"] = timestamp_now

            # Append to data insertion list
            rows_to_insert_data.append(row)

            # Append to prediction insertion list
            rows_to_insert_prediction.append({
                "unique_key": row.get("unique_key"),
                "prediction": float(pred),
                "ground_truth": row.get("trip_total"),
                "timestamp": timestamp_now,
            })

        logging.info(f"Prepared {len(rows_to_insert_data)} rows for data table")
        logging.info(f"Prepared {len(rows_to_insert_prediction)} rows for prediction table")



        # Insert into BigQuery

        errors_data = bq_client.insert_rows_json(table_id_data, rows_to_insert_data)
        if errors_data:
            logging.error(f"BigQuery Data Table Insertion Errors: {errors_data}")
            return jsonify({"error": "Failed to store data in BigQuery"}), 500

        errors_prediction = bq_client.insert_rows_json(table_id_prediction, rows_to_insert_prediction)
        if errors_prediction:
            logging.error(f"BigQuery Prediction Table Insertion Errors: {errors_prediction}")
            return jsonify({"error": "Failed to store predictions in BigQuery"}), 500

        return jsonify({"predictions": predictions.tolist()})

    except Exception as e:
        logging.exception("Exception occurred during prediction")
        return jsonify({"error": str(e)}), 400


# Health check endpoint
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    # Use the PORT environment variable or default to 8080
  #  port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=8080, debug=False)