import numpy as np
from flask import Flask, request, jsonify
import mlflow
import pandas as pd
import os
import logging
from google.cloud import bigquery
from datetime import datetime

from mlflow import MlflowClient



# Initialize BigQuery client
bq_client = bigquery.Client()
project_id = os.getenv("PROJECT_ID", "carbon-relic-439014-t0")
dataset_name = os.getenv("DATASET_NAME", "chicago_taxi")
table_name_prediction = os.getenv("TABLE_NAME_PREDICTIONS", "trip_prediction")
table_name_data = os.getenv("TABLE_NAME_DATA", "data")
table_name_drivers = os.getenv("TABLE_NAME_DRIVERS", "driver_aggregates")

table_id_prediction = f"{project_id}.{dataset_name}.{table_name_prediction}"
table_id_data = f"{project_id}.{dataset_name}.{table_name_data}"
table_id_drivers = f"{project_id}.{dataset_name}.{table_name_drivers}"

TEST_RUN = os.getenv("TEST_RUN", "True").lower() == "true"
#CHALLENGER = os.getenv("CHALLENGER", "False").lower() == "true"
MODEL_TYPE = os.getenv("MODEL_TYPE", "champion")
# Configure logging
logging.basicConfig(level=logging.INFO)

TRACKING_URI = "https://mlflow-service-974726646619.us-central1.run.app"
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.autolog(disable=True)

client = MlflowClient()
#run_id = client.get_latest_versions("xgb_pipeline_taxi_regressor")[0].run_id


run_id = client.get_model_version_by_alias("xgb_pipeline_taxi_regressor", MODEL_TYPE).run_id


model_path = f"gs://mlflow-bucket-1998/mlruns/2/{run_id}/artifacts/model"
requirements_path =  mlflow.artifacts.download_artifacts(f"{model_path}/requirements.txt")

model = mlflow.pyfunc.load_model(model_path)

# Initialize Flask app
app = Flask(__name__)
print(f"App initialized.")



def extract_time_features(df):


    df["trip_start_timestamp"] = pd.to_datetime(df["trip_start_timestamp"])


    # Extract features from trip_start_timestamp
    df["daytime"] = df["trip_start_timestamp"].dt.hour
    df['day_type'] = df['trip_start_timestamp'].dt.weekday.apply(lambda x: 'weekend' if x >= 5 else 'weekday').astype(str)
    df['month'] = df['trip_start_timestamp'].dt.month
    df['day_of_week'] = df['trip_start_timestamp'].dt.dayofweek
    df['day_of_month'] = df['trip_start_timestamp'].dt.day
    #df.drop(columns=["trip_start_timestamp"], inplace=True)

    return df

def convert_features(df):
    # Convert numeric columns, ensuring None/NaN values do not cause errors
    numeric_cols = [
        "trip_miles", "tolls", "extras", "avg_tips",
        "pickup_latitude", "pickup_longitude", "pickup_community_area"
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Convert categorical columns
    df["payment_type"] = df["payment_type"].astype("str")

    return df


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse the JSON input
        input_data = request.get_json()

        # Convert input to Pandas DataFrame
        data = pd.DataFrame(input_data['data'], columns=input_data['columns'])
        logging.info("Input Data loaded")

        data = extract_time_features(data)

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

        data = convert_features(data)
        #prediction_data = data_extracted.drop(columns=["unique_key", "taxi_id"])
        predictions = model.predict(data)

        # Prepare data for BigQuery
        timestamp_now = datetime.utcnow().isoformat()

        data.drop(inplace=True, columns=['avg_tips', "daytime", "day_of_week", "day_of_month", "month", "day_type"])
        # Convert data to dictionary format
        rows_to_insert_data = []
        rows_to_insert_prediction = []


        for row, pred in zip(data.to_dict(orient="records"), predictions):
            # Add timestamp to row data
            row["timestamp"] = timestamp_now
            if isinstance(row.get("trip_start_timestamp"), pd.Timestamp):
                row["trip_start_timestamp"] = row["trip_start_timestamp"].isoformat()

            # Append to data insertion list
            rows_to_insert_data.append(row)
            # Append to prediction insertion list
            rows_to_insert_prediction.append({
                "unique_key": row.get("unique_key"),
                "prediction": float(pred),
                "ground_truth": row.get("trip_total"),
                "timestamp": timestamp_now,
                #"run_id":RUN_ID # to check versioning
            })

        logging.info(f"Prepared {len(rows_to_insert_data)} rows for data table")
        logging.info(f"Prepared {len(rows_to_insert_prediction)} rows for prediction table")

        if not TEST_RUN:
            logging.info(f"Logging to BigQuery")
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