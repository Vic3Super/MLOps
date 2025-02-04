import json
import os

import numpy as np
from feast import FeatureStore, FeatureView
import pandas as pd
from google.cloud import bigquery
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from datetime import datetime
from google.cloud import pubsub_v1

def extract_features(data):
    store = FeatureStore(repo_path=".")
    feature_service = store.get_feature_service("taxi_drive")
    # Iterate over the feature view projections and extract all features.
    used_features = [
        feature.name
        for projection in feature_service.feature_view_projections
        for feature in projection.features
    ]
    used_features.remove("trip_total")
    return data[used_features]

def extract_target(data):
    df = data[["trip_total"]].rename(columns={"trip_total": "target"})
    df = df.dropna()
    return df

def initialize_bigquery_client():
    """Initialize and return the BigQuery client."""
    return bigquery.Client()

def get_training_data(client, project_id, dataset_name):
    table_id = f"{project_id}.{dataset_name}.training_data"
    query = f"SELECT * FROM {table_id}"
    query_job = client.query(query)
    # Convert to DataFrame
    df = query_job.result().to_dataframe()

    # Convert all object-type columns to string
    df = df.astype({col: "string" for col in df.select_dtypes(include=["object"]).columns})  # Convert object to string
    df = df.astype({col: "float64" for col in df.select_dtypes(include=["int64"]).columns})  # Convert int to float

    return df


def get_new_data(client, project_id, dataset_name):
    table_id = f"{project_id}.{dataset_name}.trip_prediction"

    query = """
    SELECT *
    FROM chicago_taxi.trip_prediction AS p
    LEFT JOIN chicago_taxi.data AS d
    ON p.unique_key = d.unique_key
    LEFT JOIN chicago_taxi.driver_aggregates as a
    ON d.taxi_id = a.taxi_id
    WHERE p.timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR);
    """
    query_job = client.query(query)
    # Convert to DataFrame
    df = query_job.result().to_dataframe()

    # Convert all object-type columns to string
    df = df.astype({col: "string" for col in df.select_dtypes(include=["object"]).columns})
    df = df.astype({col: "float64" for col in df.select_dtypes(include=["int64"]).columns})  # Convert int to float

    return df


def preprocess_data(reference_df, current_df, extract_function):
    """Preprocess reference and current data using the provided extract function."""
    return extract_function(reference_df), extract_function(current_df)

def run_data_drift_analysis(reference_df, current_df):
    """Run data drift analysis using Evidently and return the report as a dictionary."""
    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(reference_data=reference_df, current_data=current_df)
    return data_drift_report.as_dict()

def run_target_drift_analysis(reference_df, current_df):
    """Run data drift analysis using Evidently and return the report as a dictionary."""
    target_drift_report = Report(metrics=[TargetDriftPreset()])
    target_drift_report.run(reference_data=reference_df, current_data=current_df)
    return target_drift_report.as_dict()

def run_performance_analysis(current_df):
    # Actual and predicted values
    y_actual = current_df["ground_truth"].astype(float)
    y_pred = current_df["prediction"].astype(float)

    # Mean Absolute Error (MAE)
    mae = abs(y_actual - y_pred).mean()

    # R-squared
    ss_res = ((y_actual - y_pred) ** 2).sum()
    ss_tot = ((y_actual - y_actual.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot)

    # Root Mean Squared Error (RMSE)
    rmse = (((y_actual - y_pred) ** 2).mean()) ** 0.5

    return mae, r2, rmse


def cloud_function_entry_point(request):
    """Cloud Function entry point that only processes POST requests."""

    if request.method != 'POST':
        return 'Only POST requests are allowed', 405

    # Proceed with processing when a POST request is received
    return process_event()



def process_event():
    """Cloud Function entry point."""
    # Load configuration
    PROJECT_ID = os.getenv("PROJECT_ID", "carbon-relic-439014-t0")
    DATASET_NAME = os.getenv("DATASET_NAME", "chicago_taxi")
    PUBSUB_TOPIC = os.getenv("PUBSUB_TOPIC", "testing")
    TABLE_ID = os.getenv("TABLE_ID", "monitoring_job")
    # Initialize the BigQuery client
    client = initialize_bigquery_client()

    training_data = get_training_data(client, PROJECT_ID, DATASET_NAME)
    new_data = get_new_data(client, PROJECT_ID, DATASET_NAME)

    features_training_data = extract_features(training_data)
    features_new_data = extract_features(new_data)

    target_training_data = extract_target(training_data)
    target_new_data = extract_target(new_data)


    mae, r2, rmse = run_performance_analysis(new_data)

    performance_metrics = {
        "mae": mae,
        "r2": r2,
        "rmse": rmse,
    }

    # Run data drift analysis
    report_dict_data = run_data_drift_analysis(features_training_data, features_new_data)
    report_dict_target = run_target_drift_analysis(target_training_data, target_new_data)

    publish(performance_metrics, report_dict_data, report_dict_target, PUBSUB_TOPIC, PROJECT_ID, DATASET_NAME, TABLE_ID)

    return 'Triggered successfully', 200

def publish(performance_metrics, report_dict_data, report_dict_target, topic_name, project_id, dataset_id, table_id):

    # publishing logic
    # if bad metrics, send pub/sub message to trigger new build
    # else, store data simply in bigquery

    def convert_numpy_types(obj):
        """Recursively convert NumPy types to standard Python types."""
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        elif isinstance(obj, np.integer):  # Convert np.int64, np.int32
            return int(obj)
        elif isinstance(obj, np.floating):  # Convert np.float64, np.float32
            return float(obj)
        else:
            return obj  # Return original value if not a NumPy type

    # Convert NumPy types before JSON serialization
    report_dict_data = convert_numpy_types(report_dict_data)
    report_dict_target = convert_numpy_types(report_dict_target)

    # Construct message data with fixed JSON serialization
    message_data = {
        "timestamp": datetime.utcnow().isoformat(),  # Current timestamp in ISO 8601 format
        "data_drift_report": json.dumps(report_dict_data),
        "target_drift_report": json.dumps(report_dict_target),
        "MAE": float(performance_metrics["mae"]),  # Ensure float conversion
        "R2": float(performance_metrics["r2"]),
        "RMSE": float(performance_metrics["rmse"]),
    }

    retrain = False
    pubsub_message = ""
    if performance_metrics["mae"] > 10 or performance_metrics["r2"] < 0.5 or performance_metrics["rmse"] > 2:
        retrain = True

    drifted_columns = report_dict_data['metrics'][1]['result']['drift_by_columns']
    for column, details in drifted_columns.items():
        if details["drift_detected"]:
            retrain = True

    target_drift = report_dict_target["metrics"][0]["result"]["drift_detected"]
    if target_drift:
        retrain = True

    if retrain:
        # project_id = os.getenv("PROJECT_ID", "carbon-relic-439014-t0")
        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path(project_id, topic_name)



        message_bytes = json.dumps(message_data).encode("utf-8")
        future = publisher.publish(topic_path, message_bytes)
        future.result()  # Wait for the publishing to complete
        return

    # Initialize BigQuery client
    client = bigquery.Client()

    # Define your BigQuery table
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    # Prepare row to insert
    rows_to_insert = [message_data]

    # Insert rows into BigQuery
    errors = client.insert_rows_json(table_ref, rows_to_insert)
