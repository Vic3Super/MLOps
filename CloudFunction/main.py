import json
import os

import pandas as pd
from google.cloud import bigquery
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from datetime import datetime
from google.cloud import pubsub_v1

def extract_features(data):
    # Select only relevant columns
    df = data[["Trip Seconds", "Trip Miles", "Company"]]
    df = df.dropna()
    return df


def extract_target(data):
    df = data[["Trip Total"]].rename(columns={"Trip Total": "target"})
    df = df.dropna()
    return df

def initialize_bigquery_client():
    """Initialize and return the BigQuery client."""
    return bigquery.Client()


def get_bigquery_data(client, project_id, dataset_name, table_name):
    """Fetch data from a BigQuery table and return it as a pandas DataFrame."""

    table_ref = client.dataset(dataset_name).table(table_name)
    table = client.get_table(table_ref)
    #columns = [field.name for field in table.schema if field.name not in ["prediction", "timestamp"]]
    #query = f"SELECT {', '.join([f'`{column}`' for column in columns])} FROM `{project_id}.{dataset_name}.{table_name}`"
    query = f"SELECT * FROM `{project_id}.{dataset_name}.{table_name}`"
    query_job = client.query(query)  # API request
    return query_job.result().to_dataframe()



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
    y_actual = current_df["Trip Total"].astype(float)
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
    PUBSUB_TOPIC = os.getenv("PUBSUB_TOPIC", "monitoring_job")
    TABLE_ID = os.getenv("TABLE_ID", "monitoring_job")
    # Initialize the BigQuery client
    client = initialize_bigquery_client()

    # Fetch reference and current data
    result_cur = get_bigquery_data(client, PROJECT_ID, DATASET_NAME, "prediction")
    result_ref = get_bigquery_data(client, PROJECT_ID, DATASET_NAME, "raw_data")

    features_cur = extract_features(result_cur)
    features_ref = extract_features(result_ref)

    target_cur = extract_target(result_cur)
    target_ref = extract_target(result_ref)

    mae, r2, rmse = run_performance_analysis(result_cur)

    performance_metrics = {
        "mae": mae,
        "r2": r2,
        "rmse": rmse,
    }

    # Run data drift analysis
    report_dict_data = run_data_drift_analysis(features_ref, features_cur)
    report_dict_target = run_target_drift_analysis(target_ref, target_cur)

    publish(performance_metrics, report_dict_data, report_dict_target, PUBSUB_TOPIC, PROJECT_ID, DATASET_NAME, TABLE_ID)

    return 'Triggered successfully', 200

def publish(performance_metrics, report_dict_data, report_dict_target, topic_name, project_id, dataset_id, table_id):

    # publishing logic
    # if bad metrics, send pub/sub message to trigger new build
    # else, store data simply in bigquery

    # Example message aligned with BigQuery schema
    message_data = {
        "timestamp": datetime.utcnow().isoformat(),  # Current timestamp in ISO 8601 format
        "data_drift_report": json.dumps(report_dict_data),
        "target_drift_report": json.dumps(report_dict_target),
        "MAE": performance_metrics["mae"],
        "R2": performance_metrics["r2"],
        "RMSE": performance_metrics["rmse"],
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

"""
if __name__ == '__main__':
    from functions_framework import create_app
    app = create_app('cloud_function_entry_point')
    app.run(port=8080)
"""