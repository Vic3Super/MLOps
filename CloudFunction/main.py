import json
import os
import numpy as np
import pandas as pd
from google.cloud import bigquery
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from datetime import datetime
from google.cloud import pubsub_v1
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def extract_features(df):
    df['trip_start_timestamp'] = df['trip_start_timestamp'].dt.tz_localize(None).astype('datetime64[ns]')

    # Extract features from trip_start_timestamp
    df["daytime"] = df["trip_start_timestamp"].dt.hour.astype("float")
    df['day_type'] = df['trip_start_timestamp'].dt.weekday.apply(lambda x: 'weekend' if x >= 5 else 'weekday').astype('string')
    df['month'] = df['trip_start_timestamp'].dt.month.astype("float")
    df['day_of_week'] = df['trip_start_timestamp'].dt.dayofweek.astype("float")
    df['day_of_month'] = df['trip_start_timestamp'].dt.day.astype("float")
    #df.drop(columns=["trip_start_timestamp"], inplace=True)
    df = df[["daytime", "day_type", "month", "day_of_week", "day_of_month",
             "trip_miles", "tolls", "extras", "avg_tips", "payment_type", "company",
             'pickup_community_area', 'pickup_latitude', 'pickup_longitude']]
    return df

def extract_target(data):
    df = data[["trip_total"]].rename(columns={"trip_total": "target"})
    df = df.dropna()
    return df

def initialize_bigquery_client():
    """Initialize and return the BigQuery client."""
    return bigquery.Client()

def get_training_data(client, project_id, dataset_name):
    table_id = f"{project_id}.{dataset_name}.training_data"
    query = f"SELECT * FROM `{table_id}` WHERE status = 'ACTIVE'"
    query_job = client.query(query)
    # Convert to DataFrame
    df = query_job.result().to_dataframe()

    # Convert all object-type columns to string
    df = df.astype({col: "string" for col in df.select_dtypes(include=["object"]).columns})  # Convert object to string
    df = df.astype({col: "float64" for col in df.select_dtypes(include=["int64"]).columns})  # Convert int to float
    df = df.drop(columns=["status"])
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
    WHERE p.timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY);
    """
    query_job = client.query(query)
    # Convert to DataFrame
    df = query_job.result().to_dataframe()

    if df.empty:
        return None

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

def run_performance_analysis(current_df, outlier_threshold=1.5):
    # Convert to float
    y_actual = current_df["ground_truth"].astype(float)
    y_pred = current_df["prediction"].astype(float)

    # Detect and remove outliers using the IQR method
    Q1 = np.percentile(y_actual, 25)
    Q3 = np.percentile(y_actual, 75)
    IQR = Q3 - Q1

    lower_bound = Q1 - outlier_threshold * IQR
    upper_bound = Q3 + outlier_threshold * IQR

    # Mask to filter out extreme values
    valid_mask = (y_actual >= lower_bound) & (y_actual <= upper_bound)
    y_actual_filtered = y_actual[valid_mask]
    y_pred_filtered = y_pred[valid_mask]

    # Ensure there are enough valid points to compute metrics
    if len(y_actual_filtered) < 2:
        return {
            "Error": "Not enough valid data points after removing outliers."
        }

    # Compute errors
    mae = mean_absolute_error(y_actual_filtered, y_pred_filtered)
    rmse = mean_squared_error(y_actual_filtered, y_pred_filtered, squared=False)  # squared=False returns RMSE
    r2 = r2_score(y_actual_filtered, y_pred_filtered)

    # Standardized errors
    mean_actual = np.mean(y_actual_filtered)
    actual_range = np.ptp(y_actual_filtered)  # ptp = max - min

    mae_standardized = mae / mean_actual if mean_actual != 0 else np.nan
    rmse_standardized = rmse / mean_actual if mean_actual != 0 else np.nan

    mae_minmax = mae / actual_range if actual_range != 0 else np.nan
    rmse_minmax = rmse / actual_range if actual_range != 0 else np.nan


    return {
        "Data Points Used": len(y_actual_filtered),
        "MAE": mae,
        "MAE_standardized_by_mean": mae_standardized,
        "MAE_standardized_by_range": mae_minmax,
        "RMSE": rmse,
        "RMSE_standardized_by_mean": rmse_standardized,
        "RMSE_standardized_by_range": rmse_minmax,
        "R2": r2
    }

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

    # Initialize the BigQuery client
    client = initialize_bigquery_client()

    training_data = get_training_data(client, PROJECT_ID, DATASET_NAME)
    new_data = get_new_data(client, PROJECT_ID, DATASET_NAME)

    if new_data.empty:
        return "No new data to monitor", 200

    new_data = new_data.dropna(subset=["ground_truth"])

    features_training_data = training_data.drop(columns="trip_total")
    features_new_data = extract_features(new_data)

    target_training_data = extract_target(training_data)
    target_new_data = extract_target(new_data)


    performance_metrics = run_performance_analysis(new_data)

    # Run data drift analysis
    report_dict_data = run_data_drift_analysis(features_training_data, features_new_data)
    report_dict_target = run_target_drift_analysis(target_training_data, target_new_data)

    publish(performance_metrics, report_dict_data, report_dict_target, PROJECT_ID)

    return 'Triggered successfully', 200

def publish(performance_metrics, report_dict_data, report_dict_target, project_id):

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



    retrain = False
    if performance_metrics["MAE_standardized_by_mean"] > 0.3 or performance_metrics["R2"] <= 0.8:
        retrain = True

    # Construct message data with fixed JSON serialization
    message_data = {
        "timestamp": datetime.utcnow().isoformat(),  # Current timestamp in ISO 8601 format
        "data_drift_report": json.dumps(report_dict_data),
        "target_drift_report": json.dumps(report_dict_target),
        "performance_metrics":json.dumps(performance_metrics),
        "retrain": retrain,
    }

    if retrain:
        # project_id = os.getenv("PROJECT_ID", "carbon-relic-439014-t0")
        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path(project_id, "monitoring_retrain")

        message_bytes = json.dumps(message_data).encode("utf-8")
        future = publisher.publish(topic_path, message_bytes)
        future.result()  # Wait for the publishing to complete
        return

    # project_id = os.getenv("PROJECT_ID", "carbon-relic-439014-t0")
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project_id, "monitoring_alert")

    message_bytes = json.dumps(message_data).encode("utf-8")
    future = publisher.publish(topic_path, message_bytes)
    future.result()  # Wait for the publishing to complete
    return


