import json
import os

import pandas as pd
from google.cloud import bigquery
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def initialize_bigquery_client():
    """Initialize and return the BigQuery client."""
    return bigquery.Client()

def get_bigquery_data(client, project_id, dataset_name, table_name):
    """Fetch data from a BigQuery table and return it as a pandas DataFrame."""
    table_id = f"{project_id}.{dataset_name}.{table_name}"
    query = f"SELECT * FROM `{table_id}`"
    query_job = client.query(query)  # API request
    return query_job.result().to_dataframe()

def parse_json_column_to_dataframe(dataframe, column_name):
    """Parse a JSON column into individual dataframes and concatenate them."""
    dataframes = []
    for json_string in dataframe[column_name]:
        json_data = json.loads(json_string)
        df = pd.DataFrame(data=json_data["data"], columns=json_data["columns"])
        dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)

def preprocess_data(reference_df, current_df, extract_function):
    """Preprocess reference and current data using the provided extract function."""
    return extract_function(reference_df), extract_function(current_df)

def run_data_drift_analysis(reference_df, current_df):
    """Run data drift analysis using Evidently and return the report as a dictionary."""
    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(reference_data=reference_df, current_data=current_df)
    return data_drift_report.as_dict()

def cloud_function_entry_point(request):
    """Cloud Function entry point."""
    # Load configuration
    PROJECT_ID = os.getenv("PROJECT_ID","carbon-relic-439014-t0")
    DATASET_NAME = os.getenv("DATASET_NAME","chicago_taxi")

    # Initialize the BigQuery client
    client = initialize_bigquery_client()

    # Fetch reference and current data
    result_cur = get_bigquery_data(client, PROJECT_ID, DATASET_NAME, "prediction")
    result_ref_df = get_bigquery_data(client, PROJECT_ID, DATASET_NAME, "raw_data")

    # Parse and preprocess data
    result_cur_df = parse_json_column_to_dataframe(result_cur, "input_data")

    # Import your custom extraction function
    from extract_data import extract_data
    result_ref_df, result_cur_df = preprocess_data(result_ref_df, result_cur_df, extract_data)

    # Run data drift analysis
    report_dict = run_data_drift_analysis(result_ref_df, result_cur_df)
    drifted_columns = report_dict['metrics'][1]['result']['drift_by_columns']
    message = []
    for column, details in drifted_columns.items():
        if details["drift_detected"]:
            message.append(f"Drift Detected for '{column}' with Drift Score: '{details['drift_score']}'")

    print(message)
    return json.dumps({"drift_report": report_dict, "drift_summary": message}, indent=2)
