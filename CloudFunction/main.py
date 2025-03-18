from google.cloud.run_v2 import TrafficTargetAllocationType
import json
import os
import numpy as np
import pandas as pd
from google.cloud import bigquery, run_v2
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from datetime import datetime
from google.cloud import pubsub_v1
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import mlflow
from mlflow import MlflowClient
import logging


PROJECT_ID = os.getenv("PROJECT_ID", "carbon-relic-439014-t0")
REGION = os.getenv("REGION", "us-west1")
DATASET_NAME = os.getenv("DATASET_NAME", "chicago_taxi")
SERVICE_NAME = os.getenv("SERVICE_NAME", "my-mlflow-app")



logging.basicConfig(level=logging.INFO)

class NumpyTypeEncoder(json.JSONEncoder):
    # whacky https://gist.github.com/jonathanlurie/1b8d12f938b400e54c1ed8de21269b65
    # for uploading json to bigquery
    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def extract_features(df):
    df['trip_start_timestamp'] = df['trip_start_timestamp'].dt.tz_localize(None).astype('datetime64[ns]')

    # Extract features from trip_start_timestamp
    df["daytime"] = df["trip_start_timestamp"].dt.hour.astype("float")
    df['day_type'] = df['trip_start_timestamp'].dt.weekday.apply(lambda x: 'weekend' if x >= 5 else 'weekday').astype(
        'string')
    df['month'] = df['trip_start_timestamp'].dt.month.astype("float")
    df['day_of_week'] = df['trip_start_timestamp'].dt.dayofweek.astype("float")
    df['day_of_month'] = df['trip_start_timestamp'].dt.day.astype("float")
    # df.drop(columns=["trip_start_timestamp"], inplace=True)
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


def get_training_data(client, project_id, dataset_name, model_run_id):
    table_id = f"{project_id}.{dataset_name}.training_data"
    query = f"SELECT * FROM `{table_id}` WHERE model_run_id = '{model_run_id}'"
    query_job = client.query(query)
    # Convert to DataFrame
    df = query_job.result().to_dataframe()

    # Convert all object-type columns to string
    df = df.astype({col: "string" for col in df.select_dtypes(include=["object"]).columns})  # Convert object to string
    df = df.astype({col: "float64" for col in df.select_dtypes(include=["int64"]).columns})  # Convert int to float
    df.drop(columns=["model_run_id"], inplace=True)
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
    WHERE p.timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY);
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
    if current_df.empty:
        return {}

    # Convert to float
    y_actual = current_df["ground_truth"].astype(float)
    y_pred = current_df["prediction"].astype(float)

    # Detect and remove outliers using the IQR method since real data can also be incorrect
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
    actual_range = np.ptp(y_actual_filtered)

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


def get_cloud_run_revisions():
    """Fetch all active (SERVING) revisions and determine which one is the Challenger and Champion based on environment variables."""
    client = run_v2.RevisionsClient()
    parent = f"projects/{PROJECT_ID}/locations/{REGION}/services/{SERVICE_NAME}"
    revisions = client.list_revisions(parent=parent)

    champion_revision = None
    challenger_revision = None

    for revision in revisions:

        # Check if the revision is retired
        is_active = True
        for condition in revision.conditions:
            if condition.message == "Revision retired.":
                is_active = False
                break  # No need to check further conditions for this revision

        if not is_active:
            continue  # Skip retired revisions

        revision_name = revision.name.split("/")[-1]  # Extract revision name

        # Skip if containers or environment variables are not defined
        if not revision.containers or not revision.containers[0].env:
            continue

        env_vars = {env.name: env.value for env in revision.containers[0].env}

        if env_vars.get("MODEL_TYPE") == "champion":
            champion_revision = revision_name
        elif env_vars.get("MODEL_TYPE") == "challenger":
            challenger_revision = revision_name

    return champion_revision, challenger_revision


def roll_back_challenger(revision):
    """Promote the champion model back to 100% traffic."""
    if not revision:
        logging.warning("No Champion revision found. Cannot promote.")
        return
    try:
        update_mlflow_registry_for_demote_challenger()
    except Exception as e:
        print(e)
    update_cloud_run_traffic(revision)


def get_model_run_ids():
    challenger_run_id = ""
    champion_run_id = ""
    TRACKING_URI = "https://mlflow-service-974726646619.us-central1.run.app"
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    try:
        challenger = client.get_model_version_by_alias("xgb_pipeline_taxi_regressor", "challenger")
        challenger_run_id = challenger.run_id
    except Exception as e:
        print(e)
        challenger_run_id = None

    champion = client.get_model_version_by_alias("xgb_pipeline_taxi_regressor", "champion")
    champion_run_id = champion.run_id

    return challenger_run_id, champion_run_id


def update_mlflow_registry_for_demote_challenger():
    TRACKING_URI = "https://mlflow-service-974726646619.us-central1.run.app"
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    challenger = client.get_model_version_by_alias("xgb_pipeline_taxi_regressor", "challenger")
    challenger_version = challenger.version
    client.set_model_version_tag("xgb_pipeline_taxi_regressor", challenger_version, "challenger_status", "demoted")
    # client.set_registered_model_alias("xgb_pipeline_taxi_regressor", "champion", challenger_version)
    client.delete_registered_model_alias("xgb_pipeline_taxi_regressor", "challenger")


def update_mlflow_registry_for_promote_challenger():
    TRACKING_URI = "https://mlflow-service-974726646619.us-central1.run.app"
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    challenger = client.get_model_version_by_alias("xgb_pipeline_taxi_regressor", "challenger")
    challenger_version = challenger.version
    client.set_registered_model_alias("xgb_pipeline_taxi_regressor", "champion", challenger_version)
    client.delete_registered_model_alias("xgb_pipeline_taxi_regressor", "challenger")


def promote_challenger():
    """Redeploy Cloud Run service with a new revision and direct 100% traffic to it."""
    client = run_v2.ServicesClient()
    service_path = client.service_path(PROJECT_ID, REGION, SERVICE_NAME)
    new_env_var_key = "MODEL_TYPE"
    new_env_var_value = "champion"

    try:
        update_mlflow_registry_for_promote_challenger()
    except Exception as e:
        logging.warning(f"Failed to change model in MLflow Model Registry.")

    # Get existing service configuration
    try:
        service = client.get_service(name=service_path)
    except Exception as e:
        logging.warning(f"Failed to fetch Cloud Run service: {e}")
        return

    # Modify environment variables
    env_variables = {env.name: env.value for env in service.template.containers[0].env}

    # Add or update the environment variable
    env_variables[new_env_var_key] = new_env_var_value

    # Apply the updated environment variables
    service.template.containers[0].env = [
        run_v2.EnvVar(name=key, value=value) for key, value in env_variables.items()
    ]

    # Deploy the updated service (this creates a new revision)
    try:
        operation = client.update_service(service=service)
        updated_service = operation.result()  # Wait for deployment to complete

        # Get the latest revision name
        latest_revision_path = updated_service.latest_ready_revision
        if not latest_revision_path:
            logging.warning("Failed to fetch latest revision.")
            return

        logging.info(f"New revision {latest_revision_path} deployed.")
        latest_revision = latest_revision_path.split("/")[-1]  # Extract just the revision name

        # Now, shift 100% traffic to the new revision
        updated_service.traffic = [
            run_v2.
            TrafficTarget(percent=100,
                          revision=latest_revision,
                          type_=TrafficTargetAllocationType(2))
        ]

        # Apply the traffic update
        operation = client.update_service(service=updated_service)
        operation.result()  # Wait for the update to complete

        logging.info(f"Successfully shifted 100% traffic to {latest_revision}")

    except Exception as e:
        logging.warning(f"Failed to redeploy Cloud Run: {e}")


def compare_performance_analysis(challenger_performance_metrics, champion_performance_metrics):
    """Compare model performance and update Cloud Run traffic if necessary."""

    logging.info("Fetching Cloud Run revisions...")
    champion, challenger = get_cloud_run_revisions()

    if not champion or not challenger:
        print("Either Challenger or Champion revision is missing. Aborting traffic update.")
        return

    try:
        challenger_rmse = challenger_performance_metrics["RMSE_standardized_by_mean"]
        champion_rmse = champion_performance_metrics["RMSE_standardized_by_mean"]
        challenger_r2 = challenger_performance_metrics["R2"]
        champion_r2 = champion_performance_metrics["R2"]
    except KeyError as e:
        logging.info(f"Missing performance metric: {e}. Cannot proceed with comparison.")
        return

    logging.info(f"Model Performance Metrics:")
    logging.info(f"   - Challenger RMSE: {challenger_rmse:.4f}, R²: {challenger_r2:.4f}")
    logging.info(f"   - Champion RMSE: {champion_rmse:.4f}, R²: {champion_r2:.4f}")

    # Case 1: Both models underperform -> Trigger manual retraining/response
    if (challenger_r2 < 0.7 and champion_r2 < 0.7) or (challenger_rmse > 0.3 and champion_rmse > 0.3):
        logging.warning("ALERT: Both models are underperforming. Sending notification for manual intervention.")
        send_email_alert(
            "Both Deployed Models Are Underperforming!",
            f"""
            <h3>Urgent: Both Models Are Failing</h3>
            <p>Both the Champion and Challenger models are performing below acceptable thresholds.</p>
            <p><strong>Critical Performance Metrics:</strong></p>
            <ul>
                <li><strong>Challenger RMSE:</strong> {challenger_rmse:.4f}</li>
                <li><strong>Champion RMSE:</strong> {champion_rmse:.4f}</li>
                <li><strong>Challenger R²:</strong> {challenger_r2:.4f}</li>
                <li><strong>Champion R²:</strong> {champion_r2:.4f}</li>
            </ul>
            <p><strong>Immediate Action Required:</strong></p>
            <ul>
                <li>Check data pipeline for issues.</li>
                <li>Validate model drift.</li>
                <li>Trigger model retraining.</li>
            </ul>
            """
        )
        return

    # Case 2: Challenger Performs Worse -> Rollback to Champion
    if challenger_rmse > champion_rmse and challenger_r2 < champion_r2:
        logging.info(f"Challenger ({challenger}) underperforms compared to Champion ({champion}). Initiating rollback...")
        roll_back_challenger(champion)
        logging.info(f"Rollback to Champion ({champion}) completed.")
        return

    # Case 3: Challenger Performs Better -> Promote Challenger
    if challenger_rmse < champion_rmse and challenger_r2 > champion_r2:
        logging.info(f"Challenger ({challenger}) outperforms Champion ({champion}). Promoting to 100% traffic...")
        promote_challenger()
        logging.info("Challenger successfully promoted.")
        return

    # Case 4: Performance is Similar -> No Immediate Change
    logging.info("INFO: Challenger model performs similarly to Champion. No update needed.")
    send_email_alert(
        "Challenger Performance Similar to Champion",
        f"""
        <h3>Challenger Model Has Similar Performance</h3>
        <p>The challenger model has performance close to the champion model. No clear winner.</p>
        <p><strong>Metrics Comparison:</strong></p>
        <ul>
            <li><strong>Challenger RMSE:</strong> {challenger_rmse:.4f}</li>
            <li><strong>Champion RMSE:</strong> {champion_rmse:.4f}</li>
            <li><strong>Challenger R²:</strong> {challenger_r2:.4f}</li>
            <li><strong>Champion R²:</strong> {champion_r2:.4f}</li>
        </ul>
        <p>Consider additional evaluation before deciding on deployment.</p>
        """
    )


def update_cloud_run_traffic(revision):
    """Update Cloud Run to direct 100% of traffic to the given revision."""
    client = run_v2.ServicesClient()
    service_path = client.service_path(PROJECT_ID, REGION, SERVICE_NAME)

    # Get existing service configuration
    try:
        service = client.get_service(name=service_path)
    except Exception as e:
        logging.warning(f"Failed to fetch Cloud Run service: {e}")
        return

    if not service.traffic:
        logging.warning("No existing traffic settings found. Cannot update traffic.")
        return

    # Update traffic allocation to a specific revision
    service.traffic = [
        run_v2.TrafficTarget(
            type_=TrafficTargetAllocationType(2),  # TRAFFIC_TARGET_ALLOCATION_TYPE_REVISION
            revision=revision,  # the revision target
            percent=100,
        )
    ]

    # Deploy update
    try:
        operation = client.update_service(service=service)
        operation.result()  # Wait for operation to complete
        logging.info(f"Successfully shifted 100% traffic to {revision}")
    except Exception as e:
        logging.warning(f"Failed to update traffic: {e}")

    # redeploy challenger revision with new environment variable


def data_drift_detected(data_drift_report):
    try:
        return data_drift_report["metrics"][0]["result"]["dataset_drift"] is True
    except (KeyError, IndexError, TypeError):
        return False  # Default to False if the structure is incorrect


def target_drift_detected(target_drift_report):
    try:
        return target_drift_report["metrics"][0]["result"]["drift_detected"] is True
    except (KeyError, IndexError, TypeError):
        return False  # Default to False if the structure is incorrect


def check_drift(report_target, report_data, model_type):
    alerts = []
    if target_drift_detected(report_target):
        alerts.append("Target drift detected.")
    if data_drift_detected(report_data):
        alerts.append("Data drift detected.")

    if alerts:
        send_email_alert(
            f"Drift Detected in {model_type} Model",
            f"""
            <h3>Detected Drift in deployed {model_type.lower()} model during automatic monitoring.</h3>
            <p>{' '.join(alerts)}</p>
            """
        )


def send_email_alert(subject, message):
    SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")  # Fetch from environment variables
    ALERT_EMAIL = os.getenv("ALERT_EMAIL",
                            "vic3super@hotmail.de")  # Let us assume this would be a shared business account
    """Send an alert email using SendGrid."""
    if not SENDGRID_API_KEY:
        print("SendGrid API key not configured. Skipping email alert.")
        return

    mail = Mail(
        from_email="vincent.donat@ovgu.de",
        to_emails=ALERT_EMAIL,
        subject=subject,
        html_content=f"<p>{message}</p>"
    )

    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(mail)
        logging.info(f"Email sent with status code {response.status_code}")
    except Exception as e:
        logging.warning(f"Error sending email: {e}")


def cloud_function_entry_point(request):
    """Cloud Function entry point that only processes POST requests."""

    if request.method != 'POST':
        return 'Only POST requests are allowed', 405

    return process_event()


def save_to_bigquery(data_drift_report_champion, data_drift_report_challenger, target_drift_champion,
                     target_drift_challenger, performance_metrics_champion, performance_metrics_challenger, retrain):
    """Save model performance and drift reports to BigQuery."""
    client = initialize_bigquery_client()
    TABLE_NAME = "monitoring_job"
    table_id = f"{PROJECT_ID}.{DATASET_NAME}.{TABLE_NAME}"

    # Create the row to insert
    message_data = {
        "timestamp": datetime.utcnow().isoformat(),  # Current timestamp
        "data_drift_report_champion": json.dumps(data_drift_report_champion, indent=2, cls=NumpyTypeEncoder),
        "data_drift_report_challenger": json.dumps(data_drift_report_challenger, indent=2, cls=NumpyTypeEncoder),
        "target_drift_champion": json.dumps(target_drift_champion, indent=2, cls=NumpyTypeEncoder),
        "target_drift_challenger": json.dumps(target_drift_challenger, indent=2, cls=NumpyTypeEncoder),
        "performance_metrics_champion": json.dumps(performance_metrics_champion, indent=2, cls=NumpyTypeEncoder),
        "performance_metrics_challenger": json.dumps(performance_metrics_challenger, indent=2, cls=NumpyTypeEncoder),
        "retrain": retrain,
    }

    # Convert data to a list of rows (BigQuery expects an iterable)
    rows_to_insert = [message_data]

    try:
        # Insert the row into BigQuery
        errors = client.insert_rows_json(table_id, rows_to_insert)

        if errors:
            logging.warning(f"Failed to insert rows: {errors}")
        else:
            logging.info(f"Successfully inserted data into {table_id}")

    except Exception as e:
        logging.warning(f"❌ Error inserting data into BigQuery: {e}")


def publish(data_drift_report_champion, target_drift_champion,
            performance_metrics_champion):
    # publishing logic
    # if bad metrics, send pub/sub message to trigger new build
    # if drift, send email alert
    # else, store data simply in bigquery
    check_drift(data_drift_report_champion, target_drift_champion, "Champion")

    if performance_metrics_champion["RMSE_standardized_by_mean"] > 0.3 or performance_metrics_champion["R2"] <= 0.8:
        message_data = {
            "timestamp": datetime.utcnow().isoformat(),  # Current timestamp
            "data_drift_report_champion": json.dumps(data_drift_report_champion, indent=2, cls=NumpyTypeEncoder),
            "data_drift_report_challenger": json.dumps({}),
            "target_drift_champion": json.dumps(target_drift_champion, indent=2, cls=NumpyTypeEncoder),
            "target_drift_challenger": json.dumps({}),
            "performance_metrics_champion": json.dumps(performance_metrics_champion, indent=2, cls=NumpyTypeEncoder),
            "performance_metrics_challenger": json.dumps({}),
            "retrain": "True",
        }

        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path(PROJECT_ID, "monitoring_retrain")

        message_bytes = json.dumps(message_data).encode("utf-8")
        future = publisher.publish(topic_path, message_bytes)
        future.result()
        return

    save_to_bigquery(data_drift_report_champion, {}, target_drift_champion,
                     {}, performance_metrics_champion, {}, "False")
    return



def process_event():
    logging.info("Starting process_event execution")

    challenger_run_id, champion_run_id = get_model_run_ids()
    logging.info(f"Retrieved model run IDs - Challenger: {challenger_run_id}, Champion: {champion_run_id}")

    client = initialize_bigquery_client()
    logging.info("Initialized BigQuery client")

    new_data = get_new_data(client, PROJECT_ID, DATASET_NAME)
    logging.info("Fetched new data for monitoring")

    if new_data is None or new_data.empty:
        logging.info("No new data available for monitoring")
        return "No new data to monitor", 200

    new_data.dropna(subset="ground_truth", inplace=True)
    logging.info(f"New data after dropping missing ground_truth values: {len(new_data)} rows")

    if challenger_run_id is None:
        logging.info("No challenger model found. Running champion-only analysis")

        training_data = get_training_data(client, PROJECT_ID, DATASET_NAME, champion_run_id)
        logging.info(f"Fetched champion training data: {len(training_data)} rows")

        new_data = new_data.dropna(subset=["ground_truth"])
        features_training_data = training_data.drop(columns="trip_total")
        features_new_data = extract_features(new_data)
        logging.info("Extracted features from training and new data")

        target_training_data = extract_target(training_data)
        target_new_data = extract_target(new_data)
        logging.info("Extracted targets from training and new data")

        champion_data = new_data[new_data["model_type"] == "champion"]
        logging.info(f"Filtered champion data: {len(champion_data)} rows")

        performance_metrics_champion = run_performance_analysis(champion_data)
        logging.info("Ran performance analysis for champion model")

        data_drift_report_champion = run_data_drift_analysis(features_training_data, features_new_data)
        logging.info("Ran data drift analysis for champion model")

        target_drift_champion = run_target_drift_analysis(target_training_data, target_new_data)
        logging.info("Ran target drift analysis for champion model")

        publish(data_drift_report_champion, target_drift_champion, performance_metrics_champion)
        logging.info("Published monitoring results for champion model")

    else:
        logging.info("Challenger model detected. Running full comparison analysis")

        training_data_challenger = get_training_data(client, PROJECT_ID, DATASET_NAME, challenger_run_id)
        training_data_champion = get_training_data(client, PROJECT_ID, DATASET_NAME, champion_run_id)
        logging.info("Fetched training data for both challenger and champion models")

        features_training_data_challenger = training_data_challenger.drop(columns="trip_total")
        features_training_data_champion = training_data_champion.drop(columns="trip_total")
        features_new_data = extract_features(new_data)
        logging.info("Extracted features for challenger, champion, and new data")

        target_training_data_challenger = extract_target(training_data_challenger)
        target_training_data_champion = extract_target(training_data_champion)
        target_new_data = extract_target(new_data)
        logging.info("Extracted targets for challenger, champion, and new data")

        champion_data = new_data[new_data["model_type"] == "champion"]
        challenger_data = new_data[new_data["model_type"] == "challenger"]
        logging.info(f"Champion data rows: {len(champion_data)}, Challenger data rows: {len(challenger_data)}")

        champion_performance_metrics = run_performance_analysis(champion_data)
        challenger_performance_metrics = run_performance_analysis(challenger_data)
        logging.info("Completed performance analysis for both models")

        data_drift_report_champion = run_data_drift_analysis(features_training_data_champion, features_new_data)
        data_drift_report_challenger = run_data_drift_analysis(features_training_data_challenger, features_new_data)
        logging.info("Completed data drift analysis for both models")

        target_drift_champion = run_target_drift_analysis(target_training_data_champion, target_new_data)
        target_drift_challenger = run_target_drift_analysis(target_training_data_challenger, target_new_data)
        logging.info("Completed target drift analysis for both models")

        check_drift(target_drift_champion, data_drift_report_champion, "Champion")
        check_drift(target_drift_challenger, data_drift_report_challenger, "Challenger")
        logging.info("Checked drift conditions for both models")

        compare_performance_analysis(challenger_performance_metrics, champion_performance_metrics)
        logging.info("Compared performance metrics between challenger and champion")

        save_to_bigquery(
            data_drift_report_champion,
            data_drift_report_challenger,
            target_drift_champion,
            target_drift_challenger,
            champion_performance_metrics,
            challenger_performance_metrics,
            "False"
        )
        logging.info("Saved monitoring results to BigQuery")

    logging.info("process_event execution completed successfully")
    return 'Triggered successfully', 200

