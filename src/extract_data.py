import json
import logging
import os
from google.cloud import bigquery
import pandas as pd


logger = logging.getLogger(__name__)


def extract_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Processes a DataFrame by cleaning, removing outliers, and preparing for BigQuery upload.

    Args:
        data (pd.DataFrame): The input data containing taxi trip information.

    Returns:
        pd.DataFrame: The cleaned DataFrame, ready for BigQuery upload.

    Raises:
        TypeError: If `data` is not a pandas DataFrame.
        KeyError: If required columns are missing from the input DataFrame.
        ValueError: If the resulting DataFrame is empty after cleaning or outlier removal.
    """
    logger.info("Starting data extraction and preprocessing.")

    # Ensure input is a DataFrame
    if not isinstance(data, pd.DataFrame):
        logger.error("Invalid input: Data is not a pandas DataFrame.")
        raise TypeError("Input data must be a pandas DataFrame.")

    required_columns = {
        'unique_key', 'taxi_id', 'event_timestamp',
        'trip_miles', 'payment_type', 'trip_total', 'company',
        'trip_start_timestamp', 'extras', 'tolls',
        "pickup_latitude", "pickup_longitude", "pickup_community_area"
    }

    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        raise KeyError(f"Missing required columns: {missing_columns}")

    logger.info("Dropping unnecessary columns and handling missing values.")
    df = data.drop(columns=["taxi_id", "event_timestamp"], errors="ignore")


    size_before_removing_zero_entries = len(df)
    # Remove zero or negative values and trips with improbably price for their distance
    df = df[(df["trip_miles"] > 0) & (df["trip_total"] > 0) & (df["trip_miles"] < df["trip_total"])]
    removed = size_before_removing_zero_entries - len(df)
    logger.info(f"Removed {removed} zero or improbable entries from columns trip_miles, trip_total.")



    if df.empty:
        logger.error("DataFrame is empty after initial cleaning. No valid data to process.")
        raise ValueError("DataFrame is empty after cleaning. No valid data to process.")

    # Convert timestamp column to timezone-naive format
    df['trip_start_timestamp'] = df['trip_start_timestamp'].dt.tz_localize(None).astype('datetime64[ns]')

    # Feature extraction from timestamp
    logger.info("Extracting date and time-based features.")
    df["daytime"] = df["trip_start_timestamp"].dt.hour
    df['day_type'] = df['trip_start_timestamp'].dt.weekday.apply(lambda x: 'weekend' if x >= 5 else 'weekday')
    df['month'] = df['trip_start_timestamp'].dt.month
    df['day_of_week'] = df['trip_start_timestamp'].dt.dayofweek
    df['day_of_month'] = df['trip_start_timestamp'].dt.day
    df.drop(columns=["trip_start_timestamp"], inplace=True)

    df["avg_tips"] = df.groupby("unique_key")["tips"].transform("mean")
    df.drop(columns=["tips", "unique_key"], inplace=True)
    # Outlier removal
    df = remove_outliers(df)


    if df.empty:
        logger.error("All data was removed after outlier cleaning. No valid data to process.")
        raise ValueError("All data was removed after outlier cleaning. No valid data to process.")

    logger.info(f"Data extraction completed successfully. Final dataset size: {len(df)} rows.")

    return df

# Some test
def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes outliers from numeric columns using the Interquartile Range (IQR) method.

    Args:
        df (pd.DataFrame): The DataFrame from which outliers should be removed.

    Returns:
        pd.DataFrame: The DataFrame after outlier removal.

    Notes:
        - Columns that are non-numeric are skipped.
        - Uses a standard 1.5*IQR rule for filtering outliers.
    """
    logger.info("Starting outlier removal process.")
    exclude_cols = {"extras", "avg_tips", "pickup_latitude", "tolls", "pickup_longitude", "pickup_community_area"}  # Set of columns to exclude
    numeric_cols = [col for col in df.select_dtypes(include=["number"]).columns if col not in exclude_cols]
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        original_size = len(df)
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        removed = original_size - len(df)

        if removed > 0:
            logger.info(f"Removed {removed} outliers from column '{col}'.")

    logger.info("Outlier removal process completed.")
    return df


def upload_training_data_to_bigquery(cleaned_df: pd.DataFrame, model_run_id: str):
    """
    Uploads a cleaned DataFrame to a BigQuery table, setting new data to "ACTIVE"
    and marking existing data as "INACTIVE".

    Args:
        cleaned_df (pd.DataFrame): The cleaned DataFrame to upload.
        model_run_id (str): The mlflow run_id of the trained model.
    Raises:
        ValueError: If `cleaned_df` is empty or None.
        RuntimeError: If there is an error during the upload to BigQuery.
    """
    if cleaned_df is None or cleaned_df.empty:
        logger.error("Attempted to upload an empty DataFrame to BigQuery.")
        raise ValueError("Cannot upload an empty DataFrame to BigQuery.")

    project_id = os.getenv("PROJECT_ID", "carbon-relic-439014-t0")
    dataset_name = os.getenv("DATASET_NAME", "chicago_taxi")
    table_name = "training_data"
    table_id = f"{project_id}.{dataset_name}.{table_name}"

    client = bigquery.Client()

    cleaned_df["model_run_id"] = model_run_id

    # Configure job to append new rows
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND  # Append instead of replacing
    )

    logger.info(f"Starting BigQuery upload to table: {table_id}")

    try:
        job = client.load_table_from_dataframe(cleaned_df, table_id, job_config=job_config)
        job.result()  # Wait for the job to complete
        logger.info(f"Successfully uploaded {len(cleaned_df)} rows to BigQuery: {table_id}")
    except Exception as e:
        logger.critical(f"Failed to load data into BigQuery table: {table_id}. Error: {e}", exc_info=True)
        raise RuntimeError(f"Failed to load data into BigQuery table: {e}")