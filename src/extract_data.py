import json
import logging
from google.cloud import bigquery
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def extract_data(data):
    """
    Processes a DataFrame by cleaning, removing outliers, and uploading to BigQuery.
    """
    # Ensure input is a DataFrame
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")

    required_columns = {'unique_key', 'taxi_id', 'event_timestamp', 'trip_seconds',
       'trip_miles', 'payment_type', 'trip_total', 'company',
       'trip_start_timestamp', 'extras', 'tolls', 'avg_tips'}

    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    # Drop unnecessary columns
    df = data.drop(columns=["unique_key", "taxi_id", "event_timestamp"], errors="ignore")

    # Remove zero or negative values
    df = df[(df["trip_seconds"] > 0) & (df["trip_miles"] > 0) & (df["trip_total"] > 0)]

    # Drop empty rows
    df.dropna(inplace=True)

    # Extract features from trip_start_timestamp
    df["daytime"] = df["trip_start_timestamp"].dt.hour
    df['day_type'] = df['trip_start_timestamp'].dt.weekday.apply(lambda x: 'weekend' if x >= 5 else 'weekday')
    df['month'] = df['trip_start_timestamp'].dt.month
    df['day_of_week'] = df['trip_start_timestamp'].dt.dayofweek
    df['day_of_month'] = df['trip_start_timestamp'].dt.day
    df.drop(columns=["trip_start_timestamp"], inplace=True)

    if df.empty:
        raise ValueError("DataFrame is empty after cleaning. No valid data to process.")

    # Remove outliers using IQR
    def remove_outliers(df, cols):
        for col in cols:
            if df[col].dtype not in ["int64", "float64"]:
                logging.warning(f"Skipping non-numeric column: {col}")
                continue  # Skip non-numeric columns

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        return df

    # Identify numeric columns except "trip_total"
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    # cols_to_clean = [col for col in numeric_cols if col != "trip_total"]

    # Apply outlier removal
    df = remove_outliers(df, numeric_cols)

    if df.empty:
        raise ValueError("All data was removed after outlier cleaning. No valid data to upload.")

    #upload_training_data_to_bigquery(df)
    return df




def upload_training_data_to_bigquery(cleaned_df):
    """
    Uploads cleaned DataFrame to BigQuery.
    """

    if cleaned_df is None or cleaned_df.empty:
        raise ValueError("Cannot upload empty DataFrame to BigQuery.")

    project_id = "carbon-relic-439014-t0"
    dataset_name = "chicago_taxi"
    table_name = "training_data"
    table_id = f"{project_id}.{dataset_name}.{table_name}"

    client = bigquery.Client()

    # Configure job to replace table contents
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE  # Deletes existing data
    )

    # Upload DataFrame to BigQuery
    try:
        job = client.load_table_from_dataframe(cleaned_df, table_id, job_config=job_config)
        job.result()  # Wait for the job to complete
    except Exception as e:
        raise RuntimeError(f"Failed to load data into BigQuery table. Error: {e}")

    logging.info(f"Successfully uploaded {len(cleaned_df)} rows to BigQuery: {table_id}")




