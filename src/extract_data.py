import json

from google.cloud import bigquery


def extract_data(data):
    # Select only relevant columns
    #df = data[["trip_seconds", "trip_miles", "company", "payment_type", "trip_total", "avg_tips"]]

    df = data.drop(columns=["unique_key", "taxi_id"])
    # remove 0 entries
    df = df[(df["trip_seconds"] > 0) & (df["trip_miles"] > 0) & (df["trip_total"] > 0)]

    # drop empty rows
    df.dropna(inplace=True)

    # Remove outliers using IQR
    def remove_outliers(df, cols):
        for col in cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        return df

    # Identify all numeric columns except "trip_total"
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cols_to_clean = [col for col in numeric_cols if col != "trip_total"]

    # Apply outlier removal to selected numeric columns
    df = remove_outliers(df, numeric_cols)

    upload_training_data_to_bigquery(df)
    return df


def upload_training_data_to_bigquery(cleaned_df):

    project_id = "carbon-relic-439014-t0"
    dataset_name = "chicago_taxi"
    table_name = "training_data"
    table_id = f"{project_id}.{dataset_name}.{table_name}"

    client = bigquery.Client()

    # Configure job to replace table contents (truncate before inserting)
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE  # Deletes existing data
    )

    # Upload DataFrame to BigQuery
    job = client.load_table_from_dataframe(cleaned_df, table_id, job_config=job_config)
    job.result()  # Wait for the job to complete


