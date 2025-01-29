# Load data

"""
import json

import pandas as pd
from google.cloud import bigquery

def load_data(local=True, size=1000):
    if local:
        data = pd.read_csv("../taxi_trips.csv", nrows=10000)
        return data
    else:
        # Open and load the JSON file
        with open("/app/configs/config.json", "r") as file:
            config = json.load(file)

        # Access specific values
        PROJECT_ID = config["PROJECT_ID"]
        DATASET_NAME = config["DATASET_NAME"]

        # Initialize the BigQuery client
        client = bigquery.Client()

        # Format: `project_id.dataset_id.table_id`
        table_id_old = f"{PROJECT_ID}.{DATASET_NAME}.raw_data"

        # Create a BigQuery query job
        query = f"SELECT * FROM `{table_id_old}`"

        # Run the query and download the results as a pandas DataFrame
        query_job = client.query(query)  # API request
        results_old = query_job.result().to_dataframe()

        table_ref = client.dataset(DATASET_NAME).table("prediction")
        table = client.get_table(table_ref)

        columns = [field.name for field in table.schema if field.name not in ["prediction", "timestamp"]]
        query = f"SELECT {', '.join(columns)} FROM `PROJECT_ID.DATASET_NAME.prediction`"
        query_job = client.query(query)
        results_new = query_job.result().to_dataframe()

        frames = [results_old, results_new]
        result = pd.concat(frames)
        result = result.tail(size)
        return result

"""

from google.cloud import bigquery
import pandas as pd
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)


def execute_bigquery_query(client, query):
    """Executes a BigQuery query and returns the results as a DataFrame."""
    try:
        query_job = client.query(query)  # API request
        return query_job.result().to_dataframe()
    except Exception as e:
        logging.error(f"Error executing query: {query}")
        raise e


def load_data_test(local=True, size=1000, use_new_data=True):
    """
    Load taxi trip data from a local CSV or BigQuery.

    Args:
        local (bool): If True, load data from a local CSV file; otherwise, fetch from BigQuery.
        size (int): Number of rows to return from the concatenated DataFrame.

    Returns:
        df: DataFrame from BigQuery.
        :param use_new_data: If False, only return old training data, excluding new input data from predictions
    """
    if local:
        try:
            data = pd.read_csv("../taxi_trips.csv", nrows=10000)
            return data
        except FileNotFoundError:
            raise FileNotFoundError("Local CSV file not found at '../taxi_trips.csv'")

    else:
        try:
            # Load config
            config_path = Path("../configs/config.json")
            if not config_path.is_file():
                raise FileNotFoundError("Configuration file not found.")

            with open(config_path, "r") as file:
                config = json.load(file)

            PROJECT_ID = config["PROJECT_ID"]
            DATASET_NAME = config["DATASET_NAME"]

            # Initialize BigQuery client
            client = bigquery.Client()

            # Query old data
            table_id_old = f"{PROJECT_ID}.{DATASET_NAME}.raw_data"
            query_old = f"SELECT * FROM `{table_id_old}`"
            result_old = execute_bigquery_query(client, query_old)

            # Only return old training data
            if not use_new_data:
                return result_old.tail(size)

            # Query new data
            table_ref = client.dataset(DATASET_NAME).table("prediction")
            table = client.get_table(table_ref)
            columns = [field.name for field in table.schema if field.name not in ["prediction", "timestamp"]]
            query_new = f"SELECT {', '.join([f'`{column}`' for column in columns])} FROM `{PROJECT_ID}.{DATASET_NAME}.prediction`"
            results_new = execute_bigquery_query(client, query_new)

            # Concatenate and limit size
            frames = [result_old, results_new]
            result = pd.concat(frames).tail(size)
            return result

        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise e