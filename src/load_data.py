# Load data
import json

import pandas as pd
from google.cloud import bigquery

def load_data(local=True):
    if local:
        data = pd.read_csv("/app/taxi_trips.csv", nrows=10000)
        return data
    else:
        # Open and load the JSON file

       # with open("../configs/config.json", "r") as file:
       #     config = json.load(file)

        with open("/app/configs/config.json", "r") as file:
            config = json.load(file)

        # Access specific values
        PROJECT_ID = config["PROJECT_ID"]
        DATASET_NAME = config["DATASET_NAME"]

        table_id = f"{PROJECT_ID}.{DATASET_NAME}.raw_data"


        # Initialize the BigQuery client
        client = bigquery.Client()

        # Specify your table ID
        # Format: `project_id.dataset_id.table_id`
        table_id = f"{PROJECT_ID}.{DATASET_NAME}.raw_data"

        # Create a BigQuery query job
        query = f"SELECT * FROM `{table_id}`"

        # Run the query and download the results as a pandas DataFrame
        query_job = client.query(query)  # API request
        results = query_job.result().to_dataframe()

        return results

