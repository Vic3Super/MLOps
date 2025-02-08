from google.cloud import bigquery
import pandas as pd
import json
import logging
from feast import FeatureView, FeatureStore



def load_data_from_feature_store(size = 1000):

    store = FeatureStore(repo_path="/app/src/feature_store.yaml")
    feature_service = store.get_feature_service("taxi_drive")

    data_table = store.get_data_source("trip_source").get_table_query_string()
    # Get the latest feature values for unique entities
    entity_sql = f"""
        SELECT
            unique_key,
            taxi_id,
            timestamp AS event_timestamp
        FROM {data_table} 
        ORDER BY timestamp DESC
        LIMIT {size}
    """

    training_df = store.get_historical_features(
        entity_df=entity_sql,
        features=feature_service,
    ).to_df()

    return training_df
