import os

from feast import FeatureView, FeatureStore, RepoConfig


def load_data_from_feature_store(size = 1000):
    """
    repo_config = RepoConfig(
    registry=RegistryConfig(path="gs://feast_registry/registry.db"),
    project="ruling_buzzard",
    provider="gcp",
    offline_store=BigQueryOfflineStoreConfig(type="bigquery", dataset="chicago_taxi"),
    )
    """

    if not isinstance(size, int) or size < 1:
        raise ValueError("Size must be a positive integer greater than or equal to 1")

    try:
        repo_path = os.getenv("FEATURE_STORE_PATH", "src/")
        repo_path = os.path.abspath(repo_path)  # Ensure it's an absolute path
        store = FeatureStore(repo_path=repo_path)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize FeatureStore: {e}")

    try:
        feature_service = store.get_feature_service("taxi_drive")
    except Exception as e:
        raise RuntimeError(f"Feature service 'taxi_drive' not found: {e}")

    try:
        data_table = store.get_data_source("trip_source").get_table_query_string()
    except Exception as e:
        raise RuntimeError(f"Data source 'trip_source' not found or unavailable: {e}")

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

    try:
        training_df = store.get_historical_features(
            entity_df=entity_sql,
            features=feature_service,
        ).to_df()

        if training_df.empty:
            raise RuntimeError("Retrieved an empty dataset from Feature Store")

    except Exception as e:
        raise RuntimeError(f"Failed to retrieve features from Feature Store: {e}")

    return training_df

