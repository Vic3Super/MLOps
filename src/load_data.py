import os
import logging
from feast import FeatureView, FeatureStore, RepoConfig

# Setup logger
logger = logging.getLogger(__name__)

def load_data_from_feature_store(size=100000):
    """
    Loads data from Feast Feature Store.

    Args:
        size (int): Number of records to retrieve.

    Returns:
        pd.DataFrame: Retrieved feature data.

    Raises:
        ValueError: If `size` is not a positive integer.
        RuntimeError: If any step in loading data fails.
    """
    if not isinstance(size, int) or size < 1:
        logger.error("Size must be a positive integer greater than or equal to 1.")
        raise ValueError("Size must be a positive integer greater than or equal to 1")

    try:
        #repo_path = os.getenv("FEATURE_STORE_PATH", "src/")
        repo_path = os.getenv("FEATURE_STORE_PATH", "src/")

        repo_path = os.path.abspath(repo_path)  # Ensure it's an absolute path
        logger.info(f"Using Feature Store path: {repo_path}")

        store = FeatureStore(repo_path=repo_path)
        logger.info("FeatureStore initialized successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize FeatureStore: {e}", exc_info=True)
        raise RuntimeError(f"Failed to initialize FeatureStore: {e}")

    try:
        feature_service = store.get_feature_service("taxi_drive")
        logger.info("Feature service 'taxi_drive' retrieved successfully.")
    except Exception as e:
        logger.error(f"Feature service 'taxi_drive' not found: {e}")
        raise RuntimeError(f"Feature service 'taxi_drive' not found: {e}")

    try:
        data_table = store.get_data_source("trip_source").get_table_query_string()
        logger.info("Data source 'trip_source' retrieved successfully.")
    except Exception as e:
        logger.error(f"Data source 'trip_source' not found or unavailable: {e}")
        raise RuntimeError(f"Data source 'trip_source' not found or unavailable: {e}")

    # Construct SQL query to fetch the latest feature values for unique entities
    entity_sql = f"""
        SELECT
            unique_key,
            timestamp AS event_timestamp
        FROM {data_table} 
        ORDER BY timestamp DESC
        LIMIT {size}
    """
    logger.info(f"Generated SQL query for entity selection: {entity_sql}")

    try:
        training_df = store.get_historical_features(
            entity_df=entity_sql,
            features=feature_service,
        ).to_df()

        if training_df.empty:
            logger.warning("Retrieved an empty dataset from Feature Store.")
            raise RuntimeError("Retrieved an empty dataset from Feature Store.")

        logger.info(f"Successfully retrieved {len(training_df)} records from Feature Store.")
    except Exception as e:
        logger.error(f"Failed to retrieve features from Feature Store: {e}", exc_info=True)
        raise RuntimeError(f"Failed to retrieve features from Feature Store: {e}")

    return training_df