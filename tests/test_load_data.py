import os
import time
import pytest
from unittest.mock import patch, MagicMock
from src.load_data import load_data_from_feature_store
import pandas as pd


# Force authentication in tests
GCP_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

"""if not GCP_CREDENTIALS or not os.path.exists(GCP_CREDENTIALS):
    raise RuntimeError(f"Missing Google Cloud credentials: {GCP_CREDENTIALS}")
"""


@patch("src.load_data.FeatureStore")
def test_load_data_from_feature_store_success(mock_feature_store):
    """Test that the function successfully returns a DataFrame."""

    # Mock FeatureStore instance
    mock_store = mock_feature_store.return_value

    # Mock feature service
    mock_store.get_feature_service.return_value = MagicMock()

    # Mock data source query
    mock_store.get_data_source.return_value.get_table_query_string.return_value = "mock_table"

    # Mock historical features return value
    sample_df = pd.DataFrame({
        "unique_key": [1, 2, 3],
        "taxi_id": [101, 102, 103],
        "event_timestamp": ["2023-01-01", "2023-01-02", "2023-01-03"],
    })
    mock_store.get_historical_features.return_value.to_df.return_value = sample_df

    # Run function
    df = load_data_from_feature_store(size=3)

    # Assertions
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_invalid_size():
    """Test that function raises ValueError for invalid size inputs."""

    with pytest.raises(ValueError, match="Size must be a positive integer"):
        load_data_from_feature_store(size=-1)  # Negative size

    with pytest.raises(ValueError, match="Size must be a positive integer"):
        load_data_from_feature_store(size="invalid")  # Non-integer size

    with pytest.raises(ValueError, match="Size must be a positive integer"):
        load_data_from_feature_store(size=None)  # NoneType size




@patch("src.load_data.FeatureStore")
def test_feature_service_not_found(mock_feature_store):
    """Test that function raises an error when feature service is missing."""

    # Ensure FeatureStore(repo_path=".") returns a mock instance
    mock_store_instance = mock_feature_store.return_value
    mock_store_instance.get_feature_service.side_effect = RuntimeError("Feature service not found")

    with pytest.raises(RuntimeError, match="Feature service 'taxi_drive' not found"):
        load_data_from_feature_store(size=10)

@patch("src.load_data.FeatureStore")
def test_data_source_not_found(mock_feature_store):
    """Test that function raises an error when data source is missing."""

    mock_store = mock_feature_store.return_value
    mock_store.get_data_source.side_effect = Exception("Data source not found")

    with pytest.raises(RuntimeError, match="Data source 'trip_source' not found or unavailable"):
        load_data_from_feature_store(size=10)


@patch("src.load_data.FeatureStore")
def test_empty_data_returned(mock_feature_store):
    """Test that function raises an error when an empty dataset is retrieved."""

    mock_store = mock_feature_store.return_value
    mock_store.get_historical_features.return_value.to_df.return_value = pd.DataFrame()

    with pytest.raises(RuntimeError, match="Retrieved an empty dataset from Feature Store"):
        load_data_from_feature_store(size=10)



@pytest.mark.integration
def test_load_data_from_feature_store_integration():
    """Integration test: Ensure Feature Store query works correctly."""

    df = load_data_from_feature_store(size=10)

    assert df is not None
    assert not df.empty
    assert "unique_key" in df.columns
    assert isinstance(df, pd.DataFrame)

@pytest.mark.integration
def test_load_data_from_feature_store_execution_time():
    """Integration test: Ensure function runs within 120 seconds"""

    start_time = time.time()
    df = load_data_from_feature_store(size=1000)
    end_time = time.time()

    assert (end_time - start_time) <= 120, f"Execution took too long: {end_time - start_time:.2f} seconds"



@pytest.mark.integration
def test_load_large_dataset():
    """Integration test: Ensure function can handle large datasets without crashing and test domain specific aspects"""

    df = load_data_from_feature_store(size=50000)  # Large request

    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    # Extract hour from timestamps
    df["hour"] = pd.to_datetime(df["trip_start_timestamp"]).dt.hour

    # Check missing hours
    missing_hours = set(range(24)) - set(df["hour"].unique()), f"Missing hours detected"
    # Majority of hours should be covered
    assert len(missing_hours) < 6

    # Find duplicate trips based on unique identifier
    duplicate_trips = df.duplicated(subset=["unique_key"], keep=False)

    # Assert that no duplicate trips exist
    assert duplicate_trips.sum() == 0, f"Duplicate trips found: {duplicate_trips.sum()}"

