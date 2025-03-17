import pytest
import pandas as pd
from unittest.mock import patch
from src.extract_data import extract_data, upload_training_data_to_bigquery


### ✅ Test: Successful Data Cleaning ###
def test_extract_data_success():
    """Test extract_data() with valid data and ensure it cleans correctly."""
    raw_data = pd.DataFrame({
        "unique_key": [1, 2, 3, None],
        "taxi_id": [101, 102, 103, None],
        "event_timestamp": ["2025-02-02 19:35:16.886464+00:00", "2025-02-02 19:35:16.886464+00:00",
                            "2025-02-02 19:35:16.886464+00:00", None],
        "trip_miles": [5, 8, 12, None],
        "trip_total": [15.0, 25.0, 40.0, None],
        "company": ["CompanyA", "CompanyB", "CompanyC", None],
        "payment_type": ["Cash", "Card", "Card", None],
        "trip_start_timestamp":["2023-07-06 16:45:00+00:00", "2023-09-18 18:15:00+00:00", "2023-08-22 17:00:00+00:00", None],
        "extras":[0, 1, 2, None],
        "tolls":[3,4,5, None],
        "tips":[0.5,0.3,1, None],
        "pickup_community_area": [12, 14, 17, None],
        "pickup_latitude": [13.8, 14.56, 17.98, None],
        "pickup_longitude": [13.8, 14.56, 17.98, None],
    })
    raw_data["event_timestamp"] = pd.to_datetime(raw_data["event_timestamp"])
    raw_data["trip_start_timestamp"] = pd.to_datetime(raw_data["trip_start_timestamp"])
    cleaned_df = extract_data(raw_data)

    assert cleaned_df is not None
    assert len(cleaned_df.columns) > len(raw_data.columns)
    assert "unique_key" not in cleaned_df.columns  # Dropped column
    assert "taxi_id" not in cleaned_df.columns  # Dropped column
    assert "event_timestamp" not in cleaned_df.columns  # Dropped column
    assert all(cleaned_df["trip_miles"] > 0)
    assert all(cleaned_df["trip_total"] > 0)
    assert not cleaned_df.empty


### ✅ Test: Handles Missing Required Columns ###
def test_extract_data_missing_columns():
    """Test extract_data() raises KeyError if required columns are missing."""
    incomplete_data = pd.DataFrame({
        "trip_miles": [5, 8, 12],
        "trip_total": [15.0, 25.0, 40.0]
    })

    with pytest.raises(KeyError, match="Missing required columns"):
        extract_data(incomplete_data)


### ✅ Test: Handles Non-DataFrame Input ###
def test_extract_data_invalid_input():
    """Test extract_data() raises TypeError if input is not a DataFrame."""
    with pytest.raises(TypeError, match="Input data must be a pandas DataFrame"):
        extract_data("invalid input")  # Passing string instead of DataFrame


### ✅ Test: Handles Empty DataFrame After Cleaning ###
def test_extract_data_empty_after_cleaning():
    """Test extract_data() raises ValueError if all data is removed."""
    raw_data = pd.DataFrame({
        "unique_key": [1, 2, 3],
        "taxi_id": [101, 102, 103],
        "event_timestamp": ["2025-02-02 19:35:16.886464+00:00", "2025-02-02 19:35:16.886464+00:00",
                            "2025-02-02 19:35:16.886464+00:00"],
        "trip_miles": [0, 0, 0 ],
        "trip_total": [15.0, 25.0, 40.0],
        "company": ["CompanyA", "CompanyB", "CompanyC"],
        "payment_type": ["Cash", "Card", "Card"],
        "trip_start_timestamp":["2023-07-06 16:45:00+00:00", "2023-09-18 18:15:00+00:00", "2023-08-22 17:00:00+00:00"],
        "extras":[0, 1, 2],
        "tolls":[3,4,5],
        "tips": [0.5, 0.3, 1],
        "pickup_community_area": [12, 14, 17],
        "pickup_latitude": [13.8, 14.56, 17.98],
        "pickup_longitude": [13.8, 14.56, 17.98],
    })
    raw_data["event_timestamp"] = pd.to_datetime(raw_data["event_timestamp"])
    raw_data["trip_start_timestamp"] = pd.to_datetime(raw_data["trip_start_timestamp"])

    with pytest.raises(ValueError, match="DataFrame is empty after cleaning"):
        extract_data(raw_data)


### ✅ Test: Handles Outlier Removal ###
def test_extract_data_outliers():
    """Test that outlier removal does not remove all data."""
    raw_data = pd.DataFrame({
        "unique_key": list(range(4, 10)),
        "taxi_id": [104, 105, 106, 107, 108, 109],
        "event_timestamp": ["2025-02-02 19:35:16.886464+00:00"] * 6,
        "trip_miles": [6, 7, 9, 10, 11, 13],
        "trip_total": [18.0, 22.5, 28.0, 35.0, 38.5, 1000.0],  # 1000 is an outlier
        "company": ["CompanyD", "CompanyE", "CompanyF", "CompanyG", "CompanyH", "CompanyI"],
        "payment_type": ["Cash", "Card", "Cash", "Card", "Cash", "Card"],
        "trip_start_timestamp": [
            "2023-10-05 12:45:00+00:00",
            "2023-11-12 14:30:00+00:00",
            "2023-12-24 16:20:00+00:00",
            "2024-01-15 09:10:00+00:00",
            "2024-02-20 19:50:00+00:00",
            "2024-03-30 21:25:00+00:00"
        ],
        "extras": [0, 1, 2, 1, 0, 2],
        "tolls": [2, 3, 5, 4, 3, 6],
        "tips": [0.5, 0.3, 1, 1, 3, 0],
        "pickup_community_area": [12, 14, 17, 12, 15, 18],
        "pickup_latitude": [13.8, 14.56, 17.98, 12.8, 15.7, 18.9],
        "pickup_longitude": [13.8, 14.56, 17.98, 12.8, 15.7, 18.9],
    })
    raw_data["event_timestamp"] = pd.to_datetime(raw_data["event_timestamp"])
    raw_data["trip_start_timestamp"] = pd.to_datetime(raw_data["trip_start_timestamp"])
    cleaned_df = extract_data(raw_data)

    assert cleaned_df is not None
    assert cleaned_df.shape[0] < raw_data.shape[0]  # Some rows should be removed


### ✅ Test: DataFrame is Still Valid After Processing ###
def test_extract_data_output_columns():
    #Test extract_data() output has expected columns.
    raw_data = pd.DataFrame({
        "unique_key": [1, 2, 3],
        "taxi_id": [101, 102, 103],
        "event_timestamp": ["2025-02-02 19:35:16.886464+00:00", "2025-02-02 19:35:16.886464+00:00",
                            "2025-02-02 19:35:16.886464+00:00"],
        "trip_miles": [5, 8, 12 ],
        "trip_total": [15.0, 25.0, 40.0],
        "company": ["CompanyA", "CompanyB", "CompanyC"],
        "payment_type": ["Cash", "Card", "Card"],
        "trip_start_timestamp":["2023-07-06 16:45:00+00:00", "2023-09-18 18:15:00+00:00", "2023-08-22 17:00:00+00:00"],
        "extras":[0, 1, 2],
        "tolls":[3,4,5],
        "tips": [0.5, 0.3, 1],
        "pickup_community_area": [12, 14, 17],
        "pickup_latitude": [13.8, 14.56, 17.98],
        "pickup_longitude": [13.8, 14.56, 17.98],
    })
    raw_data["event_timestamp"] = pd.to_datetime(raw_data["event_timestamp"])
    raw_data["trip_start_timestamp"] = pd.to_datetime(raw_data["trip_start_timestamp"])

    cleaned_df = extract_data(raw_data)

    expected_columns = {'trip_miles', 'payment_type',
       'trip_total', 'company', 'extras', 'tolls', 'avg_tips',
       'daytime', 'day_type', 'month', 'day_of_week', 'day_of_month', "pickup_community_area","pickup_latitude", "pickup_longitude" }
    assert set(cleaned_df.columns) == expected_columns


### ✅ Test: Successful Upload ###
@patch("src.extract_data.bigquery.Client")  # Mock BigQuery client
def test_upload_training_data_to_bigquery_success(mock_bigquery_client):
    """Test that data uploads successfully without errors."""
    mock_client_instance = mock_bigquery_client.return_value
    mock_client_instance.load_table_from_dataframe.return_value.result.return_value = None  # Simulate success

    df = pd.DataFrame({
        "trip_seconds": [300, 500],
        "trip_miles": [5, 8],
        "trip_total": [15.0, 25.0]
    })

    try:
        upload_training_data_to_bigquery(df, "123")
    except Exception:
        pytest.fail("upload_training_data_to_bigquery() raised an exception unexpectedly!")


### ✅ Test: Handles Empty DataFrame ###
def test_upload_training_data_to_bigquery_empty_df():
    """Test upload_training_data_to_bigquery raises ValueError if given empty DataFrame."""
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match="Cannot upload an empty DataFrame to BigQuery."):
        upload_training_data_to_bigquery(empty_df, "123")


### ✅ Test: Handles BigQuery Errors Gracefully ###
@patch("src.extract_data.bigquery.Client")
def test_upload_training_data_to_bigquery_bigquery_error(mock_bigquery_client):
    """Test that BigQuery errors are handled properly."""
    mock_client_instance = mock_bigquery_client.return_value
    mock_client_instance.load_table_from_dataframe.side_effect = Exception("BigQuery error")

    df = pd.DataFrame({
        "trip_seconds": [300, 500],
        "trip_miles": [5, 8],
        "trip_total": [15.0, 25.0]
    })

    with pytest.raises(RuntimeError, match="Failed to load data into BigQuery table"):
        upload_training_data_to_bigquery(df, "123")

