import pytest
import pandas as pd
import numpy as np
from google.cloud import bigquery
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor
from src.train import train_pipeline, create_pipeline

@pytest.fixture
def sample_data() -> pd.DataFrame:
    project_id = "carbon-relic-439014-t0"
    dataset_id = "chicago_taxi"
    table_id = "sample_data"
    client = bigquery.Client()

    table_ref = f"{project_id}.{dataset_id}.{table_id}"

    query = f"""
    SELECT * 
    FROM `{table_ref}`
    """

    return client.query(query).to_dataframe()

def test_train_pipeline_valid_input():
    # Sample valid data
    data = pd.DataFrame({
        "trip_seconds": np.random.randint(60, 3600, size=100),
        "trip_miles": np.random.uniform(0.5, 30, size=100),
        "payment_type": np.random.choice(["Credit Card", "Cash"], size=100),
        "trip_total": np.random.uniform(5, 100, size=100),
        "company": np.random.choice(["Company A", "Company B"], size=100),
        "extras": np.random.uniform(0, 10, size=100),
        "tolls": np.random.uniform(0, 10, size=100),
        "avg_tips": np.random.uniform(0, 10, size=100),
        "daytime": np.random.randint(0, 24, size=100),
        "day_type": np.random.choice(["Weekday", "Weekend"], size=100),
        "month": np.random.randint(1, 13, size=100),
        "day_of_week": np.random.randint(0, 7, size=100),
        "day_of_month": np.random.randint(1, 31, size=100),
    })

    pipeline = create_pipeline()

    trained_pipeline, metrics, X_train, y_train, X_test, y_test, params = train_pipeline(pipeline, data)

    assert isinstance(metrics, dict)
    assert "MAE" in metrics
    assert "MAE_standardized_by_mean" in metrics
    assert "MAE_standardized_by_range" in metrics
    assert "MSE" in metrics
    assert "RMSE" in metrics
    assert "RMSE_standardized_by_mean" in metrics
    assert "RMSE_standardized_by_range" in metrics
    assert "R2" in metrics
    assert isinstance(params, dict)


def test_train_pipeline_invalid_input_type():
    pipeline = Pipeline([
        ("model", XGBRegressor(n_estimators=10, random_state=42))
    ])

    with pytest.raises(ValueError, match="Input data must be a pandas DataFrame."):
        train_pipeline(pipeline,
                       {"trip_seconds": [100, 200], "trip_miles": [2.5, 3.5]})  # Dictionary instead of DataFrame


def test_train_pipeline_empty_dataframe():
    pipeline = Pipeline([
        ("model", XGBRegressor(n_estimators=10, random_state=42))
    ])

    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match="Input data is empty."):
        train_pipeline(pipeline, empty_df)


def test_train_pipeline_missing_columns():
    pipeline = Pipeline([
        ("model", XGBRegressor(n_estimators=10, random_state=42))
    ])

    # DataFrame missing some required columns
    incomplete_data = pd.DataFrame({
        "trip_seconds": np.random.randint(60, 3600, size=100),
        "trip_miles": np.random.uniform(0.5, 30, size=100),
        "payment_type": np.random.choice(["Credit Card", "Cash"], size=100),
        "trip_total": np.random.uniform(5, 100, size=100),
        "company": np.random.choice(["Company A", "Company B"], size=100),
    })

    with pytest.raises(ValueError, match="Missing required columns:"):
        train_pipeline(pipeline, incomplete_data)


def test_train_pipeline_with_full_pipeline(sample_data):
    numerical_cols = ["trip_seconds", "trip_miles", "tolls", "extras", "daytime", "month", "day_of_week",
                      "day_of_month", "avg_tips"]
    categorical_cols = ["payment_type", "company", "day_type"]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ]), numerical_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_cols)
        ]
    )

    model = XGBRegressor(n_estimators=10, random_state=42)

    pipeline = Pipeline([
        ("preprocessing", preprocess),
        ("model", model)
    ])

    data = sample_data

    trained_pipeline, metrics, X_train, y_train, X_test, y_test, params = train_pipeline(pipeline, data)

    predictions = trained_pipeline.predict(X_test)


    assert predictions.shape == y_test.shape
    for p in predictions:
        assert p > 0
    assert len(X_train) > len(X_test), "Training set should be larger than test set"
    assert len(y_train) > len(y_test), "Training labels should be larger than test labels"
    assert not np.isnan(predictions).any(), "Predictions contain NaN values"
    assert np.isfinite(predictions).all(), "Predictions contain infinite values"
    assert trained_pipeline is not None
    assert isinstance(metrics, dict)

    sample_input = X_test.iloc[:1]
    first_prediction = trained_pipeline.predict(sample_input)
    second_prediction = trained_pipeline.predict(sample_input)
    assert np.allclose(first_prediction, second_prediction), "Model is making inconsistent predictions"

    new_data = X_test.copy()
    new_data["company"] = "New Unknown Company"  # Inject an unseen category
    try:
        trained_pipeline.predict(new_data)
    except Exception as e:
        assert "Found unknown categories" not in str(e), "Pipeline is failing on unseen categorical values"


    # Test MR4
    # Apply perturbation
    perturbed_data = data.copy()
    for col in numerical_cols:
        perturbed_data[col] += np.random.normal(0, 0.01, size=perturbed_data.shape[0])  # Small Gaussian noise

    # Use the same trained pipeline to make predictions
    X_test_perturbed = perturbed_data.loc[X_test.index]  # Keep test indices the same
    predictions_perturbed = trained_pipeline.predict(X_test_perturbed)

    # Assert that predictions remain stable
    correlation = np.corrcoef(predictions, predictions_perturbed)[0, 1]
    assert correlation > 0.95, f"Prediction correlation too low: {correlation}"

