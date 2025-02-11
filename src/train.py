import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import numpy as np



def create_pipeline():
    # Define numerical and categorical columns
    categorical_cols = ["payment_type", "company"]
    numerical_cols = ["trip_seconds", "trip_miles", "tolls", "extras", "avg_tips"]

    # ColumnTransformer for numerical and categorical processing
    column_transformer = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ]), numerical_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), categorical_cols)
        ],
        remainder="drop"
    )

    # Wrap everything in a pipeline
    preprocess = Pipeline([
        ("column_transformer", column_transformer)
    ])

    print(preprocess)

    # Define the model
    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        objective="reg:squarederror"
    )

    # Create a pipeline that chains preprocessing and modeling
    pipeline = Pipeline(steps=[
        ("preprocessing", preprocess),
        ("model", model)
    ])

    return pipeline


def train_pipeline(pipeline, data):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")
    if data.empty:
        raise ValueError("Input data is empty.")

    required_columns = {"trip_seconds", "trip_miles", "payment_type", "company", "extras", "tolls", "avg_tips", "trip_total"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    X = data[["trip_seconds", "trip_miles", "payment_type", "company", "extras", "tolls", "avg_tips"]]
    y = data["trip_total"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Train the pipeline
    pipeline.fit(X_train, y_train)

    # Evaluate the trained pipeline on the test set
    y_pred = pipeline.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred)
    }

    params = pipeline.named_steps["model"].get_params()

    return pipeline, metrics, X_train, y_train, X_test, y_test, params