
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import numpy as np

def train_pipeline(pipeline, data):
    """
    Train the pipeline and evaluate the model.

    Args:
        pipeline: The sklearn Pipeline object.
        data: Pandas DataFrame containing the training data.

    Returns:
        pipeline: The fitted pipeline.
        metrics: A dictionary containing evaluation metrics.
        X_train, y_train, X_test, y_test: Train and test splits for further use.
    """
    # Split data
    X = data[["Trip Seconds", "Trip Miles", "Company"]]
    y = data["Trip Total"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Fit pipeline
    pipeline.fit(X_train, y_train)

    # Evaluate model
    y_pred = pipeline.predict(X_test)

    # Compute evaluation metrics
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred)
    }

    print("Model Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    return pipeline, metrics, X_train, y_train, X_test, y_test


# Define pipeline for preprocessing and modeling
def create_pipeline():
    """
    Create a preprocessing and modeling pipeline.

    Returns:
        pipeline: The sklearn Pipeline object.
    """
    # Define numerical and categorical columns
    numerical_cols = ["Trip Seconds", "Trip Miles"]
    categorical_cols = ["Company"]

    # Preprocessing pipeline
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
        ],
        remainder="drop"
    )

    # Define the model
    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        objective="reg:squarederror"
    )

    # Create a pipeline
    pipeline = Pipeline(steps=[
        ("preprocessing", preprocess),
        ("model", model)
    ])

    return pipeline
