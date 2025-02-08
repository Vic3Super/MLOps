# check model being able to predict
# validate against last running model

from mlflow.models import validate_serving_input
from mlflow.models import convert_input_example_to_serving_input
import mlflow
from mlflow.models import MetricThreshold

def validate_serving(input_example, model_uri):

    serving_payload = convert_input_example_to_serving_input(input_example)

    # Validate the serving payload works on the model
    output = validate_serving_input(model_uri, serving_payload)

    return output

def validate_model(candidate_model_uri, X_test, y_test, parent_run_id, experiment):

    # construct an evaluation dataset from the test set
    eval_data = X_test
    eval_data["label"] = y_test

    with mlflow.start_run(run_name="validation_run", parent_run_id=parent_run_id, experiment_id=experiment.experiment_id) as run:
        candidate_result = mlflow.evaluate(
            candidate_model_uri,
            eval_data,
            targets="label",
            model_type="regressor",
        )

    # Define criteria for model to be validated against
    thresholds = {
        "mean_absolute_error": MetricThreshold(
            threshold=10,# accuracy should be >=0.8
            greater_is_better=False,
        ),
        "root_mean_squared_error": MetricThreshold(
            threshold=15,greater_is_better=False,
        ),
        "r2_score": MetricThreshold(
            threshold=0.7,greater_is_better=True,
        )
    }

    # Validate the candidate model agaisnt baseline
    mlflow.validate_evaluation_results(
        candidate_result=candidate_result,
        validation_thresholds=thresholds,
    )