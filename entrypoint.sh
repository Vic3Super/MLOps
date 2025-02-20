#!/bin/bash

# Set the environment variable for app.py

# Authenticate with Google Cloud
gcloud auth activate-service-account --key-file=/app/configs/keys.json

# Export the credentials file path as an environment variable
export GOOGLE_APPLICATION_CREDENTIALS=/app/configs/keys.json

export FEATURE_STORE_PATH=/app/src/


DATASET_NAME=$(jq -r '.DATASET_NAME' /app/configs/config.json)
export DATASET_NAME
PROJECT_ID=$(jq -r '.PROJECT_ID' /app/configs/config.json)
export PROJECT_ID
TEST_RUN=$(jq -r '.TEST_RUN' /app/configs/config.json)
export TEST_RUN
TRAINING_SIZE=$(jq -r '.TRAINING_SIZE' /app/configs/config.json)
export TRAINING_SIZE

# Run main.py to generate the model and requirements
python /app/src/main.py

RUN_ID=$(jq -r '.RUN_ID' /app/configs/config.json)

export RUN_ID


# Parse GCS_REQUIREMENTS_PATH from config.json
#GCS_REQUIREMENTS_PATH=$(jq -r '.RUN_ID' /app/configs/config.json)/requirements.txt

# Parse APP_ENV_VAR from config.json
#MODEL_PATH_ENV=$(jq -r '.MODEL_PATH' /app/configs/config.json)

# Validate that variables were read correctly
#if [ -z "$GCS_REQUIREMENTS_PATH" ]; then
#  echo "Error: GCS_REQUIREMENTS_PATH is not set in config.json"
#  exit 1
#fi


# Print a message to confirm the variable is set
#echo "GOOGLE_APPLICATION_CREDENTIALS is set to $GOOGLE_APPLICATION_CREDENTIALS"


#if [ -z "$MODEL_PATH_ENV" ]; then
#  echo "Error: APP_ENV_VAR is not set in config.json"
#  exit 1
#fi

#export MODEL_PATH_ENV


# Fetch the dynamically generated requirements.txt from Google Cloud Storage
gsutil cp "$RUN_ID/artifacts/model/requirements.txt" /app/requirements.txt

gsutil cp /app/logs/app.log "$RUN_ID/artifacts/artifacts/app.log"

# Install dependencies from the dynamically generated requirements
pip install --no-cache-dir -r /app/requirements.txt

# Start the application
python app.py