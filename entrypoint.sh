#!/bin/bash
export FEATURE_STORE_PATH=/app/src/

#export GOOGLE_APPLICATION_CREDENTIALS=/app/configs/keys.json


DATASET_NAME=$(jq -r '.DATASET_NAME' /app/configs/config.json)
export DATASET_NAME
PROJECT_ID=$(jq -r '.PROJECT_ID' /app/configs/config.json)
export PROJECT_ID
TEST_RUN=$(jq -r '.TEST_RUN' /app/configs/config.json)
export TEST_RUN
TRAINING_SIZE=$(jq -r '.TRAINING_SIZE' /app/configs/config.json)
export TRAINING_SIZE

gcloud config set project PROJECT_ID

python /app/src/main.py


#RUN_ID=$(jq -r '.RUN_ID' /app/configs/config.json)

#export RUN_ID


# Fetch the dynamically generated requirements.txt from Google Cloud Storage
#gsutil cp "$RUN_ID/artifacts/model/requirements.txt" /app/requirements.txt

#gsutil cp /app/logs/app.log "$RUN_ID/artifacts/artifacts/app.log"

# Install dependencies from the dynamically generated requirements
#pip install --no-cache-dir -r /app/requirements.txt

# Start the application
#python app.py