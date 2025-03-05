#!/bin/bash
export FEATURE_STORE_PATH=/app/src/



DATASET_NAME=$(jq -r '.DATASET_NAME' /app/configs/config.json)
export DATASET_NAME
PROJECT_ID=$(jq -r '.PROJECT_ID' /app/configs/config.json)
export PROJECT_ID
TEST_RUN=$(jq -r '.TEST_RUN' /app/configs/config.json)
export TEST_RUN
TRAINING_SIZE=$(jq -r '.TRAINING_SIZE' /app/configs/config.json)
export TRAINING_SIZE

# Check if authentication is set up
if gcloud auth list --format="value(account)" | grep -q "@"; then
    echo "✅ Google Cloud authentication is active."
else
    echo "❌ Google Cloud authentication is missing!"
    exit 1
fi


#
# export GOOGLE_APPLICATION_CREDENTIALS=/app/configs/keys.json

# Set default project (Replace with your project ID)

python /app/src/main.py


