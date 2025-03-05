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

python /app/src/main.py


