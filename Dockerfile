# Use the official Python image
FROM python:3.12-slim

# Install dependencies for gsutil (Google Cloud SDK)
RUN apt-get update && apt-get install -y wget curl gnupg && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    echo "deb http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    apt-get update && apt-get install -y google-cloud-sdk && \
    apt-get clean


# Install dependencies for mlflow tracking
RUN pip install --no-cache-dir pip==24.3.1 setuptools==75.6.0 wheel==0.43.0

# Install git
RUN apt-get update && apt-get upgrade -y && apt-get install -y git


# Install jq
RUN apt-get update && apt-get install -y jq

# Set the working directory
WORKDIR /app

# Copy application files
COPY . /app

# Include service account for local testing
#COPY configs/keys.json /app/keys.json

# Install local requirements
RUN pip install --no-cache-dir -r requirements-local.txt

# Expose the port the app runs on
EXPOSE 8080

# Use the entrypoint script to fetch requirements and start the app
#COPY Docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Set the entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]