#!/bin/bash

# Define variables
REPO_URL="https://github.com/dolr-ai/recommendation-engine.git"
INSTALL_DIR="/home/dataproc/recommendation-engine"

# Log all steps for debugging
set -x

echo "Starting repository cloning and setup process..."

# Retrieve metadata values and set as environment variables
echo "Retrieving environment variables from metadata..."
GCP_CREDENTIALS=$(curl -f -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/GCP_CREDENTIALS")
SERVICE_ACCOUNT=$(curl -f -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/SERVICE_ACCOUNT")

# Export as environment variables
export GCP_CREDENTIALS
export SERVICE_ACCOUNT

# Verify environment variables were set
echo "Verifying environment variables..."
if [ -z "$GCP_CREDENTIALS" ]; then
    echo "Warning: GCP_CREDENTIALS not set properly"
fi

if [ -z "$SERVICE_ACCOUNT" ]; then
    echo "Warning: SERVICE_ACCOUNT not set properly"
fi

# Clone the repository
echo "Cloning repository ${REPO_URL}..."
git clone -b dev ${REPO_URL} ${INSTALL_DIR}

# Check if clone was successful
if [ $? -ne 0 ]; then
    echo "Failed to clone repository. Exiting."
    exit 1
fi

# Change to the repository directory
cd ${INSTALL_DIR}

# Write credentials to a file
if [ ! -z "$GCP_CREDENTIALS" ]; then
    echo "Writing GCP credentials to file..."
    CREDENTIALS_FILE="${INSTALL_DIR}/credentials.json"
    echo "$GCP_CREDENTIALS" >"$CREDENTIALS_FILE"
    export GCP_CREDENTIALS_PATH="$CREDENTIALS_FILE"
    echo "Credentials file created at: $CREDENTIALS_FILE"

    # Make credentials available system-wide
    echo "Making credentials available system-wide..."
    echo "export GCP_CREDENTIALS='$GCP_CREDENTIALS'" >>/etc/profile
    echo "export GCP_CREDENTIALS_PATH='$CREDENTIALS_FILE'" >>/etc/profile

    # Also add to .bashrc for all users
    echo "export GCP_CREDENTIALS='$GCP_CREDENTIALS'" >>/etc/bash.bashrc
    echo "export GCP_CREDENTIALS_PATH='$CREDENTIALS_FILE'" >>/etc/bash.bashrc

    # Make credentials available to Spark jobs and Jupyter
    echo "Setting Spark defaults for credentials..."
    echo "spark.executorEnv.GCP_CREDENTIALS=$GCP_CREDENTIALS" >>/etc/spark/conf/spark-defaults.conf
    echo "spark.executorEnv.GCP_CREDENTIALS_PATH=$CREDENTIALS_FILE" >>/etc/spark/conf/spark-defaults.conf
    echo "spark.yarn.appMasterEnv.GCP_CREDENTIALS=$GCP_CREDENTIALS" >>/etc/spark/conf/spark-defaults.conf
    echo "spark.yarn.appMasterEnv.GCP_CREDENTIALS_PATH=$CREDENTIALS_FILE" >>/etc/spark/conf/spark-defaults.conf

    # Make credentials available to Jupyter
    echo "Setting Jupyter environment variables..."
    echo "export GCP_CREDENTIALS='$GCP_CREDENTIALS'" >>/home/jupyter/.bashrc
    echo "export GCP_CREDENTIALS_PATH='$GCP_CREDENTIALS_FILE'" >>/home/jupyter/.bashrc
fi

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Check if pip install was successful
if [ $? -ne 0 ]; then
    echo "Warning: Some dependencies may not have installed correctly."
    # Continue anyway as some might be already satisfied by Dataproc
fi

# Install the package in development mode
echo "Installing package in development mode..."
pip install -e .

# Make scripts executable if needed
find ${INSTALL_DIR} -name "*.py" -exec chmod +x {} \;

echo "Repository setup completed successfully."
