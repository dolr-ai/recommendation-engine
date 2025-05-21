#!/bin/bash

# Define variables
REPO_URL="https://github.com/dolr-ai/recommendation-engine.git"
INSTALL_DIR="/home/dataproc/recommendation-engine"

# Log all steps for debugging
set -x

echo "Starting repository cloning and setup process..."

# Clone the repository
echo "Cloning repository ${REPO_URL}..."
git clone ${REPO_URL} ${INSTALL_DIR}

# Check if clone was successful
if [ $? -ne 0 ]; then
    echo "Failed to clone repository. Exiting."
    exit 1
fi

# Change to the repository directory
cd ${INSTALL_DIR}

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
