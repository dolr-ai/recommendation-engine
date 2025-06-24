"""
Run script for the recommendation service.

This script starts the FastAPI server for the recommendation service.
"""

import os
import sys
import logging
from service.app import start

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

if __name__ == "__main__":
    # Check if GCP credentials are set
    if "GCP_CREDENTIALS" not in os.environ:
        print("ERROR: GCP_CREDENTIALS environment variable must be set.")
        print("Use: export GCP_CREDENTIALS=$(jq -c . '/path/to/credentials.json')")
        sys.exit(1)

    # Start the API server
    print("Starting recommendation service API...")
    start()
