#!/bin/bash

# Recommendation Engine Test App Launcher
# This script sets up the environment and runs the Streamlit app

echo "🎬 Starting Recommendation Engine Test App..."

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found. Please run this script from the test_app directory."
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if streamlit is installed
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "📦 Installing Streamlit dependencies..."
    pip install -r requirements.txt
fi

# Set environment variables if needed
export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."

echo "🚀 Launching Streamlit app..."
echo "📱 The app will be available at: http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the app"
echo ""

# Run the Streamlit app
streamlit run app.py --server.port 8501 --server.address 0.0.0.0