#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
uv pip install -r requirements.txt

# Run the web app
echo "Starting WanVideo Data Flow Explorer..."
streamlit run wan_dataflow_webapp.py --server.port 8501 --server.address localhost
