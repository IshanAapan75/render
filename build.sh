#!/usr/bin/env bash
# exit on error
set -o errexit

# Install system dependencies
apt-get update
apt-get install -y pandoc wkhtmltopdf libpango-1.0-0 libharfbuzz0b libpangoft2-1.0-0

# Install Python dependencies
pip install -r requirements.txt
