#!/bin/bash
# GANBLR Model Development Setup
echo "Setting up GANBLR model development environment..."

# Install dependencies in model directory
cd katabatic/models/ganblr
poetry install
cd ../../..

echo "✓ GANBLR development environment ready!"
echo "Activate with: cd katabatic/models/ganblr && poetry shell"
