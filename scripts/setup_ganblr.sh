#!/bin/bash
# GANBLR Model Development Setup
echo "Setting up GANBLR model development environment..."

# Install dependencies in model directory
cd katebatic/models/ganblr
poetry install
cd ../../..

echo "✓ GANBLR development environment ready!"
echo "Activate with: cd katebatic/models/ganblr && poetry shell"
