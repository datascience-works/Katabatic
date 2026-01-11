#!/bin/bash
# GReaT Model Development Setup
echo "Setting up GReaT model development environment..."

# Install dependencies in model directory
cd katabatic/models/great
poetry install
cd ../../..

echo "✓ GReaT development environment ready!"
echo "Activate with: cd katabatic/models/great && poetry shell"
