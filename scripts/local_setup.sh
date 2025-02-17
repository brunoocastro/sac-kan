#!/usr/bin/env bash

# # Create .env file
# if [ ! -f .env ]; then
#   cp env.example .env
# fi

# Setup VENV
if [ ! -d .venv ]; then
  python3 -m venv venv
fi

source venv/bin/activate &&
  pip install --upgrade pip &&
  pip install -r requirements.txt

# Install pre-commit hooks
# - Install pre-commit if not installed
if ! command -v pre-commit &>/dev/null; then
  pip install pre-commit
fi

pre-commit install