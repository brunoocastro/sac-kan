#!/usr/bin/env bash

# Verify if submodules are already cloned
if [ ! -d .git ]; then
  git submodule update --init --recursive
fi

# Setup VENV
if [ ! -d .venv ]; then
  python3 -m venv venv
fi

source venv/bin/activate &&
  pip install --upgrade pip &&
  pip install -r requirements.txt &&
  pip install -r pykan/requirements.txt &&
  pip install -r cleanrl/requirements/requirements.txt &&
  pip install -r cleanrl/requirements/requirements-mujoco.txt

# Install pre-commit hooks
# - Install pre-commit if not installed
if ! command -v pre-commit &>/dev/null; then
  pip install pre-commit
fi

pre-commit install
