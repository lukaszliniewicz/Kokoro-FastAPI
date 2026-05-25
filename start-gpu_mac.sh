#!/bin/bash

# Get project root directory
PROJECT_ROOT=$(pwd)

# Set other environment variables
export USE_GPU=true
export USE_ONNX=false
export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/api
export MODEL_DIR=src/models
export VOICES_DIR=src/voices/v1_0
export WEB_PLAYER_PATH=$PROJECT_ROOT/web
export DOWNLOAD_EXTERNAL_MODELS=${DOWNLOAD_EXTERNAL_MODELS:-false}
export DOWNLOAD_EXTERNAL_PROFILE_IDS=${DOWNLOAD_EXTERNAL_PROFILE_IDS:-}
export DOWNLOAD_GERMAN_MARTIN=${DOWNLOAD_GERMAN_MARTIN:-false}

export DEVICE_TYPE=mps
# Enable MPS fallback for unsupported operations
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Run FastAPI with GPU extras using uv run
uv pip install -e .
DOWNLOAD_ARGS=(--output api/src/models/v1_0)
if [ "$DOWNLOAD_EXTERNAL_MODELS" = "true" ]; then
    DOWNLOAD_ARGS+=(--with-external-profiles)
fi
if [ -n "$DOWNLOAD_EXTERNAL_PROFILE_IDS" ]; then
    DOWNLOAD_ARGS+=(--with-external-profiles)
    DOWNLOAD_ARGS+=(--external-profile-ids "$DOWNLOAD_EXTERNAL_PROFILE_IDS")
fi
if [ "$DOWNLOAD_GERMAN_MARTIN" = "true" ]; then
    DOWNLOAD_ARGS+=(--with-german-martin)
fi
uv run --no-sync python docker/scripts/download_model.py "${DOWNLOAD_ARGS[@]}"
uv run --no-sync uvicorn api.src.main:app --host 0.0.0.0 --port 8880
