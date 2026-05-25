#!/bin/bash
set -e

if [ "$DOWNLOAD_MODEL" = "true" ]; then
    DOWNLOAD_ARGS=(--output api/src/models/v1_0)

    if [ "${DOWNLOAD_EXTERNAL_MODELS:-false}" = "true" ]; then
        DOWNLOAD_ARGS+=(--with-external-profiles)
    fi

    if [ -n "${DOWNLOAD_EXTERNAL_PROFILE_IDS:-}" ]; then
        DOWNLOAD_ARGS+=(--with-external-profiles)
        DOWNLOAD_ARGS+=(--external-profile-ids "$DOWNLOAD_EXTERNAL_PROFILE_IDS")
    fi

    if [ "${DOWNLOAD_GERMAN_MARTIN:-false}" = "true" ]; then
        DOWNLOAD_ARGS+=(--with-german-martin)
    fi

    python download_model.py "${DOWNLOAD_ARGS[@]}"
fi

exec uv run --extra $DEVICE --no-sync python -m uvicorn api.src.main:app --host 0.0.0.0 --port 8880 --log-level debug
