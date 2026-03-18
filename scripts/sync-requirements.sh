#!/usr/bin/env bash
set -euo pipefail

REQUIREMENTS="deploy/requirements.txt"
TMP="${REQUIREMENTS}.tmp"

uv export --no-hashes --no-dev --extra streamlit --no-emit-project --no-annotate > "$TMP"
echo "." >> "$TMP"

if ! diff -q "$REQUIREMENTS" "$TMP" > /dev/null 2>&1; then
    mv "$TMP" "$REQUIREMENTS"
    echo "${REQUIREMENTS} was out of date and has been updated. Please stage and re-commit."
    exit 1
else
    rm "$TMP"
    exit 0
fi
