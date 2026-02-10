#!/usr/bin/env bash
set -euo pipefail

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env_config.sh"

echo "Step 1: Creating Artifact Registry ($REPO_NAME)..."
if ! gcloud artifacts repositories describe "$REPO_NAME" --project="$PROJECT_ID" --location="$REGION" >/dev/null 2>&1; then
    gcloud artifacts repositories create "$REPO_NAME" \
        --repository-format=docker \
        --location="$REGION" \
        --description="Gomoku project Docker repository" \
        --project="$PROJECT_ID"
    echo "  - Created repository $REPO_NAME in $REGION"
else
    echo "  - Repository $REPO_NAME already exists"
fi

echo "Step 2: Creating Cloud Storage Bucket (gs://$BUCKET_NAME)..."
if ! gcloud storage buckets describe "gs://$BUCKET_NAME" --project="$PROJECT_ID" >/dev/null 2>&1; then
    gcloud storage buckets create "gs://$BUCKET_NAME" \
        --location="$REGION" \
        --default-storage-class=STANDARD \
        --project="$PROJECT_ID"
    echo "  - Created bucket gs://$BUCKET_NAME"
else
    echo "  - Bucket gs://$BUCKET_NAME already exists"
fi

echo "Storage setup complete."
