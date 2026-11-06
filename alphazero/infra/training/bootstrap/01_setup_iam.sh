#!/usr/bin/env bash
set -euo pipefail

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env_config.sh"

# Check ADC credentials
if [ ! -f ~/.config/gcloud/application_default_credentials.json ]; then
    echo "ADC credentials not found. Running gcloud auth application-default login..."
    gcloud auth application-default login
fi

echo "Step 1: Enabling required services..."
gcloud services enable \
    compute.googleapis.com \
    artifactregistry.googleapis.com \
    iam.googleapis.com \
    cloudbuild.googleapis.com \
    oslogin.googleapis.com \
    cloudresourcemanager.googleapis.com \
    --project="$PROJECT_ID"


echo "Step 2: SSH Key Setup..."
rm -f ~/.ssh/id_rsa_gcp ~/.ssh/id_rsa_gcp.pub
ssh-keygen -t rsa -f ~/.ssh/id_rsa_gcp -C "$USER_EMAIL" -N ""
# Only add if not already added (to avoid errors or duplicates if run multiple times)
# But `os-login ssh-keys add` is generally idempotent or safe to re-run
gcloud compute os-login ssh-keys add --key-file ~/.ssh/id_rsa_gcp.pub --project="$PROJECT_ID" || true
gcloud compute project-info add-metadata --metadata enable-oslogin=TRUE --project="$PROJECT_ID"


echo "Step 3: Service Account Setup ($SA_NAME)..."
if ! gcloud iam service-accounts describe "$SA_EMAIL" --project="$PROJECT_ID" >/dev/null 2>&1; then
    gcloud iam service-accounts create "$SA_NAME" --display-name="Ray Autoscaler SA" --project="$PROJECT_ID"
    echo "  Waiting for service account propagation..."
    sleep 10
else
    echo "Service account $SA_NAME already exists."
fi

echo "Step 4: Granting Roles to Service Account..."
ROLES=(
    "roles/artifactregistry.reader"
    "roles/logging.logWriter"
    "roles/monitoring.metricWriter"
    "roles/compute.instanceAdmin.v1"
    "roles/storage.admin"
)

for role in "${ROLES[@]}"; do
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="serviceAccount:${SA_EMAIL}" \
        --role="$role" >/dev/null
    echo "  - Granted $role to $SA_EMAIL"
done


echo "Step 5: Granting Roles to User ($USER_EMAIL)..."
USER_ROLES=(
    "roles/compute.osAdminLogin"
    "roles/compute.instanceAdmin.v1"
    "roles/compute.securityAdmin"
    "roles/artifactregistry.reader"
    "roles/storage.admin"
)

for role in "${USER_ROLES[@]}"; do
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="user:${USER_EMAIL}" \
        --role="$role" >/dev/null
    echo "  - Granted $role to $USER_EMAIL"
done

# Grant Service Account User role to the user on the SA
gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
    --member="user:${USER_EMAIL}" \
    --role="roles/iam.serviceAccountUser" \
    --project="$PROJECT_ID" >/dev/null
echo "  - Granted roles/iam.serviceAccountUser to $USER_EMAIL on $SA_EMAIL"

echo "IAM setup complete."
