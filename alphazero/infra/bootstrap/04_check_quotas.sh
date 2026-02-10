#!/usr/bin/env bash
set -e

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env_config.sh"

echo "Checking Quotas for Project: $PROJECT_ID / Region: $REGION"
echo "---------------------------------------------------------"

# 1. Check Global GPU Quota
echo "[Global] GPUS_ALL_REGIONS:"
CURRENT_GPU=$(gcloud compute project-info describe --project="$PROJECT_ID" --format="value(quotas.metric.GPUS_ALL_REGIONS.limit)" || echo "N/A")
echo "  Current Limit: $CURRENT_GPU"
echo "  Target: 1"

# 2. Check Regional CPU Quotas (e.g., C3, N2, etc. based on C4 request or general)
# Note: 'C4' (Google Axion/new gen) might not be visible or might be 'C3_CPUS' / 'C4_CPUS' depending on rolled out features.
# We will check 'C3_CPUS' and 'CPUS' (generic) as examples.
echo ""
echo "[Region: $REGION] CPU Quotas:"
gcloud compute regions describe "$REGION" --project="$PROJECT_ID" --format="table[box](quotas.metric, quotas.limit, quotas.usage)" | grep -E "CPUS|GPUS" | head -n 10
echo "..."

echo "---------------------------------------------------------"
echo "To request quota increases, use the Google Cloud Console:"
echo ""
echo "  https://console.cloud.google.com/iam-admin/quotas?project=$PROJECT_ID"
echo ""
echo "  1. GPUS_ALL_REGIONS -> 1"
echo "  2. CPUS (all regions) -> 90"
echo "  3. C4_CPUS (${REGION}) -> 90"
echo ""
echo "Filter by 'GPUs' or 'CPUs', select the quota, and click 'EDIT QUOTAS'."
echo "---------------------------------------------------------"
