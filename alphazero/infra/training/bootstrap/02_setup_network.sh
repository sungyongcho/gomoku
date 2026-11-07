#!/usr/bin/env bash
set -euo pipefail

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env_config.sh"

export NETWORK="default"
export SUBNET="default" # Change this if your subnet name differs.
# Short region code for names, for example europe-west4 -> ew4.
REGION_CODE="$(sed -E 's/^([a-z])[a-z]*-([a-z])[a-z]*([0-9]+)$/\1\2\3/' <<<"${REGION}")"
export ROUTER="${NETWORK}-router-${REGION_CODE}"
export NAT="${NETWORK}-nat-${REGION_CODE}"

echo "Using Project: $PROJECT_ID, Region: $REGION"

echo "Step 1: Getting Subnet CIDR..."
SUBNET_RANGE=$(gcloud compute networks subnets describe ${SUBNET} \
  --project=${PROJECT_ID} \
  --region=${REGION} \
  --format='value(ipCidrRange)')
echo "  - Subnet Range: $SUBNET_RANGE"

echo "Step 2: Creating Internal Firewall Rules..."
if ! gcloud compute firewall-rules describe ray-allow-internal --project=${PROJECT_ID} >/dev/null 2>&1; then
    gcloud compute firewall-rules create ray-allow-internal \
        --project=${PROJECT_ID} \
        --network=${NETWORK} \
        --direction=INGRESS \
        --priority=1000 \
        --action=ALLOW \
        --rules=tcp,udp,icmp \
        --source-ranges=${SUBNET_RANGE} \
        --target-tags=ray-node
    echo "  - Created ray-allow-internal"
else
    RANGES=$(gcloud compute firewall-rules describe ray-allow-internal --project=${PROJECT_ID} \
        --format='value(sourceRanges.list())')
    if [[ ",${RANGES}," != *",${SUBNET_RANGE},"* ]]; then
        gcloud compute firewall-rules update ray-allow-internal \
            --project=${PROJECT_ID} \
            --source-ranges="${RANGES},${SUBNET_RANGE}"
        echo "  - Added ${SUBNET_RANGE} to ray-allow-internal"
    else
        echo "  - ray-allow-internal already covers ${SUBNET_RANGE}"
    fi
fi

echo "Step 3: Creating Firewall Rules for Home IP..."
MYIP4=$(curl -4 -s https://ifconfig.me)
if [ -n "$MYIP4" ]; then
    if ! gcloud compute firewall-rules describe ray-allow-head-from-home-v4 --project=${PROJECT_ID} >/dev/null 2>&1; then
        gcloud compute firewall-rules create ray-allow-head-from-home-v4 \
            --project=${PROJECT_ID} \
            --network=${NETWORK} \
            --direction=INGRESS \
            --priority=1000 \
            --action=ALLOW \
            --rules=tcp:22,tcp:8265,tcp:10001 \
            --source-ranges=${MYIP4}/32 \
            --target-tags=ray-head
        echo "  - Created ray-allow-head-from-home-v4 ($MYIP4)"
    else
        echo "  - ray-allow-head-from-home-v4 already exists ($MYIP4)"
        # Use 'update' to refresh IP if needed? For now just skip.
    fi
fi

# IAP SSH Access
if ! gcloud compute firewall-rules describe ray-allow-iap-ssh --project=${PROJECT_ID} >/dev/null 2>&1; then
    gcloud compute firewall-rules create ray-allow-iap-ssh \
        --project=${PROJECT_ID} \
        --network=${NETWORK} \
        --direction=INGRESS \
        --priority=900 \
        --action=ALLOW \
        --rules=tcp:22 \
        --source-ranges=35.235.240.0/20 \
        --target-tags=ray-node
    echo "  - Created ray-allow-iap-ssh"
else
    gcloud compute firewall-rules update ray-allow-iap-ssh \
        --project=${PROJECT_ID} \
        --target-tags=ray-node
    echo "  - ray-allow-iap-ssh already exists (targets ray-node)"
fi

# The default network opens SSH and RDP to the internet. Ray needs only the rules above.
for rule in default-allow-ssh default-allow-rdp; do
    if gcloud compute firewall-rules describe "${rule}" --project=${PROJECT_ID} >/dev/null 2>&1; then
        gcloud compute firewall-rules delete "${rule}" --project=${PROJECT_ID} --quiet
        echo "  - Deleted ${rule}"
    fi
done

# The head node needs an external IP. An organization can deny external IPs for every VM;
# allow them in this project only. Workers stay private because their config has no accessConfigs.
# Revert after training: gcloud resource-manager org-policies delete compute.vmExternalIpAccess --project=${PROJECT_ID}
EXT_IP_POLICY=$(gcloud resource-manager org-policies describe compute.vmExternalIpAccess \
    --project=${PROJECT_ID} --effective --format='value(listPolicy.allValues)' 2>/dev/null || true)
if [[ "${EXT_IP_POLICY}" == "DENY" ]]; then
    POLICY_FILE="$(mktemp)"
    printf 'constraint: constraints/compute.vmExternalIpAccess\nlistPolicy:\n  allValues: ALLOW\n' > "${POLICY_FILE}"
    gcloud resource-manager org-policies set-policy "${POLICY_FILE}" --project=${PROJECT_ID} >/dev/null
    rm -f "${POLICY_FILE}"
    echo "  - Allowed external IPs in ${PROJECT_ID} (organization denies them)"
else
    echo "  - External IPs allowed in ${PROJECT_ID}"
fi


echo "Step 4: Setting up Cloud NAT..."
if ! gcloud compute routers describe ${ROUTER} --project=${PROJECT_ID} --region=${REGION} >/dev/null 2>&1; then
    gcloud compute routers create ${ROUTER} \
        --project=${PROJECT_ID} \
        --network=${NETWORK} \
        --region=${REGION}
    echo "  - Created router $ROUTER"
else
    echo "  - Router $ROUTER already exists"
fi

if ! gcloud compute routers nats describe ${NAT} --router=${ROUTER} --project=${PROJECT_ID} --region=${REGION} >/dev/null 2>&1; then
    gcloud compute routers nats create ${NAT} \
        --project=${PROJECT_ID} \
        --router=${ROUTER} \
        --region=${REGION} \
        --auto-allocate-nat-external-ips \
        --nat-all-subnet-ip-ranges
    echo "  - Created NAT $NAT"
else
    echo "  - NAT $NAT already exists"
fi

echo "Network setup complete."
