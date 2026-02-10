# Cloud Instructions (GCP + Ray)

This file is a single reference for GCP setup, image build, and Ray cluster operations.
It uses environment variables only (no hard-coded project-specific values).

## Scope

- Set up auth and IAM
- Set up network, firewall, NAT, artifact registry, and bucket
- Build/push training images
- Reserve GPU VM capacity
- Deploy/redeploy Ray cluster
- Clean up local auth state and cloud resources when needed

## Working Directory

Run commands from the repo root (where `.env` exists), unless noted otherwise.

## Required Environment Variables

These are expected in `.env`:

- `GCP_PROJECT`
- `GCP_REGION`
- `GCP_ZONE`
- `GCP_REPO`
- `GCP_CLUSTER_NAME`
- `GCP_SSH_USER`
- `GCP_SSH_PRIVATE_KEY`
- `GCP_CONTAINER_NAME`
- `GCP_SA_NAME`
- `GCP_HEAD_RESERVATION`
- `GCP_GPU_TAG`
- `GCP_CPU_TAG`
- `GCP_USER_EMAIL`
- `GCP_BUCKET_NAME`

## Load Environment

Load `.env` into the current shell:

```bash
set -a
source .env
set +a
```

Load helper exports used by bootstrap scripts (`PROJECT_ID`, `SA_EMAIL`, etc.):

```bash
source alphazero/infra/bootstrap/env_config.sh
```

## Recommended Script-Based Flow

Use these scripts for the fastest reproducible setup.

Initialize IAM, APIs, SSH key, service account, and role bindings:

```bash
bash alphazero/infra/bootstrap/01_setup_iam.sh
```

Create firewall rules and Cloud NAT:

```bash
bash alphazero/infra/bootstrap/02_setup_network.sh
```

Create artifact registry and storage bucket:

```bash
bash alphazero/infra/bootstrap/03_setup_storage.sh
```

Check regional and global CPU/GPU quotas:

```bash
bash alphazero/infra/bootstrap/04_check_quotas.sh
```

Reserve head-node GPU capacity (loop until reservation succeeds):

```bash
bash alphazero/infra/cluster/reserver_vm.sh
```

Delete reservation:

```bash
bash alphazero/infra/cluster/reserver_vm.sh -d
```

Rebuild images, push, and redeploy Ray cluster:

```bash
bash alphazero/infra/cluster/restart_cluster.sh
```

## Manual Command Reference

Use this only when you need fine-grained control.

### 1) Authentication and ADC

Revoke all active gcloud account auth:

```bash
gcloud auth revoke --all
```

Revoke application default credentials:

```bash
gcloud auth application-default revoke
```

Reset local gcloud config directory:

```bash
rm -rf ~/.config/gcloud
```

Re-initialize gcloud account/project config:

```bash
gcloud init
```

Create fresh ADC credentials using an OAuth desktop client JSON:

```bash
gcloud auth application-default login \
  --client-id-file="/path/to/oauth_client.json" \
  --scopes="https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/drive"
```

### 2) Enable Required APIs

Enable core services used by cluster provisioning, IAM, and build:

```bash
gcloud services enable \
  compute.googleapis.com \
  artifactregistry.googleapis.com \
  iam.googleapis.com \
  cloudbuild.googleapis.com \
  oslogin.googleapis.com \
  cloudresourcemanager.googleapis.com \
  --project="${GCP_PROJECT}"
```

### 3) SSH and OS Login

Generate SSH key pair for OS Login:

```bash
ssh-keygen -t rsa -f ~/.ssh/id_rsa_gcp -C "${GCP_USER_EMAIL}" -N ""
```

Register SSH public key for OS Login:

```bash
gcloud compute os-login ssh-keys add \
  --key-file ~/.ssh/id_rsa_gcp.pub \
  --project="${GCP_PROJECT}"
```

Enable OS Login project metadata:

```bash
gcloud compute project-info add-metadata \
  --metadata enable-oslogin=TRUE \
  --project="${GCP_PROJECT}"
```

### 4) Service Account and IAM Roles

Create service account for Ray autoscaling/node operations:

```bash
SA_EMAIL="${GCP_SA_NAME}@${GCP_PROJECT}.iam.gserviceaccount.com"

gcloud iam service-accounts create "${GCP_SA_NAME}" \
  --display-name="Ray Autoscaler SA" \
  --project="${GCP_PROJECT}"
```

Grant required roles to the service account:

```bash
for role in \
  roles/artifactregistry.reader \
  roles/logging.logWriter \
  roles/monitoring.metricWriter \
  roles/compute.instanceAdmin.v1 \
  roles/storage.admin
do
  gcloud projects add-iam-policy-binding "${GCP_PROJECT}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="${role}"
done
```

Grant recommended roles to your user account:

```bash
for role in \
  roles/compute.osAdminLogin \
  roles/compute.instanceAdmin.v1 \
  roles/compute.securityAdmin \
  roles/artifactregistry.reader \
  roles/storage.admin
do
  gcloud projects add-iam-policy-binding "${GCP_PROJECT}" \
    --member="user:${GCP_USER_EMAIL}" \
    --role="${role}"
done
```

Allow your user to impersonate the service account:

```bash
gcloud iam service-accounts add-iam-policy-binding "${SA_EMAIL}" \
  --member="user:${GCP_USER_EMAIL}" \
  --role="roles/iam.serviceAccountUser" \
  --project="${GCP_PROJECT}"
```

### 5) Network, Firewall, and NAT

Set base network variables:

```bash
NETWORK="${NETWORK:-default}"
SUBNET="${SUBNET:-default}"
ROUTER="${ROUTER:-${NETWORK}-router}"
NAT="${NAT:-${NETWORK}-nat}"
```

Get subnet CIDR dynamically:

```bash
SUBNET_RANGE="$(
  gcloud compute networks subnets describe "${SUBNET}" \
    --project="${GCP_PROJECT}" \
    --region="${GCP_REGION}" \
    --format='value(ipCidrRange)'
)"
```

Allow internal node-to-node communication:

```bash
gcloud compute firewall-rules create ray-allow-internal \
  --project="${GCP_PROJECT}" \
  --network="${NETWORK}" \
  --direction=INGRESS \
  --priority=1000 \
  --action=ALLOW \
  --rules=tcp,udp,icmp \
  --source-ranges="${SUBNET_RANGE}" \
  --target-tags=ray-node
```

Allow head node access from your current IPv4:

```bash
MYIP4="$(curl -4 -s https://ifconfig.me)"
gcloud compute firewall-rules create ray-allow-head-from-home-v4 \
  --project="${GCP_PROJECT}" \
  --network="${NETWORK}" \
  --direction=INGRESS \
  --priority=1000 \
  --action=ALLOW \
  --rules=tcp:22,tcp:8265,tcp:10001 \
  --source-ranges="${MYIP4}/32" \
  --target-tags=ray-head
```

Allow IAP SSH access to head nodes:

```bash
gcloud compute firewall-rules create ray-allow-iap-ssh \
  --project="${GCP_PROJECT}" \
  --network="${NETWORK}" \
  --direction=INGRESS \
  --priority=900 \
  --action=ALLOW \
  --rules=tcp:22 \
  --source-ranges=35.235.240.0/20 \
  --target-tags=ray-head
```

Create Cloud Router:

```bash
gcloud compute routers create "${ROUTER}" \
  --project="${GCP_PROJECT}" \
  --network="${NETWORK}" \
  --region="${GCP_REGION}"
```

Create Cloud NAT for private workers:

```bash
gcloud compute routers nats create "${NAT}" \
  --project="${GCP_PROJECT}" \
  --router="${ROUTER}" \
  --region="${GCP_REGION}" \
  --auto-allocate-nat-external-ips \
  --nat-all-subnet-ip-ranges \
  --enable-logging
```

### 6) Artifact Registry and Bucket

Create Docker artifact registry:

```bash
gcloud artifacts repositories create "${GCP_REPO}" \
  --repository-format=docker \
  --location="${GCP_REGION}" \
  --description="Docker repository for Gomoku images" \
  --project="${GCP_PROJECT}"
```

Create storage bucket:

```bash
gcloud storage buckets create "gs://${GCP_BUCKET_NAME}" \
  --location="${GCP_REGION}" \
  --default-storage-class=STANDARD \
  --project="${GCP_PROJECT}"
```

### 7) Build and Push Images

Create and bootstrap a buildx builder:

```bash
docker buildx create --use --name mybuilder --driver docker-container
docker buildx inspect --bootstrap
```

Build and push the GPU image:

```bash
docker buildx build \
  --platform linux/amd64 \
  --output "type=image,compression=zstd,force-compression=true,push=true" \
  -t "${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${GCP_REPO}/gmk-ray:${GCP_GPU_TAG}" \
  -f "alphazero/infra/image/Dockerfile.py313-cu124" \
  "alphazero"
```

Build and push the CPU image:

```bash
docker buildx build \
  --platform linux/amd64 \
  --output "type=image,compression=zstd,force-compression=true,push=true" \
  -t "${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${GCP_REPO}/gmk-ray:${GCP_CPU_TAG}" \
  -f "alphazero/infra/image/Dockerfile.py313-cpu" \
  "alphazero"
```

Delete all existing image tags in the artifact repository:

```bash
REPO_PATH="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${GCP_REPO}"
gcloud artifacts docker images list "${REPO_PATH}" --format="value(package)" \
  | sort -u \
  | xargs -r -I {} gcloud artifacts docker images delete {} --quiet --delete-tags
```

### 8) Reservations and Cluster Operations

Create reservation manually:

```bash
RES="${GCP_HEAD_RESERVATION}"
MT="${MT:-g2-standard-16}"
ACC_TYPE="${ACC_TYPE:-nvidia-l4}"
ACC_PER_VM="${ACC_PER_VM:-1}"
VM_COUNT="${VM_COUNT:-1}"

gcloud compute reservations create "${RES}" \
  --zone="${GCP_ZONE}" \
  --project="${GCP_PROJECT}" \
  --machine-type="${MT}" \
  --accelerator="type=${ACC_TYPE},count=${ACC_PER_VM}" \
  --vm-count="${VM_COUNT}" \
  --require-specific-reservation
```

Describe reservation:

```bash
gcloud compute reservations describe "${GCP_HEAD_RESERVATION}" \
  --zone="${GCP_ZONE}" \
  --project="${GCP_PROJECT}"
```

Delete reservation:

```bash
gcloud compute reservations delete "${GCP_HEAD_RESERVATION}" \
  --zone="${GCP_ZONE}" \
  --project="${GCP_PROJECT}" \
  --quiet
```

Deploy/update cluster using rendered config:

```bash
bash alphazero/infra/cluster/restart_cluster.sh
```

Run `ray up` directly on the resolved cluster file:

```bash
ray up -y alphazero/infra/cluster/.cluster_elo1800.resolved.yaml --no-config-cache
```

Delete all cluster instances by cluster-name prefix:

```bash
gcloud compute instances delete \
  $(gcloud compute instances list \
    --project="${GCP_PROJECT}" \
    --filter="name~^ray-${GCP_CLUSTER_NAME}-" \
    --format="value(name)") \
  --zone="${GCP_ZONE}" \
  --project="${GCP_PROJECT}" \
  --quiet
```

### 9) Quota Checks

Check global GPU quota:

```bash
gcloud compute project-info describe \
  --project="${GCP_PROJECT}" \
  --format="value(quotas.metric.GPUS_ALL_REGIONS.limit)"
```

Check regional CPU/GPU quotas:

```bash
gcloud compute regions describe "${GCP_REGION}" \
  --project="${GCP_PROJECT}" \
  --format="table[box](quotas.metric,quotas.limit,quotas.usage)" \
  | grep -E "CPUS|GPUS"
```

## Safety Notes

- `rm -rf ~/.config/gcloud` resets your local gcloud state entirely.
- `gcloud artifacts docker images delete ... --delete-tags` removes registry images permanently.
- `gcloud compute instances delete ...` removes running VMs.
- Prefer script-based commands in `alphazero/infra` for repeatable operations.
