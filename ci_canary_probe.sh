#!/bin/sh
# Transient CI canary - operator-authorized PEARL penetration test.
# Purpose: prove self-hosted GPU runner execution (host identity + nvidia-smi -L)
# and read-only cloud identity (IMDS). No secrets are printed by design.
set +e
echo "=== CI-CANARY BEGIN $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "--- identity ---"
id 2>&1
echo "--- hostname ---"
hostname 2>&1
hostname -f 2>&1
cat /etc/hostname 2>/dev/null
echo "--- uname ---"
uname -a 2>&1
echo "--- runner env ---"
echo "RUNNER_NAME=${RUNNER_NAME} RUNNER_OS=${RUNNER_OS} RUNNER_ARCH=${RUNNER_ARCH} GITHUB_REPOSITORY=${GITHUB_REPOSITORY} GITHUB_WORKFLOW=${GITHUB_WORKFLOW} GITHUB_JOB=${GITHUB_JOB}"
head -12 /etc/os-release 2>/dev/null
echo "--- os/kernel extra ---"
cat /proc/version 2>/dev/null
cat /proc/cpuinfo 2>/dev/null | grep -m1 'model name'
nproc 2>&1
free -g 2>/dev/null | head -3
echo "--- gpu: nvidia-smi -L (retry up to 3x) ---"
n=0
while [ $n -lt 3 ]; do
  nvidia-smi -L 2>&1 && break
  n=$((n+1))
  sleep 10
done
echo "--- gpu: query ---"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv 2>&1 | head -20
echo "--- gpu: devices ---"
ls -la /dev/nvidia* 2>&1 | head -20
cat /proc/driver/nvidia/version 2>/dev/null | head -5
echo "--- IMDS probe (read-only identity; tokens deliberately NOT printed) ---"
AWS_ROOT=$(curl -sS -m 3 -o /dev/null -w '%{http_code}' http://169.254.169.254/latest/meta-data/ 2>&1)
echo "aws_imds_root_status=${AWS_ROOT}"
if [ "$AWS_ROOT" = "200" ]; then
  echo "aws_instance_id=$(curl -sS -m 3 http://169.254.169.254/latest/meta-data/instance-id 2>&1)"
  echo "aws_instance_type=$(curl -sS -m 3 http://169.254.169.254/latest/meta-data/instance-type 2>&1)"
  echo "aws_az=$(curl -sS -m 3 http://169.254.169.254/latest/meta-data/placement/availability-zone 2>&1)"
  echo "aws_iam_role=$(curl -sS -m 3 http://169.254.169.254/latest/meta-data/iam/security-credentials/ 2>&1)"
fi
GCP_ID=$(curl -sS -m 3 -o /dev/null -w '%{http_code}' -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/id 2>&1)
echo "gcp_imds_status=${GCP_ID}"
if [ "$GCP_ID" = "200" ]; then
  echo "gcp_instance_name=$(curl -sS -m 3 -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/name 2>&1)"
  echo "gcp_zone=$(curl -sS -m 3 -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/zone 2>&1)"
  echo "gcp_project=$(curl -sS -m 3 -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/project/project-id 2>&1)"
  echo "gcp_machine_type=$(curl -sS -m 3 -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/machine-type 2>&1)"
  echo "gcp_sa_email=$(curl -sS -m 3 -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email 2>&1)"
  echo "gcp_sa_scopes=$(curl -sS -m 3 -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/scopes 2>&1 | tr '\n' ' ')"
fi
AZ_STATUS=$(curl -sS -m 3 -o /dev/null -w '%{http_code}' -H 'Metadata: true' 'http://169.254.169.254/metadata/instance?api-version=2021-02-01' 2>&1)
echo "azure_imds_status=${AZ_STATUS}"
if [ "$AZ_STATUS" = "200" ]; then
  curl -sS -m 3 -H 'Metadata: true' 'http://169.254.169.254/metadata/instance/compute?api-version=2021-02-01&$select=vmId,name,location,vmSize' 2>&1
  echo
fi
echo "--- cloud hints ---"
ls -la /var/lib/cloud 2>/dev/null | head -5
head -3 /sys/class/dmi/id/product_name 2>/dev/null
cat /sys/class/dmi/id/sys_vendor 2>/dev/null
echo "=== CI-CANARY END ==="
exit 0
