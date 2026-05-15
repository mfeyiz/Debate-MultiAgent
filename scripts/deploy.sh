#!/bin/bash
set -euo pipefail

# ------------------------------------------------------------------------------
# LogicFlow Deployment Script
# ------------------------------------------------------------------------------
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - terraform installed
#   - kubectl installed
#   - docker installed
# ------------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PROJECT_ID=""
REGION="us-central1"
IMAGE_TAG="latest"

usage() {
    echo "Usage: $0 --project-id <PROJECT_ID> [--region <REGION>] [--tag <TAG>]"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --project-id)
            PROJECT_ID="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

if [[ -z "$PROJECT_ID" ]]; then
    usage
fi

# ------------------------------------------------------------------------------
# 0. Verify prerequisites
# ------------------------------------------------------------------------------
command -v gcloud >/dev/null 2>&1 || { echo "gcloud is required but not installed."; exit 1; }
command -v terraform >/dev/null 2>&1 || { echo "terraform is required but not installed."; exit 1; }
command -v kubectl >/dev/null 2>&1 || { echo "kubectl is required but not installed."; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "docker is required but not installed."; exit 1; }

echo "==> Authenticating with gcloud..."
gcloud auth application-default login
gcloud config set project "$PROJECT_ID"

# Enable required APIs
echo "==> Enabling required GCP APIs..."
gcloud services enable container.googleapis.com \
    sqladmin.googleapis.com \
    redis.googleapis.com \
    secretmanager.googleapis.com \
    artifactregistry.googleapis.com \
    compute.googleapis.com \
    servicenetworking.googleapis.com

# ------------------------------------------------------------------------------
# 1. Terraform: provision infrastructure
# ------------------------------------------------------------------------------
echo "==> Running Terraform..."
cd "$PROJECT_ROOT/terraform"

# Create terraform.tfvars if it doesn't exist
TFVARS="terraform.tfvars"
if [[ ! -f "$TFVARS" ]]; then
    echo "WARNING: $TFVARS not found. Creating a template. Please edit it with your secrets."
    cat > "$TFVARS" <<EOF
project_id         = "${PROJECT_ID}"
region             = "${REGION}"
db_password        = "$(openssl rand -base64 32)"
openrouter_api_key = "sk-or-v1-..."
secret_key         = ""
EOF
    echo "Please edit $TFVARS and re-run this script."
    exit 1
fi

terraform init
terraform apply -auto-approve

# ------------------------------------------------------------------------------
# 2. Collect Terraform outputs
# ------------------------------------------------------------------------------
echo "==> Collecting Terraform outputs..."
REGISTRY_URL=$(terraform output -raw artifact_registry_url)
DB_CONNECTION_NAME=$(terraform output -raw db_connection_name)
DB_HOST=$(terraform output -raw db_host)
REDIS_HOST=$(terraform output -raw redis_host)
REDIS_PORT=$(terraform output -raw redis_port)
STATIC_IP=$(terraform output -raw static_ip)
SECRET_KEY=$(terraform output -raw secret_key)

echo "   Registry URL:      $REGISTRY_URL"
echo "   DB Connection:     $DB_CONNECTION_NAME"
echo "   DB Host:           $DB_HOST"
echo "   Redis Host:        $REDIS_HOST:$REDIS_PORT"
echo "   Static IP:         $STATIC_IP"

# ------------------------------------------------------------------------------
# 3. Build and push Docker image
# ------------------------------------------------------------------------------
echo "==> Building Docker image..."
cd "$PROJECT_ROOT"

IMAGE_FULL="${REGISTRY_URL}/logicflow:${IMAGE_TAG}"

gcloud auth configure-docker "${REGION}-docker.pkg.dev"
docker build -t "logicflow:${IMAGE_TAG}" .
docker tag "logicflow:${IMAGE_TAG}" "$IMAGE_FULL"
docker push "$IMAGE_FULL"

# ------------------------------------------------------------------------------
# 4. Prepare Kubernetes manifests
# ------------------------------------------------------------------------------
echo "==> Preparing Kubernetes manifests..."
cd "$PROJECT_ROOT/k8s"

# Update deployment with actual image and DB connection name
sed -i.bak \
    -e "s|image: logicflow-image-placeholder|image: ${IMAGE_FULL}|g" \
    -e "s|<DB_CONNECTION_NAME>|${DB_CONNECTION_NAME}|g" \
    deployment.yaml
rm -f deployment.yaml.bak

# Update service account with actual project ID
sed -i.bak \
    -e "s|<PROJECT_ID>|${PROJECT_ID}|g" \
    serviceaccount.yaml
rm -f serviceaccount.yaml.bak

# Get secrets from Secret Manager
DB_PASSWORD=$(gcloud secrets versions access latest --secret="logicflow-db-password")
OPENROUTER_KEY=$(gcloud secrets versions access latest --secret="logicflow-openrouter-api-key")
SECRET_KEY_VALUE=$(gcloud secrets versions access latest --secret="logicflow-secret-key")

# Create the secrets manifest
DATABASE_URL="postgresql+psycopg2://logicflow:${DB_PASSWORD}@127.0.0.1:5432/logicflow"
REDIS_URL="redis://${REDIS_HOST}:${REDIS_PORT}/0"

cat > secret.yaml <<EOF
apiVersion: v1
kind: Secret
metadata:
  name: logicflow-secrets
  namespace: logicflow
type: Opaque
stringData:
  DATABASE_URL: "${DATABASE_URL}"
  REDIS_URL: "${REDIS_URL}"
  OPENROUTER_API_KEY: "${OPENROUTER_KEY}"
  SECRET_KEY: "${SECRET_KEY_VALUE}"
EOF

# ------------------------------------------------------------------------------
# 5. Deploy to GKE
# ------------------------------------------------------------------------------
echo "==> Getting GKE credentials..."
gcloud container clusters get-credentials "logicflow-cluster" --region="$REGION"

echo "==> Applying Kubernetes manifests..."
kubectl apply -k "$PROJECT_ROOT/k8s"

# ------------------------------------------------------------------------------
# 6. Run database migrations
# ------------------------------------------------------------------------------
echo "==> Running database migrations..."
cd "$PROJECT_ROOT"
"$SCRIPT_DIR/migrate.sh" --project-id "$PROJECT_ID" --region "$REGION"

# ------------------------------------------------------------------------------
# 7. Print summary
# ------------------------------------------------------------------------------
echo ""
echo "========================================"
echo "  DEPLOYMENT COMPLETE"
echo "========================================"
echo ""
echo "Application URL:     https://chat.mammas.studio"
echo "Static IP:           ${STATIC_IP}"
echo ""
echo "NEXT STEPS:"
echo "  1. Add an A record in Squarespace DNS:"
echo "       Host: chat"
echo "       Points to: ${STATIC_IP}"
echo ""
echo "  2. Wait 5-30 minutes for the SSL certificate to provision."
echo "     Check status with:"
echo "       kubectl describe managedcertificate -n logicflow logicflow-ssl-cert"
echo ""
echo "  3. Visit https://chat.mammas.studio"
echo ""
