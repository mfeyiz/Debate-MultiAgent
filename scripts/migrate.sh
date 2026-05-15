#!/bin/bash
set -euo pipefail

# ------------------------------------------------------------------------------
# Run database migrations as a one-off Kubernetes Job
# ------------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PROJECT_ID=""
REGION="us-central1"

usage() {
    echo "Usage: $0 --project-id <PROJECT_ID> [--region <REGION>]"
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
        *)
            usage
            ;;
    esac
done

if [[ -z "$PROJECT_ID" ]]; then
    usage
fi

gcloud config set project "$PROJECT_ID"

# Get GKE credentials
gcloud container clusters get-credentials "logicflow-cluster" --region="$REGION"

# Get the current app image from the deployment
IMAGE=$(kubectl get deployment logicflow-app -n logicflow -o jsonpath='{.spec.template.spec.containers[?(@.name=="app")].image}')

if [[ -z "$IMAGE" ]]; then
    echo "ERROR: Could not get image from deployment. Is the app deployed?"
    exit 1
fi

echo "==> Running migrations with image: $IMAGE"

# Get secrets for the migration job
DB_PASSWORD=$(gcloud secrets versions access latest --secret="logicflow-db-password")
REDIS_HOST=$(gcloud redis instances describe logicflow-redis --region="$REGION" --format='value(host)')
REDIS_PORT=$(gcloud redis instances describe logicflow-redis --region="$REGION" --format='value(port)')
SECRET_KEY=$(gcloud secrets versions access latest --secret="logicflow-secret-key")
OPENROUTER_KEY=$(gcloud secrets versions access latest --secret="logicflow-openrouter-api-key")
DB_CONNECTION_NAME=$(gcloud sql instances describe logicflow-postgres --format='value(connectionName)')

DATABASE_URL="postgresql+psycopg2://logicflow:${DB_PASSWORD}@127.0.0.1:5432/logicflow"
REDIS_URL="redis://${REDIS_HOST}:${REDIS_PORT}/0"

# Create and run a migration job
cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: logicflow-migrate-$(date +%s)
  namespace: logicflow
spec:
  ttlSecondsAfterFinished: 3600
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: migrate
          image: ${IMAGE}
          command: ["flask", "db", "upgrade"]
          env:
            - name: DATABASE_URL
              value: "${DATABASE_URL}"
            - name: REDIS_URL
              value: "${REDIS_URL}"
            - name: SECRET_KEY
              value: "${SECRET_KEY}"
            - name: OPENROUTER_API_KEY
              value: "${OPENROUTER_KEY}"
            - name: FLASK_APP
              value: "main.py"
          envFrom:
            - configMapRef:
                name: logicflow-config
        - name: cloud-sql-proxy
          image: gcr.io/cloud-sql-connectors/cloud-sql-proxy:2.8.0
          args:
            - "--structured-logs"
            - "--port=5432"
            - "${DB_CONNECTION_NAME}"
          securityContext:
            runAsNonRoot: true
EOF

echo "==> Migration job submitted. Monitor with:"
echo "    kubectl get jobs -n logicflow"
echo "    kubectl logs -n logicflow -l job-name=logicflow-migrate-..."
