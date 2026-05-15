# LogicFlow Deployment Guide

This guide walks you through deploying LogicFlow to **Google Kubernetes Engine (GKE)** with a custom domain (`chat.mammas.studio`) and SSL.

## Architecture Overview

- **GKE Autopilot** cluster (managed nodes)
- **Cloud SQL PostgreSQL** (managed database)
- **Cloud Memorystore Redis** (managed cache for multi-pod SSE)
- **Artifact Registry** (Docker image storage)
- **Google-managed SSL certificate** (auto-renewed)
- **Squarespace DNS** (custom domain)

---

## Prerequisites

Install the following tools:

- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) (`gcloud`)
- [Terraform](https://developer.hashicorp.com/terraform/downloads) (`>= 1.5.0`)
- [kubectl](https://kubernetes.io/docs/tasks/tools/)
- [Docker](https://docs.docker.com/get-docker/)

Also ensure you have:

- A Google Cloud project with billing enabled
- Owner or Editor IAM role on the project
- The custom domain `mammas.studio` registered via Squarespace

---

## Step 1: Prepare the Terraform State Bucket

Terraform needs a GCS bucket to store its state. Create it once:

```bash
export PROJECT_ID="your-gcp-project-id"
gcloud config set project "$PROJECT_ID"

# Create the state bucket (must be globally unique)
gcloud storage buckets create gs://logicflow-terraform-state \
    --location=us-central1 \
    --uniform-bucket-level-access
```

If the bucket name `logicflow-terraform-state` is taken, pick a different unique name and update `terraform/backend.tf` accordingly.

---

## Step 2: Configure Terraform Variables

```bash
cd terraform
```

Create `terraform.tfvars`:

```hcl
project_id         = "your-gcp-project-id"
region             = "us-central1"
db_password        = "a-strong-random-password-32-chars"
openrouter_api_key = "sk-or-v1-..."
secret_key         = ""  # Leave empty to auto-generate
```

**Important:**
- `db_password`: Must be at least 8 characters. Use a strong random password.
- `openrouter_api_key`: Your actual OpenRouter API key.
- `secret_key`: Leave empty and Terraform will generate a secure 32-char random key.

---

## Step 3: Deploy Everything

Run the deployment script from the project root:

```bash
./scripts/deploy.sh --project-id "$PROJECT_ID"
```

This script will:
1. Enable all required GCP APIs
2. Run `terraform apply` to create the infrastructure
3. Build and push the Docker image to Artifact Registry
4. Update Kubernetes manifests with real values
5. Deploy to GKE
6. Run database migrations

The first deployment takes **15–25 minutes** (mostly Terraform provisioning Cloud SQL and GKE).

---

## Step 4: Configure Squarespace DNS

After the script finishes, it prints the **static IP address**. You need to add an A record in Squarespace:

1. Log into [Squarespace](https://account.squarespace.com/)
2. Go to **Domains** → `mammas.studio` → **DNS Settings**
3. Click **Custom Records** → **Add Record**
4. Add an **A record**:
   - **Host:** `chat`
   - **Points to:** `34.xxx.xxx.xxx` (the static IP from the script output)
5. Save

**Note:** If Google requests domain verification for the SSL certificate, you may need to add a **TXT record** as well. The exact values will be shown in the Google Cloud Console under **Network Services → Load Balancing**, or via:

```bash
kubectl describe managedcertificate -n logicflow logicflow-ssl-cert
```

---

## Step 5: Wait for SSL Certificate

Google-managed SSL certificates typically provision within **5–30 minutes** after DNS is configured.

Check the certificate status:

```bash
kubectl describe managedcertificate -n logicflow logicflow-ssl-cert
```

Look for `Certificate Status: Active`.

Once active, visit: **https://chat.mammas.studio**

---

## Step 6: Verify the Deployment

Check pod status:

```bash
kubectl get pods -n logicflow
```

Check logs:

```bash
kubectl logs -n logicflow -l app=logicflow -c app --tail=100
```

Test the health endpoints:

```bash
curl https://chat.mammas.studio/api/healthz
curl https://chat.mammas.studio/api/ready
```

---

## Updating the Application

After making code changes, rebuild and redeploy:

### Option A: Manual

```bash
# Build and push
docker build -t logicflow:latest .
docker tag logicflow:latest us-central1-docker.pkg.dev/$PROJECT_ID/logicflow/logicflow:latest
docker push us-central1-docker.pkg.dev/$PROJECT_ID/logicflow/logicflow:latest

# Update deployment
kubectl set image deployment/logicflow-app app=us-central1-docker.pkg.dev/$PROJECT_ID/logicflow/logicflow:latest -n logicflow
kubectl rollout status deployment/logicflow-app -n logicflow
```

### Option B: Cloud Build (CI/CD)

```bash
gcloud builds submit --config=cloudbuild.yaml
```

This triggers a Cloud Build pipeline that builds, pushes, and deploys automatically.

---

## Running Database Migrations

If you add new models or modify existing ones:

1. Generate a migration locally:

```bash
flask db migrate -m "description of changes"
```

2. Commit the generated migration file in `migrations/versions/`.

3. Apply to production:

```bash
./scripts/migrate.sh --project-id "$PROJECT_ID"
```

---

## Teardown

To destroy all infrastructure and avoid ongoing charges:

```bash
cd terraform
terraform destroy
```

**Warning:** This deletes the GKE cluster, Cloud SQL instance, Redis instance, and all data. The static IP and SSL certificate are also removed.

To keep the Terraform state bucket for future use, manually delete it after:

```bash
gcloud storage rm --recursive gs://logicflow-terraform-state
```

---

## Troubleshooting

### Pods stuck in `Pending`

GKE Autopilot may take a few minutes to provision nodes. Check:

```bash
kubectl get events -n logicflow --sort-by='.lastTimestamp'
```

### Database connection errors

Ensure the Cloud SQL Proxy sidecar is running:

```bash
kubectl logs -n logicflow -l app=logicflow -c cloud-sql-proxy --tail=50
```

### SSL certificate not provisioning

1. Verify DNS propagation:
   ```bash
   nslookup chat.mammas.studio
   ```
   It should resolve to the static IP.

2. Check certificate events:
   ```bash
   kubectl describe managedcertificate -n logicflow logicflow-ssl-cert
   ```

### SSE stream not working across pods

This is usually a Redis connectivity issue. Check:

```bash
kubectl logs -n logicflow -l app=logicflow -c app --tail=50 | grep -i redis
```

Ensure `REDIS_URL` is set correctly in the secret:

```bash
kubectl get secret logicflow-secrets -n logicflow -o jsonpath='{.data.REDIS_URL}' | base64 -d
```

### High memory usage / OOMKilled

The models use ~1.14 GB. If you see OOM kills, increase the memory limit in `k8s/deployment.yaml`:

```yaml
limits:
  memory: "6Gi"  # or higher
```

Then re-apply:

```bash
kubectl apply -k k8s/
```

---

## Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| **GKE Autopilot** | Pay per pod, not per node. Google manages patching, scaling, and node sizing. |
| **CPU-only torch** | Cheaper than GPU nodes. Image is ~1.3 GB vs ~4 GB. Easy to switch to CUDA later by changing the pip index. |
| **Models in Docker image** | 1.14 GB is acceptable. Instant pod startup. No GCS complexity. |
| **Cloud SQL + Redis** | Fully managed, zero state in pods = horizontal scaling + high availability. |
| **Google-managed SSL** | Free, auto-renewed via Let's Encrypt. No manual cert management. |
| **Flask-Migrate (Alembic)** | Production-safe database schema changes. Never use `db.create_all()` in production. |

---

## Next Steps / Optional Improvements

1. **Enable Cloud CDN** on the ingress for faster static asset delivery.
2. **Add Cloud Monitoring alerts** for pod restarts, high latency, or database connections.
3. **Switch to GPU nodes** if model inference becomes a bottleneck (change torch index in Dockerfile and add `nvidia.com/gpu` resource requests).
4. **Add a staging environment** by parameterizing the Terraform workspace.
