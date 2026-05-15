output "cluster_endpoint" {
  description = "GKE cluster endpoint"
  value       = google_container_cluster.primary.endpoint
  sensitive   = true
}

output "cluster_name" {
  description = "GKE cluster name"
  value       = google_container_cluster.primary.name
}

output "db_connection_name" {
  description = "Cloud SQL connection name for the proxy"
  value       = google_sql_database_instance.main.connection_name
}

output "db_host" {
  description = "Cloud SQL private IP"
  value       = google_sql_database_instance.main.private_ip_address
}

output "redis_host" {
  description = "Redis instance host"
  value       = google_redis_instance.main.host
}

output "redis_port" {
  description = "Redis instance port"
  value       = google_redis_instance.main.port
}

output "artifact_registry_url" {
  description = "Artifact Registry repository URL"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.app.repository_id}"
}

output "static_ip" {
  description = "Global static IP for the ingress"
  value       = google_compute_global_address.ingress_ip.address
}

output "secret_key" {
  description = "Generated Flask SECRET_KEY (if not provided)"
  value       = random_password.secret_key.result
  sensitive   = true
}
