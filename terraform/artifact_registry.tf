resource "google_artifact_registry_repository" "app" {
  location      = var.region
  repository_id = "logicflow"
  format        = "DOCKER"
  description   = "LogicFlow application images"
}
