resource "google_secret_manager_secret" "db_password" {
  secret_id = "logicflow-db-password"

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "db_password" {
  secret      = google_secret_manager_secret.db_password.id
  secret_data = var.db_password
}

resource "google_secret_manager_secret" "openrouter_api_key" {
  secret_id = "logicflow-openrouter-api-key"

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "openrouter_api_key" {
  secret      = google_secret_manager_secret.openrouter_api_key.id
  secret_data = var.openrouter_api_key
}

resource "google_secret_manager_secret" "secret_key" {
  secret_id = "logicflow-secret-key"

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "secret_key" {
  secret      = google_secret_manager_secret.secret_key.id
  secret_data = coalesce(var.secret_key, random_password.secret_key.result)
}

resource "random_password" "secret_key" {
  length  = 32
  special = false
}
