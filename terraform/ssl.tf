resource "google_compute_managed_ssl_certificate" "default" {
  name = "logicflow-ssl-cert"

  managed {
    domains = [var.custom_domain]
  }
}
