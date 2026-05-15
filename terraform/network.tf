resource "google_compute_network" "vpc" {
  name                    = "logicflow-vpc"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "subnet" {
  name          = "logicflow-subnet"
  ip_cidr_range = "10.0.0.0/24"
  region        = var.region
  network       = google_compute_network.vpc.id

  private_ip_google_access = true
}

resource "google_compute_global_address" "ingress_ip" {
  name = "logicflow-ingress-ip"
}
