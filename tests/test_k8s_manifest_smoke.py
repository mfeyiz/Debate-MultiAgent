"""Smoke tests for Kubernetes manifests used in CI deployments."""

from pathlib import Path


def test_kubernetes_manifests_contains_domain() -> None:
    manifest_path = Path("kubernetes-manifests.yaml")
    assert manifest_path.exists(), "kubernetes-manifests.yaml must exist in repo root"

    contents = manifest_path.read_text(encoding="utf-8")
    assert "chat.mammas.studio" in contents
    assert "ManagedCertificate" in contents
    assert "debate-ssl-cert" in contents
