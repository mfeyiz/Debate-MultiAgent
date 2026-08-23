from fastapi.testclient import TestClient
from app.services.bert_service import ModernBERTPipeline
from main import app


def test_compare_models_direct():
    """Test real-only compare execution in ModernBERTPipeline."""
    pipeline = ModernBERTPipeline()
    res = pipeline.compare_models(
        source_text="Çünkü yapay zeka, insanların saatler süren tasarım süreçlerini saniyeler içinde tamamlayabiliyor.",
        target_text="Yapay zeka sistemleri insan yaratıcılığının yerini tamamen alacaktır."
    )
    
    assert "target_text" in res
    assert "source_text" in res
    assert "models" in res
    assert "model" in res
    assert "tokens_target" in res
    assert "tokens_source" in res
    assert "components" in res
    assert "relations" in res
    assert "overall_strength" in res
    assert "feedback" in res
    assert "warnings" in res
    assert all(c["component_type"] in {"claim", "premise", "evidence", "other"} for c in res["components"])
    assert all(r["relation_type"] in {"attack", "support", "none"} for r in res["relations"])


def test_compare_models_api():
    """Test /api/modernbert/compare POST endpoint."""
    with TestClient(app) as client:
        payload = {
            "target_text": "Yapay zeka sistemleri insan yaratıcılığının yerini tamamen alacaktır.",
            "source_text": "Çünkü yapay zeka, insanların saatler süren tasarım süreçlerini saniyeler içinde tamamlayabiliyor."
        }
        response = client.post("/api/modernbert/compare", json=payload)
        assert response.status_code == 200
        
        res = response.json()
        assert "target_text" in res
        assert "source_text" in res
        assert "models" in res
        assert "model" in res
        assert "components" in res
        assert "relations" in res
