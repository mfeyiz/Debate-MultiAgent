import pytest
from fastapi.testclient import TestClient
from app.services.bert_service import ModernBERTPipeline
from main import app


def test_compare_models_direct():
    """Test compare_models direct execution in ModernBERTPipeline."""
    pipeline = ModernBERTPipeline()
    res = pipeline.compare_models(
        source_text="Çünkü yapay zeka, insanların saatler süren tasarım süreçlerini saniyeler içinde tamamlayabiliyor.",
        target_text="Yapay zeka sistemleri insan yaratıcılığının yerini tamamen alacaktır."
    )
    
    assert "target_text" in res
    assert "source_text" in res
    assert "models" in res
    
    models = res["models"]
    for model_key in ["model_1", "model_2", "model_3"]:
        assert model_key in models
        model = models[model_key]
        assert "name" in model
        assert "description" in model
        assert "tokens_target" in model
        assert "tokens_source" in model
        assert "components" in model
        assert "relations" in model
        assert "overall_strength" in model
        assert "feedback" in model
        assert "f1_components" in model
        assert "f1_relations" in model


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
        
        models = res["models"]
        assert "model_1" in models
        assert "model_2" in models
        assert "model_3" in models
