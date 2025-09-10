import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add app to path
sys.path.append(str(Path(__file__).parent.parent))
from app.main import app

client = TestClient(app)

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_info" in data

def test_predict_api():
    """Test prediction API endpoint"""
    test_data = {"text": "This is a great project with positive outcomes"}
    response = client.post("/api/predict", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert "sentiment" in data
    assert "confidence" in data
    assert data["sentiment"] in ["positive", "negative", "neutral"]
    assert 0 <= data["confidence"] <= 1

def test_web_interface():
    """Test web interface"""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_model_reload():
    """Test model reload endpoint"""
    response = client.post("/api/reload-models")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
