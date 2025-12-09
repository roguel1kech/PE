from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_ok():
    resp = client.post("/predict", json={"text": "я счастлив"})
    assert resp.status_code == 200
    data = resp.json()
    assert "label" in data
    assert "score" in data

def test_predict_empty():
    resp = client.post("/predict", json={"text": ""})
    assert resp.status_code == 200
    data = resp.json()
    assert "label" in data
