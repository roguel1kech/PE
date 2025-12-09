from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_sentiment_api():
    response = client.post("/predict", json={"text": "я счастлив"})
    assert response.status_code == 200
    assert "label" in response.json()
