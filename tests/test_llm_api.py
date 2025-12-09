from fastapi.testclient import TestClient
from app.llm_api import app

client = TestClient(app)

def test_llm_api():
    response = client.post("/generate", json={"prompt": "Привет"})
    assert response.status_code == 200
