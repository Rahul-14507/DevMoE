from fastapi.testclient import TestClient
from devmoe.server.app import create_app
from devmoe.config import AppConfig

app = create_app(AppConfig())
client = TestClient(app)


def test_models_endpoint():
    res = client.get("/v1/models")
    assert res.status_code == 200
    data = res.json()
    assert data["object"] == "list"
    ids = [m["id"] for m in data["data"]]
    assert "devmoe-auto" in ids
    assert "devmoe-local" in ids
    assert "gemini-1.5-flash" in ids


def test_status_endpoint():
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert "hardware" in data
    assert "local_engine" in data
    assert "providers" in data


def test_recommendations_endpoint():
    res = client.get("/api/recommendations")
    assert res.status_code == 200
    data = res.json()
    assert "recommendations" in data
    assert len(data["recommendations"]) > 0


def test_test_route_endpoint():
    payload = {"prompt": "Write a regex pattern for emails"}
    res = client.post("/api/test-route", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["tier"] == "easy"
    assert data["target_provider"] == "ollama_local"


def test_chat_completions_endpoint():
    payload = {
        "model": "devmoe-auto",
        "messages": [{"role": "user", "content": "Write a python function to add two numbers"}],
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert "_devmoe_meta" in data
