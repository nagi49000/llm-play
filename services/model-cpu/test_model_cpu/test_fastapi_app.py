import pytest
from fastapi.testclient import TestClient
from model_cpu.fastapi_app import create_app, get_text_generator_llm


@pytest.fixture
def client():
    app = create_app(None, "pytest-app-name")
    return TestClient(app)


def test_healthcheck(client):
    resp = client.get("/healthcheck")
    assert resp.status_code == 200
    assert resp.json() == {"status": "OK"}


def test_version(client, monkeypatch):
    monkeypatch.setenv("APP_NAME", "pytest-app-name")
    monkeypatch.setenv("GITHASH", "abcd1234")
    monkeypatch.setenv("BUILD_TIME", "2024-06-22T20:16:21+00:00")
    resp = client.get("/version")
    assert resp.status_code == 200
    assert resp.json() == {
        "app_name": "self contained CPU model",
        "build_time": "2024-06-22T20:16:21Z",
        "githash": "abcd1234",
    }
