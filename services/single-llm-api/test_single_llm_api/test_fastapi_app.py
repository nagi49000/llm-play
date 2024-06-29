import pytest
from fastapi.testclient import TestClient
from single_llm_api.fastapi_app import create_app, get_llm_pipeline_from_yaml


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_get_healthcheck(client):
    resp = client.get("/healthcheck")
    assert resp.status_code == 200
    assert resp.json() == {"status": "OK"}


def test_get_version(client, monkeypatch):
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


def test_get_pipeline_spec(client):
    resp = client.get("/pipeline-spec")
    assert resp.status_code == 200
    pipeline_spec = resp.json()
    # sanity check some keys in the response
    assert set(pipeline_spec.keys()) == {"pipeline"}
    assert set(pipeline_spec["pipeline"].keys()) == {"pipeline-type", "kwargs", "model", "tokenizer"}


def test_get_llm_pipeline_from_yaml():
    llm = get_llm_pipeline_from_yaml()
    response = llm.invoke("what time is it")
    assert response.startswith("what time is it")


def test_post_invoke(client):
    resp = client.post("/invoke", json={"query": "what time is it"})
    assert resp.status_code == 200
    assert resp.json()["response"].startswith("what time is it")
