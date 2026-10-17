"""Fast unit tests that mock the costly ML/network dependencies."""
import importlib
import sys
import types

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api(monkeypatch):
    fake = types.ModuleType("scripts.rag_pipeline")
    fake.retrieve_chunks = lambda query: (["AI is a field of computer science"], [0.3])
    fake.check_relevance = lambda distances: bool(distances) and min(distances) <= 0.8
    fake.generate_answer = lambda query, docs: "AI is a field of computer science"
    fake.hallucination_check = lambda answer, docs: True
    monkeypatch.setitem(sys.modules, "scripts.rag_pipeline", fake)
    sys.modules.pop("api.main", None)
    mod = importlib.import_module("api.main")
    with TestClient(mod.app, raise_server_exceptions=False) as client:
        yield client, mod, fake
    sys.modules.pop("api.main", None)


def test_home(api):
    client, _, _ = api
    assert client.get("/").status_code == 200


def test_health(api):
    client, _, _ = api
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"
    assert "X-Request-ID" in resp.headers


def test_ask_success(api):
    client, _, _ = api
    resp = client.post("/ask", json={"question": "What is AI?"})
    assert resp.status_code == 200
    assert "AI is a field" in resp.json()["answer"]
    assert isinstance(resp.json()["sources"], list)
    assert resp.json()["latency_ms"] >= 0


def test_ask_blank_returns_400(api):
    client, _, _ = api
    resp = client.post("/ask", json={"question": "   "})
    assert resp.status_code == 400


def test_missing_question_returns_422(api):
    client, _, _ = api
    assert client.post("/ask", json={}).status_code == 422


def test_too_long_returns_422(api):
    client, _, _ = api
    assert client.post("/ask", json={"question": "x" * 2001}).status_code == 422


def test_out_of_domain_refusal(api):
    client, mod, fake = api
    mod.check_relevance = lambda distances: False
    resp = client.post("/ask", json={"question": "unknown"})
    assert resp.status_code == 200
    assert "don't have enough information" in resp.json()["answer"]
    assert resp.json()["sources"] == []


def test_retrieval_failure_returns_500(api):
    client, mod, fake = api
    def failure(query):
        raise RuntimeError("database unavailable")
    mod.retrieve_chunks = failure
    resp = client.post("/ask", json={"question": "What is AI?"})
    assert resp.status_code == 500
    assert resp.json()["detail"] == "Internal RAG processing error"
