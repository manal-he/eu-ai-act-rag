from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from eu_ai_act_rag.api.main import create_app
from eu_ai_act_rag.config import Settings
from eu_ai_act_rag.pipeline import RAGService


@pytest.fixture(scope="module")
def client(offline_settings: Settings, service: RAGService) -> TestClient:
    with TestClient(create_app(offline_settings, service)) as c:
        yield c  # type: ignore[misc]


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["llm"] == "mock" and body["chunks"] > 0
    assert "not legal advice" in body["disclaimer"]


def test_search(client: TestClient) -> None:
    r = client.post("/search", json={"query": "right to erasure", "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["language"] == "en" and len(body["hits"]) == 3
    assert "gdpr:art_17" in [h["chunk"]["unit_id"] for h in body["hits"]]


def test_search_validation(client: TestClient) -> None:
    assert client.post("/search", json={"query": "x"}).status_code == 422


def test_ask(client: TestClient) -> None:
    r = client.post("/ask", json={"question": "Is social scoring prohibited by the AI Act?"})
    assert r.status_code == 200
    body = r.json()
    assert body["citations"] and not body["abstained"]
    assert body["citations"][0]["unit_id"].startswith("ai_act:")


def test_ask_stream_sse(client: TestClient) -> None:
    with client.stream(
        "POST", "/ask/stream", json={"question": "What are the GDPR principles?"}
    ) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        raw = "".join(r.iter_text())
    events = [blk for blk in raw.split("\n\n") if blk.strip()]
    kinds = [e.split("\n")[0].removeprefix("event: ") for e in events]
    assert kinds[0] == "retrieval" and kinds[-1] == "final" and "token" in kinds
    final = json.loads(events[-1].split("\n", 1)[1].removeprefix("data: "))
    assert "citations" in final and "latency_ms" in final


def test_unit_lookup(client: TestClient) -> None:
    r = client.get("/units/gdpr:art_17", params={"lang": "fr"})
    assert r.status_code == 200 and r.json()["lang"] == "fr"
    assert client.get("/units/gdpr:art_999").status_code == 404
