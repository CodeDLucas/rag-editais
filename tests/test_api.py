"""Testes de integração dos endpoints FastAPI com mocks do Ollama."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app import main as main_module
from app.config import Settings
from tests.test_ingestion import DeterministicFakeEmbeddings, _make_blank_pdf, _make_sample_edital_pdf


def _make_fake_llm(reply_text: str) -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content=reply_text)

    async def _fake_astream(*args, **kwargs):
        for word in reply_text.split(" "):
            yield AIMessage(content=word + " ")

    llm.astream = _fake_astream
    return llm


def test_health_endpoints_states(monkeypatch) -> None:
    client = TestClient(main_module.app)

    # 1) Ollama online com todos os modelos configurados
    ok_resp = MagicMock()
    ok_resp.json.return_value = {
        "models": [{"name": "bge-m3:latest"}, {"name": "qwen2.5:3b"}]
    }
    ok_resp.raise_for_status = MagicMock()

    mock_async_client = AsyncMock()
    mock_async_client.__aenter__.return_value.get = AsyncMock(return_value=ok_resp)

    with patch("app.main.httpx.AsyncClient", return_value=mock_async_client):
        res = client.get("/api/health")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "ok"
        assert body["ollama_reachable"] is True
        assert body["embedding_model_available"] is True
        assert body["llm_model_available"] is True

    # 2) Ollama online mas sem o modelo de embeddings
    degraded_resp = MagicMock()
    degraded_resp.json.return_value = {"models": [{"name": "qwen2.5:3b"}]}
    degraded_resp.raise_for_status = MagicMock()
    mock_async_client.__aenter__.return_value.get = AsyncMock(return_value=degraded_resp)

    with patch("app.main.httpx.AsyncClient", return_value=mock_async_client):
        res = client.get("/api/health")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "degraded"
        assert body["embedding_model_available"] is False
        assert "bge-m3" in body["message"]

    # 3) Ollama offline
    mock_async_client.__aenter__.return_value.get = AsyncMock(
        side_effect=httpx.ConnectError("Connection refused")
    )
    with patch("app.main.httpx.AsyncClient", return_value=mock_async_client):
        res = client.get("/api/health")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "offline"
        assert body["ollama_reachable"] is False


def test_upload_validation_and_chat_flow(tmp_path: Path, monkeypatch) -> None:
    isolated_settings = Settings(
        chroma_persist_dir=str(tmp_path / "chroma_api"),
        chroma_collection_name="api_test_collection",
        chunk_size=300,
        chunk_overlap=50,
        default_top_k=2,
    )
    monkeypatch.setattr(main_module, "settings", isolated_settings)

    fake_emb = DeterministicFakeEmbeddings()
    fake_llm = _make_fake_llm(
        "O prazo de validade do concurso é de 2 anos [Página 1 — 1. DAS DISPOSIÇÕES PRELIMINARES]."
    )

    client = TestClient(main_module.app)

    # Sem documento indexado -> /api/document retorna null e /api/chat retorna 400
    doc_res = client.get("/api/document")
    assert doc_res.status_code == 200
    assert doc_res.json()["document"] is None

    chat_empty = client.post("/api/chat", json={"question": "Qual o prazo de validade?", "top_k": 2})
    assert chat_empty.status_code == 400

    # Rejeita arquivo que não é PDF
    bad_ext = client.post(
        "/api/upload",
        files={"file": ("edital.txt", b"conteudo qualquer", "text/plain")},
    )
    assert bad_ext.status_code == 400

    # Rejeita PDF vazio (sem texto)
    blank_pdf = client.post(
        "/api/upload",
        files={"file": ("vazio.pdf", _make_blank_pdf(), "application/pdf")},
    )
    assert blank_pdf.status_code == 400

    # Upload válido com embeddings fake
    with patch("app.ingestion.get_embeddings", return_value=fake_emb):
        upload_res = client.post(
            "/api/upload",
            files={"file": ("edital_2026.pdf", _make_sample_edital_pdf(), "application/pdf")},
        )
    assert upload_res.status_code == 200
    upload_data = upload_res.json()
    assert upload_data["status"] == "ok"
    assert upload_data["filename"] == "edital_2026.pdf"
    assert upload_data["total_pages"] == 2
    assert upload_data["total_chunks"] >= 2

    # /api/document agora retorna o edital ativo
    doc_res_after = client.get("/api/document")
    assert doc_res_after.status_code == 200
    assert doc_res_after.json()["document"]["filename"] == "edital_2026.pdf"

    # /api/chat (modo híbrido) retorna answer, sources, augmented_prompt e metrics
    with patch("app.ingestion.get_embeddings", return_value=fake_emb), patch(
        "app.rag_service.ChatOllama", return_value=fake_llm
    ):
        chat_res = client.post(
            "/api/chat",
            json={
                "question": "Qual o prazo de validade do concurso?",
                "top_k": 2,
                "search_mode": "hybrid",
            },
        )
    assert chat_res.status_code == 200
    chat_data = chat_res.json()
    assert "2 anos" in chat_data["answer"]
    assert 1 <= len(chat_data["sources"]) <= 2
    first_source = chat_data["sources"][0]
    assert {"content", "page", "section", "similarity_score", "lexical_score", "rrf_score"} <= set(
        first_source.keys()
    )
    assert 0.0 <= first_source["similarity_score"] <= 1.0
    assert "TRECHOS RECUPERADOS DO EDITAL" in chat_data["augmented_prompt"]
    assert "Qual o prazo de validade do concurso?" in chat_data["augmented_prompt"]
    assert chat_data["metrics"]["search_mode"] == "hybrid"
    assert chat_data["metrics"]["retrieval_ms"] >= 0

    # /api/chat/stream (SSE) transmite sources -> token -> done
    with patch("app.ingestion.get_embeddings", return_value=fake_emb), patch(
        "app.rag_service.ChatOllama", return_value=fake_llm
    ):
        stream_res = client.post(
            "/api/chat/stream",
            json={
                "question": "Qual o prazo de validade do concurso?",
                "top_k": 2,
                "search_mode": "vector",
            },
        )
    assert stream_res.status_code == 200
    assert "text/event-stream" in stream_res.headers["content-type"]
    assert "event: sources" in stream_res.text
    assert "event: token" in stream_res.text
    assert "event: done" in stream_res.text


def test_static_index_served() -> None:
    client = TestClient(main_module.app)
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "RAG Editais" in res.text
