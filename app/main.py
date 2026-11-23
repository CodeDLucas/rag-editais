"""Aplicação FastAPI: endpoints de saúde, upload de PDF, consulta RAG e frontend estático."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from app.config import settings
from app.ingestion import (
    EmptyPDFError,
    InvalidPDFError,
    get_active_document,
    ingest_pdf,
)
from app.rag_service import (
    NoDocumentIndexedError,
    OllamaConnectionError,
    OllamaTimeoutError,
    RAGServiceError,
    ask_question,
)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(
    title="RAG Editais - Busca Vetorial",
    description="Aplicação acadêmica de RAG para consulta de editais de concursos públicos em PDF.",
    version="1.0.0",
)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "offline"]
    ollama_reachable: bool
    embedding_model: str
    embedding_model_available: bool
    llm_model: str
    llm_model_available: bool
    installed_models: list[str]
    message: str


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    total_pages: int
    total_chunks: int
    indexed_at: str


class DocumentStatusResponse(BaseModel):
    document: DocumentSummary | None


class UploadResponse(BaseModel):
    status: str
    document_id: str
    filename: str
    total_pages: int
    total_chunks: int


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1500)
    top_k: int = Field(default=settings.default_top_k, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        cleaned = value.strip()
        if len(cleaned) < 2:
            raise ValueError("A pergunta deve conter pelo menos 2 caracteres.")
        return cleaned


class SourceChunk(BaseModel):
    content: str
    page: int
    section: str
    similarity_score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    augmented_prompt: str


def _model_is_available(target: str, installed: list[str]) -> bool:
    """Verifica disponibilidade do modelo considerando tag :latest implícita."""
    normalized_installed = {m.lower() for m in installed}
    target_lower = target.lower()
    if target_lower in normalized_installed:
        return True
    if ":" not in target_lower and f"{target_lower}:latest" in normalized_installed:
        return True
    return False


@app.get("/api/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Verifica conectividade com o Ollama e disponibilidade dos modelos configurados."""
    url = f"{settings.ollama_base_url.rstrip('/')}/api/tags"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            payload = resp.json()
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError):
        return HealthResponse(
            status="offline",
            ollama_reachable=False,
            embedding_model=settings.embedding_model,
            embedding_model_available=False,
            llm_model=settings.llm_model,
            llm_model_available=False,
            installed_models=[],
            message=(
                f"Servidor Ollama offline ou inacessível em {settings.ollama_base_url}. "
                "Inicie o Ollama antes de indexar ou consultar editais."
            ),
        )

    installed = [m.get("name", "") for m in payload.get("models", []) if m.get("name")]
    emb_ok = _model_is_available(settings.embedding_model, installed)
    llm_ok = _model_is_available(settings.llm_model, installed)

    missing: list[str] = []
    if not emb_ok:
        missing.append(settings.embedding_model)
    if not llm_ok:
        missing.append(settings.llm_model)

    if missing:
        pulls = " e ".join(f"ollama pull {m}" for m in missing)
        return HealthResponse(
            status="degraded",
            ollama_reachable=True,
            embedding_model=settings.embedding_model,
            embedding_model_available=emb_ok,
            llm_model=settings.llm_model,
            llm_model_available=llm_ok,
            installed_models=installed,
            message=f"Ollama ativo, mas faltam modelos configurados. Execute: {pulls}",
        )

    return HealthResponse(
        status="ok",
        ollama_reachable=True,
        embedding_model=settings.embedding_model,
        embedding_model_available=True,
        llm_model=settings.llm_model,
        llm_model_available=True,
        installed_models=installed,
        message="Ollama online e modelos prontos para uso.",
    )


@app.get("/api/document", response_model=DocumentStatusResponse)
async def get_document_status() -> DocumentStatusResponse:
    """Retorna o resumo do edital atualmente indexado no ChromaDB."""
    doc = get_active_document(settings)
    if not doc:
        return DocumentStatusResponse(document=None)
    return DocumentStatusResponse(document=DocumentSummary(**doc))


@app.post("/api/upload", response_model=UploadResponse)
async def upload_edital(file: UploadFile = File(...)) -> UploadResponse:
    """Recebe um PDF de edital, extrai texto com metadados e indexa no ChromaDB."""
    filename = file.filename or "edital.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Formato inválido. Envie apenas arquivos com extensão .pdf.",
        )

    pdf_bytes = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(pdf_bytes) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"Arquivo excede o limite máximo de {settings.max_upload_size_mb} MB.",
        )

    try:
        result = ingest_pdf(pdf_bytes=pdf_bytes, filename=filename, settings=settings)
    except (InvalidPDFError, EmptyPDFError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        msg = str(exc).lower()
        if "timeout" in msg or "timed out" in msg:
            raise HTTPException(
                status_code=504,
                detail="Tempo limite excedido ao gerar embeddings no Ollama.",
            ) from exc
        if "connect" in msg or "11434" in msg:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Não foi possível conectar ao Ollama em {settings.ollama_base_url}. "
                    "Verifique se o servidor está em execução."
                ),
            ) from exc
        raise HTTPException(
            status_code=500,
            detail=f"Erro interno ao processar o edital: {exc}",
        ) from exc

    return UploadResponse(
        status=result["status"],
        document_id=result["document_id"],
        filename=result["filename"],
        total_pages=result["total_pages"],
        total_chunks=result["total_chunks"],
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_edital(payload: ChatRequest) -> ChatResponse:
    """Executa busca k-NN e gera resposta fundamentada com inspeção vetorial."""
    try:
        result = ask_question(
            question=payload.question,
            top_k=payload.top_k,
            settings=settings,
        )
    except NoDocumentIndexedError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OllamaConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except OllamaTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except RAGServiceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(**result)


if STATIC_DIR.exists():
    @app.get("/", include_in_schema=False)
    async def serve_index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
