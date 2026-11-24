"""Testes unitários do pipeline de ingestão de PDFs e indexação no ChromaDB."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pymupdf
import pytest
from langchain_core.embeddings import Embeddings

from app.config import Settings
from app.ingestion import (
    EmptyPDFError,
    InvalidPDFError,
    chunk_pages,
    detect_section_heading,
    extract_pages_from_pdf,
    get_active_document,
    ingest_pdf,
)


class DeterministicFakeEmbeddings(Embeddings):
    """Embeddings determinísticos em memória para testes sem depender do Ollama."""

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vec = [((digest[i % len(digest)] / 255.0) * 2.0 - 1.0) for i in range(self.dim)]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _make_sample_edital_pdf() -> bytes:
    """Gera um PDF sintético realista de edital em memória usando PyMuPDF."""
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text(
        (50, 72),
        (
            "EDITAL Nº 01/2026 - CONCURSO PÚBLICO\n\n"
            "1. DAS DISPOSIÇÕES PRELIMINARES\n"
            "O prazo de validade do concurso público será de 2 (dois) anos, "
            "podendo ser prorrogado uma única vez por igual período.\n"
        ),
    )

    p2 = doc.new_page()
    p2.insert_text(
        (50, 72),
        (
            "2. DOS CARGOS, VAGAS E REMUNERAÇÃO\n"
            "Cargo: Analista de Sistemas. Vagas: 5 imediatas.\n"
            "Remuneração inicial: R$ 8.500,00 para carga horária de 40 horas semanais.\n\n"
            "ANEXO I - CONTEÚDO PROGRAMÁTICO\n"
            "Bancos de Dados Não Convencionais, SQL, NoSQL e Busca Vetorial.\n"
        ),
    )
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def _make_blank_pdf() -> bytes:
    doc = pymupdf.open()
    doc.new_page()
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_detect_section_heading() -> None:
    assert detect_section_heading("1. DAS DISPOSIÇÕES PRELIMINARES") == "1. DAS DISPOSIÇÕES PRELIMINARES"
    assert detect_section_heading("ANEXO I - CONTEÚDO PROGRAMÁTICO") == "ANEXO I - CONTEÚDO PROGRAMÁTICO"
    assert detect_section_heading("CAPÍTULO II - DAS PROVAS") == "CAPÍTULO II - DAS PROVAS"
    assert detect_section_heading("1. DAS DISPOSIÇÕES PRELIMINARES ............ 3") is None
    assert detect_section_heading("O prazo de validade será de dois anos.") is None
    assert detect_section_heading("12345") is None


def test_extract_pages_and_sections() -> None:
    pages = extract_pages_from_pdf(_make_sample_edital_pdf(), "edital_teste.pdf")
    assert len(pages) == 2
    assert pages[0]["page"] == 1
    assert "EDITAL" in pages[0]["section"] or "DISPOSIÇÕES" in pages[0]["section"]
    assert pages[1]["page"] == 2
    assert "2. DOS CARGOS, VAGAS E REMUNERAÇÃO" in pages[1]["sections_found"]


def test_extract_raises_on_empty_and_invalid_pdf() -> None:
    with pytest.raises(EmptyPDFError):
        extract_pages_from_pdf(_make_blank_pdf(), "vazio.pdf")

    with pytest.raises(EmptyPDFError):
        extract_pages_from_pdf(b"", "empty_bytes.pdf")

    with pytest.raises(InvalidPDFError):
        extract_pages_from_pdf(b"not a pdf file at all", "corrompido.pdf")


def test_chunk_pages_preserves_metadata() -> None:
    pages = extract_pages_from_pdf(_make_sample_edital_pdf(), "edital_teste.pdf")
    chunks = chunk_pages(
        pages=pages,
        filename="edital_teste.pdf",
        document_id="doc123",
        chunk_size=180,
        chunk_overlap=30,
    )
    assert len(chunks) >= 2
    for idx, chunk in enumerate(chunks):
        assert chunk.metadata["source"] == "edital_teste.pdf"
        assert chunk.metadata["document_id"] == "doc123"
        assert chunk.metadata["chunk_index"] == idx
        assert chunk.metadata["page"] in (1, 2)
        assert isinstance(chunk.metadata["section"], str) and chunk.metadata["section"]


def test_ingest_pdf_and_safe_reingestion(tmp_path: Path) -> None:
    test_settings = Settings(
        chroma_persist_dir=str(tmp_path / "chroma_db"),
        chroma_collection_name="test_editais",
        chunk_size=250,
        chunk_overlap=40,
    )
    fake_emb = DeterministicFakeEmbeddings()

    first = ingest_pdf(
        pdf_bytes=_make_sample_edital_pdf(),
        filename="edital_v1.pdf",
        settings=test_settings,
        embeddings=fake_emb,
    )
    assert first["status"] == "ok"
    assert first["filename"] == "edital_v1.pdf"
    assert first["total_pages"] == 2
    assert first["total_chunks"] >= 2

    active = get_active_document(test_settings)
    assert active is not None
    assert active["filename"] == "edital_v1.pdf"

    # Re-ingestão no mesmo diretório persistente não deve falhar por lock no Windows
    second = ingest_pdf(
        pdf_bytes=_make_sample_edital_pdf(),
        filename="edital_v2.pdf",
        settings=test_settings,
        embeddings=fake_emb,
    )
    assert second["filename"] == "edital_v2.pdf"
    assert second["document_id"] != first["document_id"]

    active_after = get_active_document(test_settings)
    assert active_after is not None
    assert active_after["filename"] == "edital_v2.pdf"
