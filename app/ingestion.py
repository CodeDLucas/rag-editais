"""Pipeline de ingestão de editais em PDF: extração, detecção de seções, chunking contextual e ChromaDB."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
import pymupdf
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings

DEFAULT_SECTION = "Introdução / Preâmbulo"
ACTIVE_DOC_FILENAME = "active_document.json"

# Padrões comuns de títulos/seções em editais brasileiros
_NUMBERED_HEADING_RE = re.compile(
    r"^\s*\d+(?:\.\d+)*\s*(?:[.\-–—:]|\s)\s*[A-ZÁÀÂÃÉÈÊÍÏÓÔÕÖÚÇÑ0-9].{2,90}$"
)
_STRUCTURAL_HEADING_RE = re.compile(
    r"^\s*(?:CAP[ÍI]TULO|SE[ÇC][ÃA]O|T[ÍI]TULO|ANEXO|EDITAL)\s+[IVXLCDM0-9]+.*$",
    re.IGNORECASE,
)
_TOC_DOT_LEADER_RE = re.compile(r"\.{3,}|…{2,}")


class IngestionError(Exception):
    """Erro base no pipeline de ingestão de PDF."""


class InvalidPDFError(IngestionError):
    """Arquivo inválido ou corrompido que não pôde ser aberto como PDF."""


class EmptyPDFError(IngestionError):
    """PDF sem páginas ou sem texto extraível (ex.: escaneado sem OCR)."""


def detect_section_heading(line: str) -> str | None:
    """Identifica se uma linha representa um título de seção/capítulo de edital."""
    cleaned = re.sub(r"\s+", " ", line).strip()
    if len(cleaned) < 4 or len(cleaned) > 100:
        return None

    # Ignora linhas de sumário com pontilhado (ex.: "1. DAS DISPOSIÇÕES ........ 3")
    if _TOC_DOT_LEADER_RE.search(cleaned):
        return None

    # Evita capturar linhas puramente numéricas, datas ou pontuações
    letters = re.findall(r"[A-Za-zÀ-ÿ]", cleaned)
    if len(letters) < 3:
        return None

    if _STRUCTURAL_HEADING_RE.match(cleaned):
        return cleaned.rstrip(".:- ")

    if _NUMBERED_HEADING_RE.match(cleaned):
        # Exige predominância de maiúsculas no título numerado para reduzir falsos positivos
        upper_count = sum(1 for ch in letters if ch.isupper())
        if upper_count / len(letters) >= 0.6:
            return cleaned.rstrip(".:- ")

    # Linha isolada toda em caixa alta (ex.: "DAS DISPOSIÇÕES FINAIS")
    if len(cleaned) <= 90 and cleaned == cleaned.upper():
        return cleaned.rstrip(".:- ")

    return None


def _clean_page_text(raw_text: str) -> str:
    """Normaliza quebras de linha e espaços preservando estrutura de parágrafos."""
    lines = [line.rstrip() for line in raw_text.splitlines()]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages_from_pdf(pdf_bytes: bytes, filename: str) -> list[dict[str, Any]]:
    """Extrai texto página a página com PyMuPDF e rastreia seções detectadas."""
    if not pdf_bytes:
        raise EmptyPDFError("O arquivo PDF enviado está vazio.")

    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise InvalidPDFError(f"Não foi possível abrir '{filename}' como PDF válido.") from exc

    try:
        if doc.page_count == 0:
            raise EmptyPDFError("O arquivo PDF não possui páginas.")

        pages: list[dict[str, Any]] = []
        current_section = DEFAULT_SECTION
        total_chars = 0

        for idx in range(doc.page_count):
            page = doc.load_page(idx)
            cleaned_text = _clean_page_text(page.get_text("text"))
            total_chars += len(re.sub(r"\s+", "", cleaned_text))

            page_initial_section = current_section
            sections_found: list[str] = []

            for line in cleaned_text.splitlines():
                heading = detect_section_heading(line)
                if heading:
                    current_section = heading
                    sections_found.append(heading)

            pages.append(
                {
                    "page": idx + 1,
                    "text": cleaned_text,
                    "section": sections_found[0] if sections_found else page_initial_section,
                    "initial_section": page_initial_section,
                    "sections_found": sections_found,
                }
            )

        if total_chars < 20:
            raise EmptyPDFError(
                "Não foi possível extrair texto do PDF. O arquivo pode estar vazio ou escaneado sem OCR."
            )

        return pages
    finally:
        doc.close()


def _resolve_chunk_section(chunk_text: str, active_section: str) -> str:
    """Associa ao chunk o título de seção mais recente encontrado nele ou herdado."""
    resolved = active_section
    for line in chunk_text.splitlines():
        heading = detect_section_heading(line)
        if heading:
            resolved = heading
    return resolved


def chunk_pages(
    pages: list[dict[str, Any]],
    filename: str,
    document_id: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Divide o texto das páginas em chunks com enriquecimento contextual de seção/página."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents: list[Document] = []
    global_idx = 0
    running_section = DEFAULT_SECTION

    for page_info in pages:
        text = page_info.get("text", "")
        if not text.strip():
            continue

        page_num = int(page_info["page"])
        if page_info.get("initial_section"):
            running_section = page_info["initial_section"]

        chunks = splitter.split_text(text)
        for chunk_text in chunks:
            cleaned_chunk = chunk_text.strip()
            if not cleaned_chunk:
                continue

            chunk_section = _resolve_chunk_section(cleaned_chunk, running_section)
            running_section = chunk_section

            contextual_content = f"[Seção: {chunk_section} | Página: {page_num}]\n{cleaned_chunk}"

            documents.append(
                Document(
                    page_content=contextual_content,
                    metadata={
                        "source": filename,
                        "page": page_num,
                        "section": chunk_section,
                        "chunk_index": global_idx,
                        "document_id": document_id,
                        "raw_content": cleaned_chunk,
                    },
                )
            )
            global_idx += 1

    return documents


def get_embeddings(settings: Settings) -> OllamaEmbeddings:
    """Instancia o gerador de embeddings locais via Ollama."""
    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )


def get_chroma_client(persist_dir: str) -> chromadb.PersistentClient:
    """Cria cliente persistente do ChromaDB com telemetria desativada."""
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_vectorstore(
    settings: Settings,
    embeddings: Embeddings | None = None,
    client: chromadb.ClientAPI | None = None,
) -> Chroma:
    """Retorna instância do Chroma configurada com similaridade de cosseno."""
    resolved_embeddings = embeddings or get_embeddings(settings)
    resolved_client = client or get_chroma_client(settings.chroma_persist_dir)
    return Chroma(
        client=resolved_client,
        collection_name=settings.chroma_collection_name,
        embedding_function=resolved_embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )


def get_all_collection_documents(
    client: chromadb.ClientAPI,
    collection_name: str,
) -> list[Document]:
    """Recupera todos os documentos e metadados da coleção para busca lexical BM25."""
    collection = client.get_collection(name=collection_name)
    payload = collection.get(include=["documents", "metadatas"])
    raw_docs = payload.get("documents") or []
    raw_metas = payload.get("metadatas") or []

    docs: list[Document] = []
    for text, meta in zip(raw_docs, raw_metas):
        if text is None:
            continue
        docs.append(Document(page_content=text, metadata=meta or {}))
    return docs


def _reset_collection(client: chromadb.ClientAPI, collection_name: str) -> None:
    """Remove a coleção anterior via API do ChromaDB sem apagar arquivos em uso."""
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass


def _active_doc_path(persist_dir: str) -> Path:
    return Path(persist_dir) / ACTIVE_DOC_FILENAME


def save_active_document(persist_dir: str, summary: dict[str, Any]) -> None:
    """Salva resumo do documento atualmente indexado."""
    path = _active_doc_path(persist_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def get_active_document(
    settings: Settings,
    client: chromadb.ClientAPI | None = None,
) -> dict[str, Any] | None:
    """Retorna metadados do edital ativo se existir e possuir vetores na coleção."""
    path = _active_doc_path(settings.chroma_persist_dir)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    try:
        resolved_client = client or get_chroma_client(settings.chroma_persist_dir)
        collection = resolved_client.get_collection(name=settings.chroma_collection_name)
        if collection.count() == 0:
            return None
    except Exception:
        return None

    return data


def ingest_pdf(
    pdf_bytes: bytes,
    filename: str,
    settings: Settings,
    embeddings: Embeddings | None = None,
    client: chromadb.ClientAPI | None = None,
) -> dict[str, Any]:
    """Executa pipeline completo: extração, chunking, limpeza segura e indexação vetorial."""
    pages = extract_pages_from_pdf(pdf_bytes, filename)
    document_id = uuid.uuid4().hex[:12]

    chunks = chunk_pages(
        pages=pages,
        filename=filename,
        document_id=document_id,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    if not chunks:
        raise EmptyPDFError("Nenhum chunk válido foi gerado a partir do PDF.")

    resolved_client = client or get_chroma_client(settings.chroma_persist_dir)
    _reset_collection(resolved_client, settings.chroma_collection_name)

    vectorstore = get_vectorstore(
        settings=settings,
        embeddings=embeddings,
        client=resolved_client,
    )
    vectorstore.add_documents(chunks)

    summary = {
        "status": "ok",
        "document_id": document_id,
        "filename": filename,
        "total_pages": len(pages),
        "total_chunks": len(chunks),
        "indexed_at": datetime.now(timezone.utc).isoformat(),
    }
    save_active_document(settings.chroma_persist_dir, summary)
    return summary
