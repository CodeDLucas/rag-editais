"""Serviço RAG: busca vetorial/híbrida (Cosseno + BM25 RRF), streaming SSE e geração grounded."""

from __future__ import annotations

import math
import re
import time
import unicodedata
from collections import Counter
from collections.abc import AsyncIterator
from typing import Any, Literal

import chromadb
import httpx
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from app.config import Settings
from app.ingestion import (
    get_all_collection_documents,
    get_chroma_client,
    get_vectorstore,
)

SYSTEM_PROMPT = """Você é um assistente especializado na análise de editais de concursos públicos brasileiros.

REGRAS OBRIGATÓRIAS:
1. Responda de forma objetiva, clara e organizada (use tópicos quando houver múltiplos itens) em português.
2. Utilize EXCLUSIVAMENTE as informações presentes nos trechos recuperados do edital fornecidos no contexto.
3. Cite obrigatoriamente a página e a seção de origem em português para cada informação apresentada (exemplo: [Página 3 — 2. DOS CARGOS]). Nunca use "Page".
4. Se a pergunta tiver mais de um item (ex.: "X ou Y") e apenas parte constar nos trechos, responda o que consta e informe o que não foi encontrado. Só responda "Essa informação não consta nos trechos recuperados do edital." se NENHUM dado relevante estiver nos trechos."""

_PT_STOPWORDS = {
    "a", "ao", "aos", "as", "com", "como", "da", "das", "de", "do", "dos",
    "e", "em", "entre", "essa", "esse", "esta", "este", "foi", "ha", "na",
    "nas", "no", "nos", "o", "os", "ou", "para", "pela", "pelas", "pelo",
    "pelos", "por", "qual", "quais", "que", "se", "sem", "ser", "sera",
    "sobre", "sua", "suas", "seu", "seus", "um", "uma", "uns", "umas",
}


class RAGServiceError(Exception):
    """Erro base no serviço de consulta RAG."""


class NoDocumentIndexedError(RAGServiceError):
    """Nenhum edital foi indexado no ChromaDB ainda."""


class OllamaConnectionError(RAGServiceError):
    """Falha de conexão com o servidor Ollama local."""


class OllamaTimeoutError(RAGServiceError):
    """Tempo limite excedido ao aguardar resposta do Ollama."""


def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def _tokenize_pt(text: str) -> list[str]:
    """Tokeniza texto em português normalizando acentos e removendo stopwords."""
    normalized = _strip_accents(text.lower())
    tokens = re.findall(r"[a-z0-9]{2,}", normalized)
    return [t for t in tokens if t not in _PT_STOPWORDS]


def _bm25_scores(
    query: str,
    documents: list[Document],
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """Calcula scores BM25 léxicos em memória para uma lista de documentos."""
    if not documents:
        return []

    query_tokens = _tokenize_pt(query)
    if not query_tokens:
        return [0.0] * len(documents)

    tokenized_docs = [_tokenize_pt(doc.page_content) for doc in documents]
    doc_freqs: Counter[str] = Counter()
    for tokens in tokenized_docs:
        doc_freqs.update(set(tokens))

    n_docs = len(documents)
    avgdl = sum(len(t) for t in tokenized_docs) / max(n_docs, 1)

    scores: list[float] = []
    for tokens in tokenized_docs:
        if not tokens:
            scores.append(0.0)
            continue
        tf_map = Counter(tokens)
        dl = len(tokens)
        score = 0.0
        for q_tok in query_tokens:
            df = doc_freqs.get(q_tok, 0)
            if df == 0:
                continue
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            tf = tf_map.get(q_tok, 0)
            denom = tf + k1 * (1.0 - b + b * (dl / max(avgdl, 1.0)))
            score += idf * (tf * (k1 + 1.0)) / max(denom, 1e-9)
        scores.append(score)

    return scores


def _doc_key(doc: Document) -> tuple[Any, ...]:
    """Chave estável de identificação de chunk para fusão RRF."""
    if "chunk_index" in doc.metadata:
        return ("idx", doc.metadata.get("document_id", ""), doc.metadata["chunk_index"])
    return ("txt", doc.metadata.get("page", 0), doc.page_content[:120])


def _extract_display_content(doc: Document) -> str:
    """Retorna o texto limpo do chunk (sem o prefixo contextual de indexação)."""
    raw = doc.metadata.get("raw_content")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return doc.page_content.strip()


def _retrieve_chunks(
    question: str,
    top_k: int,
    search_mode: Literal["hybrid", "vector"],
    settings: Settings,
    embeddings: Embeddings | None = None,
    client: chromadb.ClientAPI | None = None,
) -> list[dict[str, Any]]:
    """Executa busca k-NN vetorial pura ou híbrida (Cosseno + BM25 RRF)."""
    resolved_client = client or get_chroma_client(settings.chroma_persist_dir)

    try:
        collection = resolved_client.get_collection(name=settings.chroma_collection_name)
        total_docs = collection.count()
        if total_docs == 0:
            raise NoDocumentIndexedError(
                "Nenhum edital foi indexado ainda. Faça upload de um PDF primeiro."
            )
    except NoDocumentIndexedError:
        raise
    except Exception as exc:
        raise NoDocumentIndexedError(
            "Nenhum edital foi indexado ainda. Faça upload de um PDF primeiro."
        ) from exc

    vectorstore = get_vectorstore(
        settings=settings,
        embeddings=embeddings,
        client=resolved_client,
    )

    candidate_k = top_k if search_mode == "vector" else min(max(top_k * 4, 16), total_docs)

    try:
        raw_dense = vectorstore.similarity_search_with_relevance_scores(question, k=candidate_k)
    except Exception as exc:
        if _is_timeout_error(exc):
            raise OllamaTimeoutError(
                "Tempo limite excedido ao gerar embedding da pergunta no Ollama."
            ) from exc
        if _is_connection_error(exc):
            raise OllamaConnectionError(
                f"Não foi possível conectar ao Ollama em {settings.ollama_base_url}."
            ) from exc
        raise RAGServiceError(f"Erro na busca vetorial: {exc}") from exc

    if search_mode == "vector":
        return [
            {
                "doc": doc,
                "similarity_score": round(max(0.0, min(1.0, float(score))), 4),
                "lexical_score": None,
                "rrf_score": None,
            }
            for doc, score in raw_dense[:top_k]
        ]

    # Busca Híbrida: combina ranking denso (Cosseno) + ranking léxico (BM25) via RRF
    all_docs = (
        get_all_collection_documents(resolved_client, settings.chroma_collection_name)
        if total_docs <= 2000
        else [doc for doc, _ in raw_dense]
    )

    dense_rank_map: dict[tuple[Any, ...], tuple[int, float, Document]] = {}
    for rank, (doc, score) in enumerate(raw_dense, start=1):
        key = _doc_key(doc)
        if key not in dense_rank_map:
            dense_rank_map[key] = (rank, round(max(0.0, min(1.0, float(score))), 4), doc)

    bm25_raw = _bm25_scores(question, all_docs)
    max_bm25 = max(bm25_raw) if bm25_raw else 0.0
    bm25_indexed = sorted(
        zip(all_docs, bm25_raw),
        key=lambda item: item[1],
        reverse=True,
    )

    bm25_rank_map: dict[tuple[Any, ...], tuple[int, float, Document]] = {}
    for rank, (doc, raw_score) in enumerate(bm25_indexed, start=1):
        if raw_score <= 0.0:
            continue
        key = _doc_key(doc)
        norm_lex = round(raw_score / max_bm25, 4) if max_bm25 > 0 else 0.0
        if key not in bm25_rank_map:
            bm25_rank_map[key] = (rank, norm_lex, doc)

    rrf_k = 60
    candidate_keys = set(dense_rank_map.keys()) | {
        k for k, (r, _, _) in bm25_rank_map.items() if r <= max(top_k * 4, 16)
    }

    fused: list[dict[str, Any]] = []
    for key in candidate_keys:
        rrf = 0.0
        cos_score = 0.0
        lex_score = 0.0
        doc: Document | None = None

        if key in dense_rank_map:
            d_rank, cos_score, doc = dense_rank_map[key]
            rrf += 1.0 / (rrf_k + d_rank)
        if key in bm25_rank_map:
            b_rank, lex_score, b_doc = bm25_rank_map[key]
            rrf += 1.0 / (rrf_k + b_rank)
            if doc is None:
                doc = b_doc

        if doc is not None:
            fused.append(
                {
                    "doc": doc,
                    "similarity_score": cos_score,
                    "lexical_score": lex_score,
                    "rrf_score": round(rrf, 4),
                }
            )

    fused.sort(key=lambda item: (item["rrf_score"], item["similarity_score"]), reverse=True)
    return fused[:top_k]


def build_context_block(retrieved_items: list[dict[str, Any]]) -> str:
    """Formata os trechos recuperados com metadados e scores."""
    blocks: list[str] = []
    for idx, item in enumerate(retrieved_items, start=1):
        doc: Document = item["doc"]
        page = doc.metadata.get("page", "?")
        section = doc.metadata.get("section", "Não identificada")
        cos_score = item["similarity_score"]
        lex_score = item.get("lexical_score")
        score_info = f"Cosseno: {cos_score:.4f}"
        if lex_score is not None:
            score_info += f" | BM25: {lex_score:.4f}"
        blocks.append(
            f"[Trecho {idx} | Página {page} | Seção: {section} | {score_info}]\n"
            f"{_extract_display_content(doc)}"
        )
    return "\n\n---\n\n".join(blocks)


def build_user_prompt(question: str, context_block: str) -> str:
    """Monta a mensagem do usuário contendo os trechos recuperados e a pergunta."""
    return (
        f"TRECHOS RECUPERADOS DO EDITAL (ORDENADOS POR RELEVÂNCIA):\n\n"
        f"{context_block}\n\n"
        f"PERGUNTA DO USUÁRIO:\n{question}\n\n"
        f"Responda estritamente com base nos trechos acima, citando página e seção."
    )


def build_augmented_prompt(question: str, context_block: str) -> str:
    """Retorna o prompt aumentado completo (system + contexto + pergunta) para inspeção."""
    user_prompt = build_user_prompt(question, context_block)
    return f"=== SYSTEM PROMPT ===\n{SYSTEM_PROMPT}\n\n=== USER PROMPT ===\n{user_prompt}"


def _format_sources(retrieved_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "content": _extract_display_content(item["doc"]),
            "page": int(item["doc"].metadata.get("page", 0)),
            "section": str(item["doc"].metadata.get("section", "Não identificada")),
            "similarity_score": item["similarity_score"],
            "lexical_score": item.get("lexical_score"),
            "rrf_score": item.get("rrf_score"),
        }
        for item in retrieved_items
    ]


def _is_connection_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return isinstance(exc, (httpx.ConnectError, ConnectionError)) or any(
        token in msg for token in ("connection refused", "connecterror", "failed to connect", "11434")
    )


def _is_timeout_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return isinstance(exc, (httpx.TimeoutException, TimeoutError)) or "timed out" in msg or "timeout" in msg


def _build_llm(active_model: str, settings: Settings) -> ChatOllama:
    return ChatOllama(
        model=active_model,
        base_url=settings.ollama_base_url,
        temperature=0.1,
        client_kwargs={"timeout": settings.ollama_timeout},
    )


def ask_question(
    question: str,
    top_k: int,
    settings: Settings,
    search_mode: Literal["hybrid", "vector"] = "hybrid",
    model: str | None = None,
    embeddings: Embeddings | None = None,
    llm: BaseChatModel | None = None,
    client: chromadb.ClientAPI | None = None,
) -> dict[str, Any]:
    """Executa busca vetorial/híbrida no ChromaDB e gera resposta grounded com o LLM local."""
    active_model = (model or "").strip() or settings.llm_model

    t_ret0 = time.perf_counter()
    retrieved_items = _retrieve_chunks(
        question=question,
        top_k=top_k,
        search_mode=search_mode,
        settings=settings,
        embeddings=embeddings,
        client=client,
    )
    retrieval_ms = round((time.perf_counter() - t_ret0) * 1000.0, 1)

    context_block = build_context_block(retrieved_items)
    user_prompt = build_user_prompt(question, context_block)
    augmented_prompt = build_augmented_prompt(question, context_block)
    sources = _format_sources(retrieved_items)

    resolved_llm = llm or _build_llm(active_model, settings)

    t_gen0 = time.perf_counter()
    try:
        response = resolved_llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
    except Exception as exc:
        if _is_timeout_error(exc):
            raise OllamaTimeoutError(
                f"Tempo limite excedido ao aguardar resposta do modelo '{active_model}'."
            ) from exc
        if _is_connection_error(exc):
            raise OllamaConnectionError(
                f"Não foi possível conectar ao Ollama em {settings.ollama_base_url}."
            ) from exc
        raise RAGServiceError(f"Erro ao gerar resposta com o LLM: {exc}") from exc
    generation_ms = round((time.perf_counter() - t_gen0) * 1000.0, 1)

    answer_text = (
        response.content
        if isinstance(response.content, str)
        else str(response.content)
    ).strip()

    return {
        "answer": answer_text,
        "sources": sources,
        "augmented_prompt": augmented_prompt,
        "metrics": {
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "model_used": active_model,
            "search_mode": search_mode,
        },
    }


async def stream_question(
    question: str,
    top_k: int,
    settings: Settings,
    search_mode: Literal["hybrid", "vector"] = "hybrid",
    model: str | None = None,
    embeddings: Embeddings | None = None,
    llm: BaseChatModel | None = None,
    client: chromadb.ClientAPI | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Executa busca e transmite tokens do LLM em tempo real via eventos estruturados."""
    active_model = (model or "").strip() or settings.llm_model

    t_ret0 = time.perf_counter()
    retrieved_items = _retrieve_chunks(
        question=question,
        top_k=top_k,
        search_mode=search_mode,
        settings=settings,
        embeddings=embeddings,
        client=client,
    )
    retrieval_ms = round((time.perf_counter() - t_ret0) * 1000.0, 1)

    context_block = build_context_block(retrieved_items)
    user_prompt = build_user_prompt(question, context_block)
    augmented_prompt = build_augmented_prompt(question, context_block)
    sources = _format_sources(retrieved_items)

    yield {
        "event": "sources",
        "data": {
            "sources": sources,
            "augmented_prompt": augmented_prompt,
            "retrieval_ms": retrieval_ms,
            "model_used": active_model,
            "search_mode": search_mode,
        },
    }

    resolved_llm = llm or _build_llm(active_model, settings)
    t_gen0 = time.perf_counter()
    try:
        async for chunk in resolved_llm.astream(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        ):
            token = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
            if token:
                yield {"event": "token", "data": {"token": token}}
    except Exception as exc:
        if _is_timeout_error(exc):
            raise OllamaTimeoutError(
                f"Tempo limite excedido ao aguardar resposta do modelo '{active_model}'."
            ) from exc
        if _is_connection_error(exc):
            raise OllamaConnectionError(
                f"Não foi possível conectar ao Ollama em {settings.ollama_base_url}."
            ) from exc
        raise RAGServiceError(f"Erro ao gerar resposta com o LLM: {exc}") from exc

    generation_ms = round((time.perf_counter() - t_gen0) * 1000.0, 1)
    yield {
        "event": "done",
        "data": {
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "model_used": active_model,
            "search_mode": search_mode,
        },
    }
