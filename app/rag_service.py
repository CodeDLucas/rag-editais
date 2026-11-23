"""Serviço RAG: busca k-NN por similaridade de cosseno no ChromaDB e geração grounded via Ollama."""

from __future__ import annotations

from typing import Any

import chromadb
import httpx
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from app.config import Settings
from app.ingestion import get_chroma_client, get_vectorstore

SYSTEM_PROMPT = """Você é um assistente especializado na análise de editais de concursos públicos brasileiros.

REGRAS OBRIGATÓRIAS:
1. Responda de forma objetiva, clara e direta em português.
2. Utilize EXCLUSIVAMENTE as informações presentes nos trechos recuperados do edital fornecidos no contexto.
3. Cite obrigatoriamente a página e a seção de origem para cada informação apresentada (exemplo: [Página 3 — 2. DOS CARGOS E VAGAS]).
4. Se a resposta não constar nos trechos recuperados, responda exatamente: "Essa informação não consta nos trechos recuperados do edital." e NÃO utilize conhecimento externo nem faça suposições."""


class RAGServiceError(Exception):
    """Erro base no serviço de consulta RAG."""


class NoDocumentIndexedError(RAGServiceError):
    """Nenhum edital foi indexado no ChromaDB ainda."""


class OllamaConnectionError(RAGServiceError):
    """Falha de conexão com o servidor Ollama local."""


class OllamaTimeoutError(RAGServiceError):
    """Tempo limite excedido ao aguardar resposta do Ollama."""


def build_context_block(docs_with_scores: list[tuple[Document, float]]) -> str:
    """Formata os trechos recuperados com metadados e score de similaridade."""
    blocks: list[str] = []
    for idx, (doc, score) in enumerate(docs_with_scores, start=1):
        page = doc.metadata.get("page", "?")
        section = doc.metadata.get("section", "Não identificada")
        blocks.append(
            f"[Trecho {idx} | Página {page} | Seção: {section} | Similaridade: {score:.4f}]\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(blocks)


def build_user_prompt(question: str, context_block: str) -> str:
    """Monta a mensagem do usuário contendo os trechos recuperados e a pergunta."""
    return (
        f"TRECHOS RECUPERADOS DO EDITAL (ORDENADOS POR RELEVÂNCIA k-NN):\n\n"
        f"{context_block}\n\n"
        f"PERGUNTA DO USUÁRIO:\n{question}\n\n"
        f"Responda estritamente com base nos trechos acima, citando página e seção."
    )


def build_augmented_prompt(question: str, context_block: str) -> str:
    """Retorna o prompt aumentado completo (system + contexto + pergunta) para inspeção."""
    user_prompt = build_user_prompt(question, context_block)
    return f"=== SYSTEM PROMPT ===\n{SYSTEM_PROMPT}\n\n=== USER PROMPT ===\n{user_prompt}"


def _is_connection_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return isinstance(exc, (httpx.ConnectError, ConnectionError)) or any(
        token in msg for token in ("connection refused", "connecterror", "failed to connect", "11434")
    )


def _is_timeout_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return isinstance(exc, (httpx.TimeoutException, TimeoutError)) or "timed out" in msg or "timeout" in msg


def ask_question(
    question: str,
    top_k: int,
    settings: Settings,
    embeddings: Embeddings | None = None,
    llm: BaseChatModel | None = None,
    client: chromadb.ClientAPI | None = None,
) -> dict[str, Any]:
    """Executa busca k-NN no ChromaDB e gera resposta grounded com o LLM local."""
    resolved_client = client or get_chroma_client(settings.chroma_persist_dir)

    try:
        collection = resolved_client.get_collection(name=settings.chroma_collection_name)
        if collection.count() == 0:
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

    try:
        raw_results = vectorstore.similarity_search_with_relevance_scores(question, k=top_k)
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

    docs_with_scores: list[tuple[Document, float]] = [
        (doc, round(max(0.0, min(1.0, float(score))), 4)) for doc, score in raw_results
    ]

    context_block = build_context_block(docs_with_scores)
    user_prompt = build_user_prompt(question, context_block)
    augmented_prompt = build_augmented_prompt(question, context_block)

    resolved_llm = llm or ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=0.1,
        client_kwargs={"timeout": settings.ollama_timeout},
    )

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
                f"Tempo limite excedido ao aguardar resposta do modelo '{settings.llm_model}'."
            ) from exc
        if _is_connection_error(exc):
            raise OllamaConnectionError(
                f"Não foi possível conectar ao Ollama em {settings.ollama_base_url}."
            ) from exc
        raise RAGServiceError(f"Erro ao gerar resposta com o LLM: {exc}") from exc

    answer_text = (
        response.content
        if isinstance(response.content, str)
        else str(response.content)
    ).strip()

    sources = [
        {
            "content": doc.page_content,
            "page": int(doc.metadata.get("page", 0)),
            "section": str(doc.metadata.get("section", "Não identificada")),
            "similarity_score": score,
        }
        for doc, score in docs_with_scores
    ]

    return {
        "answer": answer_text,
        "sources": sources,
        "augmented_prompt": augmented_prompt,
    }
