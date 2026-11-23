# RAG Editais — Consulta Vetorial de Editais de Concursos Públicos

Aplicação web local de **RAG (Retrieval-Augmented Generation)** desenvolvida para a disciplina de **Bancos de Dados Não Convencionais**, demonstrando na prática a ingestão de documentos PDF com metadados estruturais, geração de embeddings densos locais, armazenamento em banco vetorial (**ChromaDB**), busca **k-NN por similaridade de cosseno** e geração fundamentada (*grounded*) via **Ollama**.

---

## Arquitetura Vetorial

```mermaid
flowchart LR
    UI["Frontend Vanilla (HTML/CSS/JS)"] -->|"POST /api/upload (PDF)"| API["FastAPI (app/main.py)"]
    UI -->|"POST /api/chat"| API
    UI -->|"GET /api/health & /api/document"| API

    API --> ING["Ingestion Service (app/ingestion.py)"]
    ING -->|"PyMuPDF + Detecção de Seções"| SPLIT["RecursiveCharacterTextSplitter (800/150)"]
    SPLIT -->|"Embeddings bge-m3"| OLLAMA["Ollama Local"]
    SPLIT -->|"Persistência (hnsw:space = cosine)"| CHROMA[("ChromaDB ./chroma_db")]

    API --> RAG["RAG Service (app/rag_service.py)"]
    RAG -->|"Busca k-NN Top-K"| CHROMA
    RAG -->|"Prompt Aumentado + Chunks"| OLLAMA
    RAG -->|"answer + sources + augmented_prompt"| UI
```

### Conceitos de Bancos de Dados Não Convencionais Demonstrados

1. **Chunking com Enriquecimento de Metadados:** O `PyMuPDF` extrai o texto página a página e aplica heurísticas para identificar títulos e seções típicas de editais (`1. DAS DISPOSIÇÕES PRELIMINARES`, `CAPÍTULO`, `ANEXO`), propagando `source`, `page`, `section`, `chunk_index` e `document_id` para cada chunk.
2. **Representação Vetorial Densa (`bge-m3`):** Tanto os chunks do edital quanto a pergunta do usuário são convertidos em vetores densos localmente pelo modelo `bge-m3` via Ollama.
3. **Banco Vetorial & Índice HNSW por Cosseno (`ChromaDB`):** Os vetores e metadados são persistidos em `./chroma_db` com `collection_metadata={"hnsw:space": "cosine"}`, permitindo busca aproximada **k-NN** (`similarity_search_with_relevance_scores`) ordenada por similaridade de cosseno.
4. **RAG Grounded & Transparência Acadêmica:** O LLM (`qwen2.5:3b`, configurável via `.env`) é instruído a responder **estritamente** com base nos trechos recuperados, citando página e seção. A interface exibe os **Top-K chunks**, os scores de similaridade e permite inspecionar o **prompt aumentado** completo enviado ao modelo.

---

## Estrutura do Projeto

```text
rag-editais/
├── app/
│   ├── __init__.py
│   ├── config.py          # Configurações via Pydantic Settings (.env)
│   ├── ingestion.py       # Extração PyMuPDF, detecção de seções, chunking e ChromaDB
│   ├── rag_service.py     # Busca k-NN, prompt aumentado e chamada ChatOllama
│   └── main.py            # Rotas FastAPI e servidor de arquivos estáticos
├── static/
│   ├── index.html         # Interface web (upload, chat, painel k-NN e modal do prompt)
│   ├── style.css          # Estilização responsiva
│   └── app.js             # Consumo das APIs via Fetch API
├── tests/
│   ├── test_ingestion.py  # Testes unitários de extração, chunking e ChromaDB
│   └── test_api.py        # Testes dos endpoints FastAPI com mocks
├── .env.example
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Como Executar Localmente

### Pré-requisitos
- **Python 3.11+**
- **Ollama** instalado e em execução ([ollama.com](https://ollama.com))

### 1. Baixar os modelos no Ollama

```powershell
ollama pull bge-m3
ollama pull qwen2.5:3b
```

### 2. Configurar o ambiente Python (Windows PowerShell)

```powershell
cd rag-editais
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

*(Em Linux/macOS, ative com `source .venv/bin/activate` e copie com `cp .env.example .env`.)*

### 3. Iniciar a aplicação

```powershell
uvicorn app.main:app --reload
```

Acesse no navegador: **`http://127.0.0.1:8000`**

---

## Endpoints da API

| Método | Rota | Descrição |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Verifica se o Ollama está acessível e se `EMBEDDING_MODEL` e `LLM_MODEL` estão disponíveis |
| `GET` | `/api/document` | Retorna o resumo do edital atualmente indexado no ChromaDB (ou `null`) |
| `POST` | `/api/upload` | Recebe `.pdf` via `multipart/form-data`, extrai seções, gera chunks e indexa no ChromaDB |
| `POST` | `/api/chat` | Recebe `{ "question", "top_k" }`, executa busca k-NN e retorna `{ answer, sources, augmented_prompt }` |

---

## Testes Automatizados

Os testes validam o pipeline de extração/chunking/re-ingestão no ChromaDB e todos os endpoints da API (usando mocks determinísticos para o Ollama):

```powershell
.\.venv\Scripts\pytest.exe -v
```
