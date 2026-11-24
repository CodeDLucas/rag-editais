(() => {
  const healthPill = document.getElementById("health-pill");
  const healthBanner = document.getElementById("health-banner");
  const healthBannerText = document.getElementById("health-banner-text");
  const btnRefreshHealth = document.getElementById("btn-refresh-health");

  const dropzone = document.getElementById("dropzone");
  const pdfInput = document.getElementById("pdf-input");
  const selectedFileName = document.getElementById("selected-file-name");
  const btnUpload = document.getElementById("btn-upload");
  const uploadSpinner = document.getElementById("upload-spinner");
  const uploadBtnText = document.getElementById("upload-btn-text");
  const uploadStatus = document.getElementById("upload-status");

  const docEmpty = document.getElementById("doc-empty");
  const docSummary = document.getElementById("doc-summary");
  const docFilename = document.getElementById("doc-filename");
  const docPages = document.getElementById("doc-pages");
  const docChunks = document.getElementById("doc-chunks");
  const docIndexedAt = document.getElementById("doc-indexed-at");

  const topKInput = document.getElementById("top-k-input");
  const topKValue = document.getElementById("top-k-value");
  const searchModeSelect = document.getElementById("search-mode-select");
  const llmModelSelect = document.getElementById("llm-model-select");

  const chatMessages = document.getElementById("chat-messages");
  const chatForm = document.getElementById("chat-form");
  const questionInput = document.getElementById("question-input");
  const btnSend = document.getElementById("btn-send");
  const sendSpinner = document.getElementById("send-spinner");
  const btnClearChat = document.getElementById("btn-clear-chat");

  const promptModal = document.getElementById("prompt-modal");
  const modalPromptContent = document.getElementById("modal-prompt-content");
  const btnCloseModal = document.getElementById("btn-close-modal");
  const btnCopyPrompt = document.getElementById("btn-copy-prompt");

  let selectedFile = null;
  let hasActiveDocument = false;
  let isBusy = false;

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text ?? "";
    return div.innerHTML;
  }

  function formatAnswerHtml(text) {
    const escaped = escapeHtml(text);
    const withBold = escaped.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    return withBold
      .split(/\n{2,}/)
      .map((para) => `<p>${para.replace(/\n/g, "<br />")}</p>`)
      .join("");
  }

  function setChatEnabled(enabled) {
    hasActiveDocument = enabled;
    questionInput.disabled = !enabled || isBusy;
    btnSend.disabled = !enabled || isBusy;
    questionInput.placeholder = enabled
      ? "Digite sua pergunta sobre o edital..."
      : "Carregue um edital em PDF ao lado para começar a perguntar...";
  }

  function renderActiveDocument(doc) {
    if (!doc) {
      docEmpty.classList.remove("hidden");
      docSummary.classList.add("hidden");
      setChatEnabled(false);
      return;
    }

    docEmpty.classList.add("hidden");
    docSummary.classList.remove("hidden");
    docFilename.textContent = doc.filename;
    docPages.textContent = String(doc.total_pages);
    docChunks.textContent = String(doc.total_chunks);

    if (doc.indexed_at) {
      const date = new Date(doc.indexed_at);
      docIndexedAt.textContent = `Indexado em ${date.toLocaleString("pt-BR")}`;
    } else {
      docIndexedAt.textContent = "";
    }
    setChatEnabled(true);
  }

  function populateModelSelector(chatModels, defaultModel) {
    const prev = llmModelSelect.value;
    llmModelSelect.innerHTML = "";

    const models = Array.isArray(chatModels) && chatModels.length > 0
      ? chatModels
      : [defaultModel].filter(Boolean);

    models.forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m === defaultModel ? `${m} (padrão)` : m;
      llmModelSelect.appendChild(opt);
    });

    if (prev && models.includes(prev)) {
      llmModelSelect.value = prev;
    } else if (defaultModel && models.includes(defaultModel)) {
      llmModelSelect.value = defaultModel;
    }
  }

  async function checkHealth() {
    healthPill.textContent = "Verificando Ollama...";
    healthPill.className = "health-pill pill-loading";

    try {
      const res = await fetch("/api/health");
      const data = await res.json();

      populateModelSelector(data.chat_models, data.llm_model);

      if (data.status === "ok") {
        healthPill.textContent = `Ollama OK (${data.embedding_model} + ${data.llm_model})`;
        healthPill.className = "health-pill pill-ok";
        healthBanner.classList.add("hidden");
      } else if (data.status === "degraded") {
        healthPill.textContent = "Modelos pendentes no Ollama";
        healthPill.className = "health-pill pill-degraded";
        healthBanner.classList.remove("hidden", "is-offline");
        healthBannerText.textContent = data.message;
      } else {
        healthPill.textContent = "Ollama offline";
        healthPill.className = "health-pill pill-offline";
        healthBanner.classList.remove("hidden");
        healthBanner.classList.add("is-offline");
        healthBannerText.textContent = data.message;
      }
    } catch {
      healthPill.textContent = "Erro ao checar API";
      healthPill.className = "health-pill pill-offline";
      healthBanner.classList.remove("hidden");
      healthBanner.classList.add("is-offline");
      healthBannerText.textContent = "Não foi possível comunicar com o backend FastAPI.";
    }
  }

  async function loadActiveDocument() {
    try {
      const res = await fetch("/api/document");
      const data = await res.json();
      renderActiveDocument(data.document || null);
    } catch {
      renderActiveDocument(null);
    }
  }

  function selectFile(file) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      uploadStatus.textContent = "Selecione apenas arquivos com extensão .pdf.";
      uploadStatus.className = "status-feedback is-error";
      return;
    }
    selectedFile = file;
    selectedFileName.textContent = `${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
    btnUpload.disabled = false;
    uploadStatus.textContent = "";
    uploadStatus.className = "status-feedback";
  }

  async function handleUpload() {
    if (!selectedFile || isBusy) return;

    isBusy = true;
    btnUpload.disabled = true;
    uploadSpinner.classList.remove("hidden");
    uploadBtnText.textContent = "Indexando vetores...";
    uploadStatus.textContent = "Extraindo texto, gerando embeddings e salvando no ChromaDB...";
    uploadStatus.className = "status-feedback";
    setChatEnabled(hasActiveDocument);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const res = await fetch("/api/upload", { method: "POST", body: formData });
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || "Falha ao processar o arquivo PDF.");
      }

      uploadStatus.textContent = `Sucesso! ${data.total_pages} páginas e ${data.total_chunks} chunks indexados.`;
      uploadStatus.className = "status-feedback is-ok";
      await loadActiveDocument();
    } catch (err) {
      uploadStatus.textContent = err.message;
      uploadStatus.className = "status-feedback is-error";
    } finally {
      isBusy = false;
      btnUpload.disabled = !selectedFile;
      uploadSpinner.classList.add("hidden");
      uploadBtnText.textContent = "Processar e Indexar Edital";
      setChatEnabled(hasActiveDocument);
    }
  }

  function removeWelcomeIfPresent() {
    const welcome = chatMessages.querySelector(".chat-welcome");
    if (welcome) welcome.remove();
  }

  function appendUserMessage(text) {
    removeWelcomeIfPresent();
    const div = document.createElement("div");
    div.className = "msg-user";
    div.textContent = text;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function createStreamingAssistantCard() {
    const card = document.createElement("div");
    card.className = "msg-assistant";

    const answerDiv = document.createElement("div");
    answerDiv.className = "assistant-answer";

    const statusDiv = document.createElement("div");
    statusDiv.className = "streaming-status";
    statusDiv.textContent = "Executando busca no ChromaDB...";

    const telemetryDiv = document.createElement("div");
    telemetryDiv.className = "telemetry-bar hidden";

    const knnPanel = document.createElement("div");
    knnPanel.className = "knn-panel hidden";

    card.appendChild(answerDiv);
    card.appendChild(statusDiv);
    card.appendChild(telemetryDiv);
    card.appendChild(knnPanel);

    chatMessages.appendChild(card);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    return { card, answerDiv, statusDiv, telemetryDiv, knnPanel };
  }

  function renderKnnPanel(knnPanel, sources, augmentedPrompt, searchMode) {
    knnPanel.classList.remove("hidden");
    knnPanel.innerHTML = "";

    const knnHeader = document.createElement("div");
    knnHeader.className = "knn-header";

    const modeLabel = searchMode === "hybrid" ? "Híbrida: Cosseno + BM25" : "Vetorial: Cosseno k-NN";
    const title = document.createElement("span");
    title.className = "knn-title";
    title.textContent = `Inspeção Vetorial · Top-${sources.length} Chunks (${modeLabel})`;

    const btnPrompt = document.createElement("button");
    btnPrompt.type = "button";
    btnPrompt.className = "btn-secondary";
    btnPrompt.textContent = "Ver Prompt Aumentado";
    btnPrompt.addEventListener("click", () => {
      modalPromptContent.textContent = augmentedPrompt;
      promptModal.showModal();
    });

    knnHeader.appendChild(title);
    knnHeader.appendChild(btnPrompt);
    knnPanel.appendChild(knnHeader);

    const grid = document.createElement("div");
    grid.className = "sources-grid";

    sources.forEach((src, index) => {
      const pct = Math.max(0, Math.min(100, Math.round(src.similarity_score * 100)));
      const lexBadge =
        typeof src.lexical_score === "number"
          ? `<span class="source-score-lex">bm25: ${src.lexical_score.toFixed(2)}</span>`
          : "";

      const card = document.createElement("article");
      card.className = "source-card";
      card.innerHTML = `
        <div class="source-meta">
          <span class="source-page">#${index + 1} · Página ${escapeHtml(String(src.page))}</span>
          <div class="score-badges">
            <span class="source-score">cos: ${src.similarity_score.toFixed(4)}</span>
            ${lexBadge}
          </div>
        </div>
        <div class="score-bar-track">
          <div class="score-bar-fill" style="width: ${pct}%"></div>
        </div>
        <div class="source-section">${escapeHtml(src.section)}</div>
        <div class="source-content">${escapeHtml(src.content)}</div>
      `;
      grid.appendChild(card);
    });

    knnPanel.appendChild(grid);
  }

  function renderTelemetry(telemetryDiv, metrics, getFullText) {
    telemetryDiv.classList.remove("hidden");
    const genSec = (metrics.generation_ms / 1000).toFixed(2);
    const modeText = metrics.search_mode === "hybrid" ? "Híbrida" : "Vetorial";

    const infoSpan = document.createElement("span");
    infoSpan.textContent = `Busca: ${metrics.retrieval_ms} ms · LLM: ${genSec} s · Modelo: ${metrics.model_used} · Modo: ${modeText}`;

    const copyBtn = document.createElement("button");
    copyBtn.type = "button";
    copyBtn.className = "btn-secondary";
    copyBtn.textContent = "Copiar resposta";
    copyBtn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(getFullText());
        copyBtn.textContent = "Copiada!";
        setTimeout(() => {
          copyBtn.textContent = "Copiar resposta";
        }, 1500);
      } catch {
        copyBtn.textContent = "Falhou";
      }
    });

    telemetryDiv.appendChild(infoSpan);
    telemetryDiv.appendChild(copyBtn);
  }

  async function sendQuestion(questionText) {
    const question = questionText.trim();
    if (!question || !hasActiveDocument || isBusy) return;

    isBusy = true;
    setChatEnabled(true);
    sendSpinner.classList.remove("hidden");
    questionInput.value = "";

    appendUserMessage(question);
    const ui = createStreamingAssistantCard();
    let accumulatedAnswer = "";

    try {
      const topK = Number(topKInput.value) || 4;
      const searchMode = searchModeSelect.value || "hybrid";
      const model = llmModelSelect.value || null;

      const res = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          top_k: topK,
          search_mode: searchMode,
          model,
        }),
      });

      if (!res.ok || !res.body) {
        const errPayload = await res.json().catch(() => ({}));
        throw new Error(errPayload.detail || "Erro ao consultar o edital.");
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop() || "";

        for (const rawEvent of events) {
          const lines = rawEvent.split("\n");
          let eventName = "message";
          let dataStr = "";

          for (const line of lines) {
            if (line.startsWith("event:")) {
              eventName = line.slice(6).trim();
            } else if (line.startsWith("data:")) {
              dataStr += line.slice(5).trim();
            }
          }

          if (!dataStr) continue;
          const payload = JSON.parse(dataStr);

          if (eventName === "sources") {
            renderKnnPanel(ui.knnPanel, payload.sources, payload.augmented_prompt, payload.search_mode);
            ui.statusDiv.textContent = `Trechos recuperados (${payload.retrieval_ms} ms). Gerando resposta com ${payload.model_used}...`;
          } else if (eventName === "token") {
            accumulatedAnswer += payload.token;
            ui.answerDiv.innerHTML = formatAnswerHtml(accumulatedAnswer);
            chatMessages.scrollTop = chatMessages.scrollHeight;
          } else if (eventName === "done") {
            ui.statusDiv.classList.add("hidden");
            renderTelemetry(ui.telemetryDiv, payload, () => accumulatedAnswer);
          } else if (eventName === "error") {
            throw new Error(payload.detail || "Erro durante o streaming.");
          }
        }
      }
    } catch (err) {
      ui.card.classList.add("is-error");
      ui.statusDiv.classList.add("hidden");
      ui.answerDiv.textContent = err.message;
    } finally {
      isBusy = false;
      sendSpinner.classList.add("hidden");
      setChatEnabled(hasActiveDocument);
      chatMessages.scrollTop = chatMessages.scrollHeight;
      questionInput.focus();
    }
  }

  // Event Listeners
  btnRefreshHealth.addEventListener("click", checkHealth);

  topKInput.addEventListener("input", () => {
    topKValue.textContent = topKInput.value;
  });

  btnClearChat.addEventListener("click", () => {
    if (isBusy) return;
    chatMessages.innerHTML = `
      <div class="chat-welcome">
        <h3>Conversa limpa</h3>
        <p>Envie uma nova pergunta ou escolha uma das sugestões acima.</p>
      </div>
    `;
  });

  dropzone.addEventListener("click", () => pdfInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      pdfInput.click();
    }
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      selectFile(e.dataTransfer.files[0]);
    }
  });

  pdfInput.addEventListener("change", () => {
    if (pdfInput.files && pdfInput.files[0]) {
      selectFile(pdfInput.files[0]);
    }
  });

  btnUpload.addEventListener("click", handleUpload);

  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    sendQuestion(questionInput.value);
  });

  questionInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendQuestion(questionInput.value);
    }
  });

  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const q = chip.getAttribute("data-question") || "";
      if (!hasActiveDocument) {
        uploadStatus.textContent = "Carregue e indexe um edital em PDF antes de fazer perguntas.";
        uploadStatus.className = "status-feedback is-error";
        return;
      }
      sendQuestion(q);
    });
  });

  btnCloseModal.addEventListener("click", () => promptModal.close());
  btnCopyPrompt.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(modalPromptContent.textContent || "");
      btnCopyPrompt.textContent = "Copiado!";
      setTimeout(() => {
        btnCopyPrompt.textContent = "Copiar";
      }, 1500);
    } catch {
      btnCopyPrompt.textContent = "Falhou";
    }
  });

  // Inicialização
  checkHealth();
  loadActiveDocument();
})();
