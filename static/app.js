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

  const chatMessages = document.getElementById("chat-messages");
  const chatForm = document.getElementById("chat-form");
  const questionInput = document.getElementById("question-input");
  const btnSend = document.getElementById("btn-send");
  const sendSpinner = document.getElementById("send-spinner");

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

  async function checkHealth() {
    healthPill.textContent = "Verificando Ollama...";
    healthPill.className = "health-pill pill-loading";

    try {
      const res = await fetch("/api/health");
      const data = await res.json();

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

  function appendLoadingMessage() {
    const div = document.createElement("div");
    div.className = "msg-assistant";
    div.textContent = "Buscando trechos k-NN no ChromaDB e gerando resposta...";
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return div;
  }

  function renderAssistantCard(container, payload) {
    container.innerHTML = "";

    const answerDiv = document.createElement("div");
    answerDiv.className = "assistant-answer";
    answerDiv.innerHTML = formatAnswerHtml(payload.answer);
    container.appendChild(answerDiv);

    const knnPanel = document.createElement("div");
    knnPanel.className = "knn-panel";

    const knnHeader = document.createElement("div");
    knnHeader.className = "knn-header";

    const title = document.createElement("span");
    title.className = "knn-title";
    title.textContent = `Inspeção Vetorial · Top-${payload.sources.length} Chunks Recuperados (Cosseno)`;

    const btnPrompt = document.createElement("button");
    btnPrompt.type = "button";
    btnPrompt.className = "btn-secondary";
    btnPrompt.textContent = "Ver Prompt Aumentado";
    btnPrompt.addEventListener("click", () => {
      modalPromptContent.textContent = payload.augmented_prompt;
      promptModal.showModal();
    });

    knnHeader.appendChild(title);
    knnHeader.appendChild(btnPrompt);
    knnPanel.appendChild(knnHeader);

    const grid = document.createElement("div");
    grid.className = "sources-grid";

    payload.sources.forEach((src, index) => {
      const pct = Math.max(0, Math.min(100, Math.round(src.similarity_score * 100)));
      const card = document.createElement("article");
      card.className = "source-card";
      card.innerHTML = `
        <div class="source-meta">
          <span class="source-page">#${index + 1} · Página ${escapeHtml(String(src.page))}</span>
          <span class="source-score">cos: ${src.similarity_score.toFixed(4)} (${pct}%)</span>
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
    container.appendChild(knnPanel);
  }

  async function sendQuestion(questionText) {
    const question = questionText.trim();
    if (!question || !hasActiveDocument || isBusy) return;

    isBusy = true;
    setChatEnabled(true);
    sendSpinner.classList.remove("hidden");
    questionInput.value = "";

    appendUserMessage(question);
    const assistantBubble = appendLoadingMessage();

    try {
      const topK = Number(topKInput.value) || 4;
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, top_k: topK }),
      });
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || "Erro ao consultar o edital.");
      }

      renderAssistantCard(assistantBubble, data);
    } catch (err) {
      assistantBubble.classList.add("is-error");
      assistantBubble.textContent = err.message;
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
