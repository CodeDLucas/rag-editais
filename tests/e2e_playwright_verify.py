"""Verificação ponta a ponta (E2E) da interface Beta usando Playwright + Uvicorn + Ollama reais."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

PROJECT_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = Path(
    os.environ.get(
        "ARTIFACT_DIR",
        r"C:\Users\LucasLD\.gemini\antigravity-cli\brain\4d1d23af-9ea7-4aaa-8a62-eb2d673e96c5",
    )
)
PORT = 8011
BASE_URL = f"http://127.0.0.1:{PORT}"


def wait_for_server(url: str, timeout: float = 20.0) -> None:
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/api/health", timeout=2) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(0.4)
    raise RuntimeError("Uvicorn server did not become ready in time.")


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    uvicorn_exe = PROJECT_DIR / ".venv" / "Scripts" / "uvicorn.exe"

    proc = subprocess.Popen(
        [str(uvicorn_exe), "app.main:app", "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=str(PROJECT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    try:
        wait_for_server(BASE_URL)
        page_errors: list[str] = []
        console_errors: list[str] = []

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            page.on("pageerror", lambda err: page_errors.append(str(err)))
            page.on(
                "console",
                lambda msg: console_errors.append(msg.text) if msg.type == "error" else None,
            )

            page.goto(BASE_URL, wait_until="networkidle")

            # 1) Verifica status do Ollama e edital ativo carregado
            page.wait_for_selector("#health-pill.pill-ok", timeout=15000)
            page.wait_for_selector("#doc-summary:not(.hidden)", timeout=10000)
            doc_name = page.locator("#doc-filename").inner_text()
            doc_pages = page.locator("#doc-pages").inner_text()
            doc_chunks = page.locator("#doc-chunks").inner_text()
            models = page.locator("#llm-model-select option").all_inner_texts()
            print(f"[E2E] Active doc: {doc_name} ({doc_pages} pages, {doc_chunks} chunks)")
            print(f"[E2E] Models in dropdown: {models}")

            page.screenshot(path=str(ARTIFACT_DIR / "ui_beta_initial.png"), full_page=True)

            # 2) Clica na pergunta sugerida de prazo de validade (streaming SSE + busca híbrida)
            page.locator('.chip[data-question="Qual o prazo de validade do concurso?"]').click()

            # Aguarda painel k-NN aparecer e telemetria final (done) ser exibida
            page.wait_for_selector(".knn-panel:not(.hidden)", timeout=20000)
            page.wait_for_selector(".telemetry-bar:not(.hidden)", timeout=90000)

            answer_text = page.locator(".assistant-answer").last.inner_text()
            telemetry_text = page.locator(".telemetry-bar").last.inner_text()
            source_cards_count = page.locator(".source-card").count()
            print(f"[E2E] Answer: {answer_text[:180].replace(chr(10), ' ')}")
            print(f"[E2E] Telemetry: {telemetry_text}")
            print(f"[E2E] Source cards rendered: {source_cards_count}")
            assert source_cards_count == 4, f"Expected 4 source cards, got {source_cards_count}"
            assert "bm25:" in page.locator(".sources-grid").last.inner_text()

            page.screenshot(path=str(ARTIFACT_DIR / "ui_beta_chat_knn.png"), full_page=True)

            # 3) Abre o modal do Prompt Aumentado e captura screenshot
            page.locator('button:has-text("Ver Prompt Aumentado")').last.click()
            page.wait_for_selector("#prompt-modal[open]", timeout=5000)
            modal_text = page.locator("#modal-prompt-content").inner_text()
            assert "SYSTEM PROMPT" in modal_text
            page.screenshot(path=str(ARTIFACT_DIR / "ui_beta_prompt_modal.png"), full_page=True)
            page.locator("#btn-close-modal").click()

            assert not page_errors, f"Page errors detected: {page_errors}"
            assert not console_errors, f"Console errors detected: {console_errors}"
            print("[E2E] Playwright verification PASSED with 0 console/page errors!")
            browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[E2E] FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
