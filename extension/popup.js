const apiBaseInput = document.getElementById("api-base");
const analyzePageButton = document.getElementById("analyze-page");
const analyzeSelectionButton = document.getElementById("analyze-selection");
const statusBox = document.getElementById("status");
const resultBox = document.getElementById("result");

chrome.storage.local.get({ apiBase: "http://127.0.0.1:5000" }, ({ apiBase }) => {
  apiBaseInput.value = apiBase;
});

apiBaseInput.addEventListener("change", () => {
  chrome.storage.local.set({ apiBase: normalizedBase() });
});

analyzePageButton.addEventListener("click", () => analyze("page"));
analyzeSelectionButton.addEventListener("click", () => analyze("selection"));

async function analyze(mode) {
  setBusy(true, "Sayfa metni okunuyor...");
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const [{ result }] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      files: ["content.js"],
    });
    const text = mode === "selection" ? result.selectionText : result.articleText;
    if (!text || text.trim().split(/\s+/).length < 8) {
      throw new Error("Yeterli haber metni bulunamadı. Metni seçip tekrar deneyin.");
    }

    setBusy(true, "Logos analiz ediyor...");
    const response = await fetch(`${normalizedBase()}/api/fact-checks`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: result.url, text }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Teyit analizi başarısız oldu.");
    }
    renderResult(payload);
  } catch (error) {
    statusBox.textContent = error.message || "Beklenmeyen hata.";
    resultBox.hidden = true;
  } finally {
    setBusy(false);
  }
}

function renderResult(payload) {
  const impact = payload.impact || {};
  document.getElementById("logic-score").textContent = `${impact.logic_score ?? 0}%`;
  document.getElementById("claim-count").textContent = impact.claim_count ?? 0;
  document.getElementById("source-count").textContent = impact.source_count ?? 0;
  document.getElementById("flag-count").textContent = (payload.findings || []).length;
  document.getElementById("summary").textContent = payload.run?.summary || "";
  document.getElementById("open-graph").href = `${normalizedBase()}/fact-check?run_id=${payload.run.id}`;
  statusBox.textContent = "Analiz tamamlandı.";
  resultBox.hidden = false;
}

function setBusy(isBusy, message = "Hazır.") {
  analyzePageButton.disabled = isBusy;
  analyzeSelectionButton.disabled = isBusy;
  statusBox.textContent = message;
}

function normalizedBase() {
  return apiBaseInput.value.replace(/\/+$/, "");
}
