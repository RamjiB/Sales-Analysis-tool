const uploadLog = document.getElementById("uploadLog");
const datasetSelect = document.getElementById("datasetSelect");
const chatBox = document.getElementById("chat");
const ingestProgress = document.getElementById("ingestProgress");
const ingestStage = document.getElementById("ingestStage");
const statusPill = document.getElementById("statusPill");
const chatLockedHint = document.getElementById("chatLockedHint");
const messageInput = document.getElementById("message");
const sendBtn = document.getElementById("sendBtn");

const sessionId = crypto.randomUUID();
let ws;
let statusTimer = null;
let wsConnected = false;
let activeAssistantBlock = null;

function logUpload(obj) {
  uploadLog.textContent = typeof obj === "string" ? obj : JSON.stringify(obj, null, 2);
}

function createBubble(role, text) {
  const bubble = document.createElement("div");
  bubble.className = `bubble ${role}`;

  const label = document.createElement("div");
  label.className = "bubble-label";
  label.textContent = role === "user" ? "You" : "Assistant";

  const body = document.createElement("div");
  if (role === "assistant") {
    body.innerHTML = renderMarkdown(text);
  } else {
    body.textContent = text;
  }

  bubble.appendChild(label);
  bubble.appendChild(body);
  chatBox.appendChild(bubble);
  chatBox.scrollTop = chatBox.scrollHeight;
  return { bubble, body };
}

function escapeHtml(raw) {
  return String(raw || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function renderMarkdown(md) {
  let html = escapeHtml(md);
  html = html.replace(/^### (.*)$/gm, "<h4>$1</h4>");
  html = html.replace(/^## (.*)$/gm, "<h3>$1</h3>");
  html = html.replace(/^# (.*)$/gm, "<h2>$1</h2>");
  html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*(.*?)\*/g, "<em>$1</em>");
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/^\- (.*)$/gm, "<li>$1</li>");
  html = html.replace(/(<li>.*<\/li>)/gs, "<ul>$1</ul>");
  html = html.replace(/\n/g, "<br>");
  return html;
}

function createAssistantThinkingBlock() {
  const { bubble, body } = createBubble("assistant", "Thinking...");

  const thinking = document.createElement("details");
  thinking.className = "thinking-box";
  thinking.open = true;

  const summary = document.createElement("summary");
  summary.textContent = "Thinking";
  thinking.appendChild(summary);

  const traceContainer = document.createElement("div");
  thinking.appendChild(traceContainer);

  const resultPreview = document.createElement("pre");
  resultPreview.textContent = "";
  thinking.appendChild(resultPreview);

  bubble.appendChild(thinking);

  activeAssistantBlock = { bubble, body, thinking, traceContainer, resultPreview };
}

function appendThinkingStep(step) {
  if (!activeAssistantBlock) return;
  const item = document.createElement("div");
  item.className = "thinking-item";
  const stage = step.stage ? `[${step.stage}] ` : "";
  item.textContent = `${stage}${step.message || ""}`;
  if (step.details) {
    const d = document.createElement("pre");
    d.textContent = JSON.stringify(step.details, null, 2);
    item.appendChild(d);
  }
  activeAssistantBlock.traceContainer.appendChild(item);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function finalizeAssistant(answer, mode, sql, rows) {
  if (!activeAssistantBlock) {
    createAssistantThinkingBlock();
  }
  activeAssistantBlock.body.innerHTML = renderMarkdown(answer || "No answer generated.");
  activeAssistantBlock.resultPreview.textContent = JSON.stringify(
    {
      mode,
      sql,
      rows_preview: Array.isArray(rows) ? rows.slice(0, 10) : [],
    },
    null,
    2,
  );
  activeAssistantBlock.thinking.open = false;
  activeAssistantBlock = null;
}

function setChatEnabled(enabled) {
  messageInput.disabled = !enabled;
  sendBtn.disabled = !enabled;
  chatLockedHint.style.display = enabled ? "none" : "block";
  if (!enabled && ws && ws.readyState === WebSocket.OPEN) {
    ws.close();
    ws = null;
    wsConnected = false;
  }
  if (enabled && !wsConnected) {
    connectWS();
  }
}

async function refreshDatasetsOnly() {
  const res = await fetch("/datasets");
  const payload = await res.json();
  const datasets = payload.datasets || {};
  const active = payload.active_dataset_id;
  datasetSelect.innerHTML = "";
  for (const [id, data] of Object.entries(datasets)) {
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = `${data.filename} | ${data.status}`;
    if (id === active) opt.selected = true;
    datasetSelect.appendChild(opt);
  }
}

async function refreshIngestionStatus(datasetId) {
  if (!datasetId) return;
  const res = await fetch(`/ingest/status/${datasetId}`);
  const payload = await res.json();
  const job = payload.job || {};
  let progress = Number(job.progress || 0);
  const stage = job.stage || "unknown";
  const state = payload.status || "unknown";

  if (state === "ingested" && stage === "unknown") {
    progress = 100;
  }
  ingestProgress.value = progress;
  ingestStage.textContent = `${stage}: ${job.message || ""}`;
  statusPill.textContent = state;
  statusPill.className = `pill ${state}`;
  setChatEnabled(state === "ingested");

  const done = job.done === true || stage === "ingested" || stage === "failed";
  if (done) {
    ingestProgress.value = 100;
    if (statusTimer) {
      clearInterval(statusTimer);
      statusTimer = null;
    }
    await refreshDatasetsOnly();
  }
}

function watchProgress(datasetId) {
  if (statusTimer) clearInterval(statusTimer);
  ingestProgress.value = 0;
  refreshIngestionStatus(datasetId);
  statusTimer = setInterval(() => refreshIngestionStatus(datasetId), 1200);
}

async function refreshDatasets() {
  await refreshDatasetsOnly();
  const id = datasetSelect.value;
  if (id) {
    await refreshIngestionStatus(id);
  }
}

async function uploadFile() {
  const file = document.getElementById("fileInput").files[0];
  if (!file) return logUpload("Choose a file first.");

  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch("/upload", { method: "POST", body: formData });
  const payload = await res.json();
  logUpload(payload);
  await refreshDatasets();
  const datasetId = payload?.dataset?.dataset_id;
  if (datasetId) watchProgress(datasetId);
}

function connectWS() {
  if (wsConnected) return;
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${protocol}://${location.host}/ws/chat`);

  ws.onopen = () => {
    wsConnected = true;
    createBubble("assistant", "Connected. Ask any sales question.");
  };

  ws.onmessage = (evt) => {
    const msg = JSON.parse(evt.data);
    if (msg.type === "error") {
      createBubble("assistant", `Error: ${msg.message}`);
      return;
    }
    if (msg.type === "thinker_step") {
      appendThinkingStep(msg.payload || {});
      return;
    }
    if (msg.type === "final_answer") {
      const p = msg.payload || {};
      finalizeAssistant(p.answer || "", p.mode, p.sql, p.rows);
    }
  };

  ws.onclose = () => {
    wsConnected = false;
  };
}

function sendMessage() {
  const text = messageInput.value.trim();
  if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;
  const datasetId = datasetSelect.value || null;

  createBubble("user", text);
  createAssistantThinkingBlock();

  ws.send(
    JSON.stringify({
      type: "user_message",
      session_id: sessionId,
      dataset_id: datasetId,
      text,
    }),
  );

  messageInput.value = "";
}

document.getElementById("uploadBtn").addEventListener("click", uploadFile);
document.getElementById("refreshBtn").addEventListener("click", refreshDatasets);
document.getElementById("sendBtn").addEventListener("click", sendMessage);
messageInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendMessage();
});
datasetSelect.addEventListener("change", () => {
  const id = datasetSelect.value;
  if (id) watchProgress(id);
});

refreshDatasets();
setChatEnabled(false);
