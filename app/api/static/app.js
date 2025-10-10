const uploadForm = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const uploadStatus = document.getElementById("upload-status");
const documentList = document.getElementById("document-list");
const documentEmptyState = document.getElementById("document-empty");
const previewFrame = document.getElementById("preview-frame");
const previewText = document.getElementById("preview-text");
const previewEmpty = document.getElementById("preview-empty");
const activeDocumentEl = document.getElementById("active-document");

const form = document.getElementById("chat-form");
const messagesEl = document.getElementById("messages");
const threadInput = document.getElementById("thread");
const messageInput = document.getElementById("message");
const statusEl = document.getElementById("status");
const sendBtn = document.getElementById("send");
const resetBtn = document.getElementById("reset");

let documents = [];
let activeDocument = null;

const roles = {
  user: "Du",
  assistant: "Assistent",
  system: "System",
};

function setUploadState(isLoading, message = "") {
  uploadStatus.hidden = !message && !isLoading;
  uploadStatus.textContent = message || (isLoading ? "Datei wird verarbeitet …" : "");
  uploadForm.classList.toggle("is-loading", isLoading);
}

function createMessageElement(role, content) {
  const li = document.createElement("li");
  li.classList.add("message", role);

  const roleEl = document.createElement("span");
  roleEl.className = "role";
  roleEl.textContent = roles[role] ?? role;

  const bodyEl = document.createElement("div");
  bodyEl.className = "content";
  bodyEl.innerText = content;

  li.append(roleEl, bodyEl);
  return li;
}

function appendMessage(role, content) {
  const messageNode = createMessageElement(role, content);
  messagesEl.appendChild(messageNode);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function setLoading(isLoading) {
  statusEl.hidden = !isLoading;
  sendBtn.disabled = isLoading;
  resetBtn.disabled = isLoading;
}

function renderDocumentList() {
  documentList.innerHTML = "";
  if (!documents.length) {
    documentEmptyState.hidden = false;
    return;
  }
  documentEmptyState.hidden = true;

  documents.forEach((doc) => {
    const item = document.createElement("li");
    item.className = "document-item";
    item.dataset.filename = doc.filename;

    const title = document.createElement("button");
    title.type = "button";
    title.className = "document-button";
    title.textContent = doc.filename;
    title.addEventListener("click", () => setActiveDocument(doc));

    const meta = document.createElement("span");
    meta.className = "document-meta";
    const sizeKb = Math.max(doc.size / 1024, 1).toFixed(1);
    const date = new Date(doc.updated_at * 1000).toLocaleString();
    meta.textContent = `${sizeKb} kB · ${date}`;

    item.append(title, meta);
    if (activeDocument && activeDocument.filename === doc.filename) {
      item.classList.add("is-active");
    }
    documentList.appendChild(item);
  });
}

function showPreview(doc) {
  if (!doc) {
    previewFrame.hidden = true;
    previewText.hidden = true;
    previewEmpty.hidden = false;
    previewFrame.src = "";
    previewText.textContent = "";
    return;
  }

  previewEmpty.hidden = true;
  const suffix = doc.suffix ?? "";
  if (suffix === ".pdf") {
    previewFrame.hidden = false;
    previewText.hidden = true;
    previewFrame.src = doc.url;
  } else {
    previewFrame.hidden = true;
    previewFrame.src = "";
    previewText.hidden = false;
    previewText.textContent = doc.preview ?? "(Keine Vorschau verfügbar)";
  }
}

function setActiveDocument(doc) {
  activeDocument = doc;
  activeDocumentEl.textContent = doc ? doc.filename : "–";
  showPreview(doc);
  renderDocumentList();
  if (doc) {
    appendMessage("system", `Aktives Dokument: ${doc.filename}`);
  }
}

async function fetchDocuments(selectFirst = false) {
  try {
    const response = await fetch("/documents");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const data = await response.json();
    documents = data.documents ?? [];
    renderDocumentList();
    if (selectFirst && documents.length) {
      setActiveDocument(documents[0]);
    }
  } catch (error) {
    console.error(error);
    setUploadState(false, "Dokumente konnten nicht geladen werden.");
  }
}

async function handleUpload(event) {
  event.preventDefault();
  const file = fileInput.files?.[0];
  if (!file) {
    setUploadState(false, "Bitte wähle eine Datei aus.");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  setUploadState(true);

  try {
    const response = await fetch("/upload", {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      const message = await response.json().catch(() => ({}));
      throw new Error(message.detail || `Upload fehlgeschlagen (HTTP ${response.status})`);
    }
    const doc = await response.json();
    setUploadState(false, `Upload erfolgreich: ${doc.filename}`);
    fileInput.value = "";
    await fetchDocuments();
    const current = documents.find((entry) => entry.filename === doc.filename);
    if (current) {
      setActiveDocument(current);
    }
  } catch (error) {
    console.error(error);
    setUploadState(false, error.message);
  }
}

async function sendMessage(event) {
  event.preventDefault();
  const message = messageInput.value.trim();
  if (!message) {
    return;
  }

  const threadId = threadInput.value.trim() || "default";
  appendMessage("user", message);
  messageInput.value = "";
  setLoading(true);

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        thread_id: threadId,
        message,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    appendMessage("assistant", data.answer ?? "Keine Antwort erhalten.");
  } catch (error) {
    console.error(error);
    appendMessage("system", `Fehler: ${error.message}`);
    messagesEl.lastChild?.classList.add("error");
  } finally {
    setLoading(false);
    messageInput.focus();
  }
}

function resetThread() {
  messagesEl.innerHTML = "";
  const threadId = crypto.randomUUID().slice(0, 8);
  threadInput.value = threadId;
  appendMessage("system", `Neuer Thread angelegt: ${threadId}`);
  if (activeDocument) {
    appendMessage("system", `Aktives Dokument: ${activeDocument.filename}`);
  }
}

uploadForm.addEventListener("submit", handleUpload);
form.addEventListener("submit", sendMessage);
resetBtn.addEventListener("click", resetThread);

resetThread();
appendMessage(
  "system",
  "Willkommen! Lade ein Dokument hoch oder wähle eines aus der Liste, um darüber zu chatten.",
);
fetchDocuments(true);
messageInput.focus();
