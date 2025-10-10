const uploadForm = document.getElementById("upload-form");
const uploadTrigger = document.getElementById("upload-trigger");
const fileInput = document.getElementById("file-input");
const actionStatus = document.getElementById("action-status");
const previewFrame = document.getElementById("preview-frame");
const previewText = document.getElementById("preview-text");
const previewPlaceholder = document.getElementById("preview-placeholder");

const messagesEl = document.getElementById("messages");
const messageInput = document.getElementById("message");
const statusEl = document.getElementById("status");
const sendBtn = document.getElementById("send");
const chatDock = document.querySelector(".chat-dock");

const roles = {
  user: "Du",
  assistant: "Assistent",
};

let activeDocument = null;
let threadId = crypto.randomUUID().slice(0, 8);
let statusTimeout = null;

function setBusy(isLoading) {
  if (uploadForm) {
    uploadForm.classList.toggle("is-loading", isLoading);
  }
  if (uploadTrigger) {
    uploadTrigger.disabled = isLoading;
  }
}

function showStatus(message, { isError = false, persist = false } = {}) {
  if (!actionStatus) {
    return;
  }

  if (statusTimeout) {
    clearTimeout(statusTimeout);
    statusTimeout = null;
  }

  actionStatus.classList.toggle("error", Boolean(isError));

  if (!message) {
    actionStatus.hidden = true;
    actionStatus.textContent = "";
    return;
  }

  actionStatus.hidden = false;
  actionStatus.textContent = message;

  if (!persist) {
    statusTimeout = setTimeout(() => {
      actionStatus.hidden = true;
      actionStatus.textContent = "";
      actionStatus.classList.remove("error");
      statusTimeout = null;
    }, 4000);
  }
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
  if (!messagesEl) {
    return;
  }
  const node = createMessageElement(role, content);
  messagesEl.append(node);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function setLoading(isLoading) {
  if (statusEl) {
    statusEl.hidden = !isLoading;
  }
  if (chatDock) {
    chatDock.classList.toggle("loading", isLoading);
  }
  if (sendBtn) {
    sendBtn.disabled = isLoading;
  }
}

function clearPreview() {
  previewPlaceholder.hidden = false;
  previewFrame.hidden = true;
  previewFrame.src = "";
  previewText.hidden = true;
  previewText.textContent = "";
}

function showPreview(doc) {
  if (!doc) {
    clearPreview();
    return;
  }

  const suffix = doc.suffix ?? "";
  previewPlaceholder.hidden = true;

  if (suffix === ".pdf") {
    previewFrame.hidden = false;
    previewFrame.src = doc.url;
    previewText.hidden = true;
    previewText.textContent = "";
  } else {
    previewFrame.hidden = true;
    previewFrame.src = "";
    previewText.hidden = false;
    previewText.textContent = doc.preview ?? "(Keine Vorschau verfügbar)";
  }
}

function resetConversation() {
  threadId = crypto.randomUUID().slice(0, 8);
  if (!messagesEl) {
    return;
  }
  messagesEl.innerHTML = "";
}

function setActiveDocument(doc, { announce = true } = {}) {
  const changed = doc?.filename !== activeDocument?.filename;
  activeDocument = doc ?? null;

  if (!doc) {
    clearPreview();
    if (changed && announce) {
      resetConversation();
      showStatus("Lade ein Dokument hoch, um loszulegen.", {
        persist: true,
      });
    }
    return;
  }

  showPreview(doc);
  if (changed && announce) {
    resetConversation();
    showStatus(`Dokument geöffnet: ${doc.filename}`);
  }
}

async function fetchLatestDocument() {
  try {
    const response = await fetch("/documents");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const data = await response.json();
    const [first] = data.documents ?? [];
    if (first) {
      setActiveDocument(first, { announce: false });
    }
  } catch (error) {
    console.error(error);
    showStatus("Dokumente konnten nicht geladen werden.", {
      isError: true,
      persist: true,
    });
  }
}

function extractSuffix(filename) {
  const match = filename?.match(/\.[^\.]+$/);
  return match ? match[0].toLowerCase() : "";
}

async function handleUpload(file) {
  if (!file) {
    showStatus("Bitte wähle eine Datei aus.", { isError: true });
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  setBusy(true);
  showStatus("Datei wird verarbeitet …", { persist: true });

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
    const suffix = extractSuffix(doc.filename);
    const enriched = {
      ...doc,
      suffix,
    };

    setBusy(false);
    showStatus(`Upload erfolgreich – ${doc.filename}`);
    if (fileInput) {
      fileInput.value = "";
    }
    setActiveDocument(enriched);
  } catch (error) {
    console.error(error);
    setBusy(false);
    showStatus(error.message, { isError: true, persist: true });
  }
}

async function sendMessage(event) {
  event.preventDefault();
  if (!messageInput) {
    return;
  }

  const message = messageInput.value.trim();
  if (!message) {
    return;
  }

  if (!activeDocument) {
    showStatus("Bitte lade zuerst ein Dokument hoch.", {
      isError: true,
      persist: true,
    });
    messageInput.value = "";
    return;
  }

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
    showStatus(`Fehler beim Antworten: ${error.message}`, {
      isError: true,
      persist: true,
    });
  } finally {
    setLoading(false);
    messageInput?.focus();
  }
}

if (uploadTrigger && fileInput) {
  uploadTrigger.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    const [file] = fileInput.files ?? [];
    handleUpload(file);
  });
}

if (uploadForm) {
  uploadForm.addEventListener("submit", (event) => event.preventDefault());
}

const chatForm = document.getElementById("chat-form");
if (chatForm) {
  chatForm.addEventListener("submit", sendMessage);
}

resetConversation();
showStatus("Lade ein Dokument hoch, um zu starten.", { persist: true });
fetchLatestDocument();
messageInput?.focus?.();
