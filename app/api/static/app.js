const form = document.getElementById("chat-form");
const messagesEl = document.getElementById("messages");
const messageInput = document.getElementById("message");
const uploadForm = document.getElementById("upload-form");
const fileInput = document.getElementById("file-upload");
const pdfViewer = document.getElementById("pdf-viewer");
const documentList = document.getElementById("document-list");

async function fetchDocuments() {
  try {
    const response = await fetch("/documents");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const data = await response.json();
    documentList.innerHTML = "";
    data.documents.forEach(doc => {
      const li = document.createElement("li");
      li.textContent = doc;
      li.addEventListener("click", () => {
        pdfViewer.src = `/data/${doc}`;
      });
      documentList.appendChild(li);
    });
  } catch (error) {
    console.error("Fehler beim Abrufen der Dokumente:", error);
  }
}

const roles = {
  user: "Du",
  assistant: "Assistent",
  system: "System",
};

function createMessageElement(role, content) {
  const li = document.createElement("li");
  li.classList.add("message");
  li.classList.add(role);

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

async function sendMessage(event) {
  event.preventDefault();
  const message = messageInput.value.trim();
  if (!message) {
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

form.addEventListener("submit", sendMessage);

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file) {
    alert("Bitte wähle eine Datei aus.");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/upload", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    pdfViewer.src = `/data/${data.filename}`;
    alert("Datei erfolgreich hochgeladen.");
    fetchDocuments();
  } catch (error) {
    console.error(error);
    alert("Fehler beim Hochladen der Datei.");
  }
});

appendMessage(
  "system",
  "Willkommen! Stelle deine Frage oder lade ein Dokument hoch, um zu starten."
);
fetchDocuments();
messageInput.focus();
