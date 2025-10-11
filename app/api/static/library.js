document.addEventListener("DOMContentLoaded", () => {
  const documentGrid = document.getElementById("document-grid");
  const uploadForm = document.getElementById("upload-form");
  const fileInput = document.getElementById("file-upload");
  const chatDrawer = document.getElementById("chat-drawer");
  const chatToggle = document.getElementById("chat-toggle");
  const chatClose = document.getElementById("chat-close");
  const chatHistory = document.getElementById("chat-history");
  const chatForm = document.getElementById("chat-form");
  const messageInput = document.getElementById("message");
  const aboutAllBtn = document.getElementById("about-all");

  // Upload handling
  uploadForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const file = fileInput.files[0];
    if (!file) {
      UIHelpers.showToast("Bitte wähle eine Datei aus.", "error");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("/upload", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      UIHelpers.debug("Upload erfolgreich", data);
      // Index ist serverseitig fertig – Seite neu laden, damit Liste/Cache frisch ist
      window.location.reload();
    } catch (error) {
      console.error("Fehler beim Hochladen der Datei:", error);
      UIHelpers.showToast("Fehler beim Hochladen.", "error");
    }
  });

  async function fetchDocuments() {
    try {
      const response = await fetch("/documents");
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      documentGrid.innerHTML = "";
      data.documents.forEach((doc) => {
        const docElement = document.createElement("a");
        docElement.href = `/reader?doc_id=${encodeURIComponent(doc.id)}`;
        docElement.className = "document-item";

        const thumbnail = document.createElement("div");
        thumbnail.className = "document-thumbnail";
        thumbnail.innerText = "PDF";

        const title = document.createElement("div");
        title.className = "document-title";
        title.textContent = doc.filename;

        docElement.appendChild(thumbnail);
        docElement.appendChild(title);
        documentGrid.appendChild(docElement);
      });
      UIHelpers.debug("Dokumente geladen", data.documents?.length);
    } catch (error) {
      console.error("Fehler beim Abrufen der Dokumente:", error);
      UIHelpers.showToast("Fehler beim Laden der Dokumente", "error");
    }
  }

  fetchDocuments();

  // Global chat UI
  function openDrawer() {
    chatDrawer.classList.add("open");
    chatDrawer.setAttribute("aria-hidden", "false");
  }
  function closeDrawer() {
    chatDrawer.classList.remove("open");
    chatDrawer.setAttribute("aria-hidden", "true");
  }
  if (chatToggle) chatToggle.addEventListener("click", openDrawer);
  if (chatClose) chatClose.addEventListener("click", closeDrawer);

  function appendMessage(role, content) {
    const messageElement = document.createElement("div");
    messageElement.className = `chat-message ${role}`;
    messageElement.innerText = content;
    chatHistory.appendChild(messageElement);
    chatHistory.scrollTop = chatHistory.scrollHeight;
  }

  chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const message = messageInput.value.trim();
    if (!message) return;
    appendMessage("user", message);
    messageInput.value = "";
    UIHelpers.debug("Global chat submit", { message });
    try {
      const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message,
          document_id: null, // global scope
          thread_id: "all",
        }),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      appendMessage("assistant", data.answer ?? "Keine Antwort erhalten.");
      UIHelpers.showToast("Antwort erhalten", "success");
    } catch (error) {
      console.error(error);
      UIHelpers.showToast("Chat-Fehler", "error");
      appendMessage("system", `Fehler: ${error.message}`);
    }
  });

  if (aboutAllBtn) {
    aboutAllBtn.addEventListener("click", async () => {
      appendMessage("user", "Über alle PDFs");
      UIHelpers.debug("Quick action: Über alle PDFs");
      try {
        const response = await fetch("/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message:
              "Gib mir eine kurze Übersicht zu den wichtigsten Themen und Dateien in allen indizierten PDFs.",
            document_id: null,
            thread_id: "all",
          }),
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        appendMessage("assistant", data.answer ?? "Keine Antwort erhalten.");
        UIHelpers.showToast("Übersicht erstellt", "success");
      } catch (error) {
        console.error(error);
        UIHelpers.showToast("Chat-Fehler", "error");
        appendMessage("system", `Fehler: ${error.message}`);
      }
    });
  }
});
