document.addEventListener("DOMContentLoaded", () => {
    const pdfViewer = document.getElementById("pdf-viewer");
    const chatDrawer = document.getElementById("chat-drawer");
    const chatHistory = document.getElementById("chat-history");
    const chatForm = document.getElementById("chat-form");
    const messageInput = document.getElementById("message");
    const chatToggle = document.getElementById("chat-toggle");
    const chatClose = document.getElementById("chat-close");
    const docLabel = document.getElementById("doc-label");
    const aboutDocBtn = document.getElementById("about-doc");

    const urlParams = new URLSearchParams(window.location.search);
    const documentId = urlParams.get("doc_id");
    let threadId = documentId ? `doc:${documentId}` : "default";
    UIHelpers.debug("Reader init", { documentId, threadId });

    if (documentId) {
        pdfViewer.src = `/data_by_id/${encodeURIComponent(documentId)}`;
        // Fetch filename for label
        fetch(`/document/${encodeURIComponent(documentId)}`)
            .then(r => r.ok ? r.json() : null)
            .then(info => {
                if (info && docLabel) {
                    docLabel.textContent = info.filename;
                    UIHelpers.showToast(`PDF geladen: ${info.filename}`, 'info');
                }
            })
            .catch((e) => { UIHelpers.debug('doc meta fetch error', e); });
    }

    // Toggle chat drawer
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

    chatForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        const message = messageInput.value.trim();
        if (!message) {
            return;
        }

        appendMessage("user", message);
        messageInput.value = "";

        try {
            const response = await fetch("/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    message,
                    document_id: documentId || null,
                    thread_id: threadId,
                }),
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const data = await response.json();
            appendMessage("assistant", data.answer ?? "Keine Antwort erhalten.");
            UIHelpers.showToast("Antwort erhalten", "success");
        } catch (error) {
            console.error(error);
            UIHelpers.showToast("Chat-Fehler", "error");
            appendMessage("system", `Fehler: ${error.message}`);
        }
    });

    // About this PDF quick action
    if (aboutDocBtn) {
        aboutDocBtn.addEventListener("click", async () => {
            appendMessage("user", "Über dieses PDF");
            UIHelpers.debug("Quick action: Über dieses PDF", { documentId });
            try {
                const response = await fetch("/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        message: "Fasse das aktuelle PDF kurz zusammen: Thema, Zweck, wichtigste Punkte und Struktur.",
                        document_id: documentId || null,
                        thread_id: threadId,
                    }),
                });
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                const data = await response.json();
                appendMessage("assistant", data.answer ?? "Keine Antwort erhalten.");
                UIHelpers.showToast("Zusammenfassung erstellt", "success");
            } catch (error) {
                console.error(error);
                UIHelpers.showToast("Chat-Fehler", "error");
                appendMessage("system", `Fehler: ${error.message}`);
            }
        });
    }

    function appendMessage(role, content) {
        const messageElement = document.createElement("div");
        messageElement.className = `chat-message ${role}`;
        messageElement.innerText = content;
        chatHistory.appendChild(messageElement);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }
});
