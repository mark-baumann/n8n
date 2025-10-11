document.addEventListener("DOMContentLoaded", () => {
    const pdfViewer = document.getElementById("pdf-viewer");
    const chatDrawer = document.getElementById("chat-drawer");
    const chatHistory = document.getElementById("chat-history");
    const chatForm = document.getElementById("chat-form");
    const messageInput = document.getElementById("message");
    const chatToggle = document.getElementById("chat-toggle");
    const chatClose = document.getElementById("chat-close");
    const docLabel = document.getElementById("doc-label");

    const urlParams = new URLSearchParams(window.location.search);
    const documentName = urlParams.get("document");
    let threadId = documentName ? `doc:${documentName}` : "default";

    if (documentName) {
        pdfViewer.src = `/data/${documentName}`;
        if (docLabel) docLabel.textContent = documentName;
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
                    document: documentName || null,
                    thread_id: threadId,
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
        }
    });

    function appendMessage(role, content) {
        const messageElement = document.createElement("div");
        messageElement.className = `chat-message ${role}`;
        messageElement.innerText = content;
        chatHistory.appendChild(messageElement);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }
});
