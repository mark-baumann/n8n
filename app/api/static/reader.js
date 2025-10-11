document.addEventListener("DOMContentLoaded", () => {
    const pdfViewer = document.getElementById("pdf-viewer");
    const chatHistory = document.getElementById("chat-history");
    const chatForm = document.getElementById("chat-form");
    const messageInput = document.getElementById("message");

    const urlParams = new URLSearchParams(window.location.search);
    const documentName = urlParams.get("document");

    if (documentName) {
        pdfViewer.src = `/data/${documentName}`;
    }

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