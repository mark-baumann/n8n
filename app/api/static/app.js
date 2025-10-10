const form = document.getElementById("chat-form");
const messagesEl = document.getElementById("messages");
const threadInput = document.getElementById("thread");
const messageInput = document.getElementById("message");
const statusEl = document.getElementById("status");
const sendBtn = document.getElementById("send");
const resetBtn = document.getElementById("reset");

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
}

form.addEventListener("submit", sendMessage);
resetBtn.addEventListener("click", resetThread);

appendMessage(
  "system",
  "Willkommen! Stelle deine Frage oder wähle eine eigene Thread-ID, um einen bestehenden Verlauf fortzusetzen."
);
messageInput.focus();
