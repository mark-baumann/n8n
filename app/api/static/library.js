document.addEventListener("DOMContentLoaded", () => {
    const documentGrid = document.getElementById("document-grid");
    const uploadForm = document.getElementById("upload-form");
    const fileInput = document.getElementById("file-upload");

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

            await response.json();
            fetchDocuments();
        } catch (error) {
            console.error("Fehler beim Hochladen der Datei:", error);
            alert("Fehler beim Hochladen der Datei.");
        }
    });

    async function fetchDocuments() {
        try {
            const response = await fetch("/documents");
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            const data = await response.json();
            documentGrid.innerHTML = "";
            data.documents.forEach(doc => {
                const docElement = document.createElement("a");
                docElement.href = `/reader?document=${encodeURIComponent(doc)}`;
                docElement.className = "document-item";
                
                const thumbnail = document.createElement("div");
                thumbnail.className = "document-thumbnail";
                thumbnail.innerText = "PDF"; // Placeholder

                const title = document.createElement("div");
                title.className = "document-title";
                title.textContent = doc;

                docElement.appendChild(thumbnail);
                docElement.appendChild(title);
                documentGrid.appendChild(docElement);
            });
        } catch (error) {
            console.error("Fehler beim Abrufen der Dokumente:", error);
        }
    }

    fetchDocuments();
});