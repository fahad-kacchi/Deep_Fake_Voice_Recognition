/**
 * Deep Fake Voice Recognition - Client-Side Controller
 * Handles drag-and-drop uploads, validation, audio previews, loading states, and async prediction.
 */

document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById("audio-upload");
    const dropZone = document.getElementById("drop-zone");
    const fileInfoPanel = document.getElementById("file-info-panel");
    const fileNameDisplay = document.getElementById("file-name-display");
    const fileSizeDisplay = document.getElementById("file-size-display");
    const clearFileBtn = document.getElementById("clear-file-btn");
    const audioPreview = document.getElementById("audio-preview");
    const submitBtn = document.getElementById("submit-btn");
    const btnText = document.getElementById("btn-text");
    const loadingState = document.getElementById("loading-state");
    const clientError = document.getElementById("client-error");
    const resultContainer = document.getElementById("result-container");
    const uploadForm = document.getElementById("upload-form");

    const ALLOWED_EXTENSIONS = ["wav", "mp3", "ogg"];
    const MAX_FILE_SIZE_BYTES = 16 * 1024 * 1024; // 16 MB

    let isSubmitting = false;
    let currentObjectUrl = null;

    // Helper: format bytes into human-readable string
    function formatBytes(bytes) {
        if (bytes === 0) return "0 Bytes";
        const k = 1024;
        const sizes = ["Bytes", "KB", "MB"];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
    }

    // Show client-side error
    function showError(message) {
        clientError.textContent = message;
        clientError.classList.remove("hidden");
    }

    // Clear client-side error
    function clearError() {
        clientError.textContent = "";
        clientError.classList.add("hidden");
    }

    // Reset file selection
    function resetFileSelection() {
        fileInput.value = "";
        if (currentObjectUrl) {
            URL.revokeObjectURL(currentObjectUrl);
            currentObjectUrl = null;
        }
        audioPreview.src = "";
        fileInfoPanel.classList.add("hidden");
        submitBtn.disabled = true;
        clearError();
    }

    // Validate and process selected file
    function handleFile(file) {
        clearError();

        if (!file) {
            resetFileSelection();
            return;
        }

        const ext = file.name.split(".").pop().toLowerCase();
        if (!ALLOWED_EXTENSIONS.includes(ext)) {
            showError(`Unsupported file format (.${ext}). Supported formats are: WAV, MP3, and OGG.`);
            resetFileSelection();
            return;
        }

        if (file.size === 0) {
            showError("The selected file is empty (0 bytes). Please choose a valid audio recording.");
            resetFileSelection();
            return;
        }

        if (file.size > MAX_FILE_SIZE_BYTES) {
            showError(`File size (${formatBytes(file.size)}) exceeds the maximum 16 MB limit.`);
            resetFileSelection();
            return;
        }

        // Display file details
        fileNameDisplay.textContent = file.name;
        fileSizeDisplay.textContent = formatBytes(file.size);
        fileInfoPanel.classList.remove("hidden");

        // Audio preview playback
        if (currentObjectUrl) {
            URL.revokeObjectURL(currentObjectUrl);
        }
        currentObjectUrl = URL.createObjectURL(file);
        audioPreview.src = currentObjectUrl;

        // Enable submit button
        submitBtn.disabled = false;
    }

    // File input change listener
    fileInput.addEventListener("change", function () {
        if (fileInput.files && fileInput.files.length > 0) {
            handleFile(fileInput.files[0]);
        }
    });

    // Clear file button
    if (clearFileBtn) {
        clearFileBtn.addEventListener("click", function () {
            resetFileSelection();
        });
    }

    // Drag and Drop support
    if (dropZone) {
        ["dragenter", "dragover"].forEach(eventName => {
            dropZone.addEventListener(eventName, function (e) {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.add("drag-active");
            });
        });

        ["dragleave", "drop"].forEach(eventName => {
            dropZone.addEventListener(eventName, function (e) {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.remove("drag-active");
            });
        });

        dropZone.addEventListener("drop", function (e) {
            if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                const droppedFile = e.dataTransfer.files[0];
                fileInput.files = e.dataTransfer.files;
                handleFile(droppedFile);
            }
        });

        // Click on dropzone triggers file picker
        dropZone.addEventListener("click", function (e) {
            // Prevent double-trigger if clicking directly on child interactive elements
            if (e.target !== fileInput) {
                fileInput.click();
            }
        });

        // Keyboard accessibility (Enter / Space)
        dropZone.addEventListener("keydown", function (e) {
            if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                fileInput.click();
            }
        });
    }

    // Render result card in UI
    function renderResult(message, isSuccess = true) {
        resultContainer.innerHTML = "";
        const card = document.createElement("div");
        card.className = "result-card";

        if (isSuccess && message.includes("REAL")) {
            card.classList.add("result-real");
            card.setAttribute("role", "status");
            card.innerHTML = `
                <div class="result-badge-pill badge-real">HUMAN VOICE DETECTED</div>
                <h2 class="result-heading">Audio is REAL</h2>
                <p class="result-desc">Acoustic frequency distributions align with genuine human vocal characteristics.</p>
                <span class="disclaimer-note">ML model prediction based on 40-dimensional Mel-frequency cepstral coefficients.</span>
            `;
        } else if (isSuccess && message.includes("FAKE")) {
            card.classList.add("result-fake");
            card.setAttribute("role", "status");
            card.innerHTML = `
                <div class="result-badge-pill badge-fake">SYNTHETIC SPEECH DETECTED</div>
                <h2 class="result-heading">Audio is FAKE</h2>
                <p class="result-desc">Acoustic anomalies detected; spectral characteristics indicate AI synthesis or voice conversion.</p>
                <span class="disclaimer-note">ML model prediction based on 40-dimensional Mel-frequency cepstral coefficients.</span>
            `;
        } else {
            card.classList.add("result-notice");
            card.setAttribute("role", "alert");
            card.innerHTML = `
                <div class="result-badge-pill badge-notice">PROCESSING NOTICE</div>
                <h2 class="result-heading">${escapeHtml(message)}</h2>
                <p class="result-desc">Please verify the audio file meets the requirements and try again.</p>
            `;
        }
        resultContainer.appendChild(card);
    }

    // Helper: HTML escape
    function escapeHtml(text) {
        const div = document.createElement("div");
        div.textContent = text;
        return div.innerHTML;
    }

    // Asynchronous Form Submission
    uploadForm.addEventListener("submit", async function (e) {
        e.preventDefault();

        if (isSubmitting) {
            return; // Prevent duplicate concurrent submissions
        }

        if (!fileInput.files || fileInput.files.length === 0) {
            showError("Please select an audio file first.");
            return;
        }

        const file = fileInput.files[0];
        if (file.size > MAX_FILE_SIZE_BYTES) {
            showError(`File size exceeds 16 MB. Please select a smaller file.`);
            return;
        }

        // Set loading / submission state
        isSubmitting = true;
        submitBtn.disabled = true;
        btnText.textContent = "Analyzing Audio...";
        loadingState.classList.remove("hidden");
        clearError();
        resultContainer.innerHTML = "";

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch("/upload", {
                method: "POST",
                body: formData
            });

            const html = await response.text();
            
            // Extract prediction_text from returned HTML
            const parser = new DOMParser();
            const doc = parser.parseFromString(html, "text/html");
            const resultDesc = doc.querySelector(".result-heading");
            const legacyP = doc.querySelector(".content p");

            let resultMessage = "Processed";
            if (resultDesc) {
                resultMessage = resultDesc.textContent.trim();
            } else if (legacyP) {
                resultMessage = legacyP.textContent.trim();
            }

            renderResult(resultMessage, response.ok);

        } catch (error) {
            renderResult("Network error: Unable to connect to the server. Please ensure the backend application is running.", false);
        } finally {
            // Restore button and controls
            isSubmitting = false;
            submitBtn.disabled = false;
            btnText.textContent = "Check Audio";
            loadingState.classList.add("hidden");
        }
    });
});
