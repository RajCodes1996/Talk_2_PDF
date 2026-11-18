// src/api/static/js/app.js

async function postJSON(url, data) {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data)
  });
  if (!resp.ok) {
    const txt = await resp.text();
    throw new Error(`Error ${resp.status}: ${txt}`);
  }
  return await resp.json();
}

// Upload handler
document.getElementById("uploadBtn").addEventListener("click", async () => {
  const fileInput = document.getElementById("pdfFile");
  if (!fileInput.files.length) {
    alert("Please select a PDF file first.");
    return;
  }
  const file = fileInput.files[0];
  const form = new FormData();
  form.append("file", file);

  const resDiv = document.getElementById("uploadResult");
  resDiv.textContent = "Processing PDF (extracting text & building chunks)...";
  resDiv.style.color = "#60a5fa";

  try {
    const resp = await fetch("/upload", { method: "POST", body: form });
    const data = await resp.json();
    if (!resp.ok) throw new Error(JSON.stringify(data));
    
    window.currentDocId = data.doc_id;
    resDiv.textContent = `✓ Ready! Loaded "${file.name}" (doc_id = ${data.doc_id}${data.cached ? ', cached' : ''})`;
    resDiv.style.color = "var(--success)";
    
    document.getElementById("question-section").style.display = "block";
    document.getElementById("questionInput").focus();
  } catch (e) {
    resDiv.textContent = "Upload failed: " + e.message;
    resDiv.style.color = "#f87171";
  }
});

// Question handler
async function handleAsk() {
  const query = document.getElementById("questionInput").value.trim();
  if (!query) {
    alert("Please enter a question.");
    return;
  }
  if (!window.currentDocId) {
    alert("Please upload a PDF first.");
    return;
  }

  const answerBox = document.getElementById("answerBox");
  const metaBox = document.getElementById("metaBox");
  const askBtn = document.getElementById("askBtn");

  answerBox.style.display = "block";
  answerBox.textContent = "Thinking...";
  metaBox.textContent = "";
  askBtn.disabled = true;

  try {
    const result = await postJSON("/question", { doc_id: window.currentDocId, query });
    const ans = result.answer;

    if (ans && ans.answer) {
      answerBox.textContent = ans.answer;
      let meta = [];
      if (ans.type) meta.push(`Type: ${ans.type}`);
      if (ans.confidence !== undefined && ans.confidence !== null) {
        meta.push(`Confidence: ${(ans.confidence * 100).toFixed(0)}%`);
      }
      metaBox.textContent = meta.join(" • ");
    } else {
      answerBox.textContent = typeof ans === "object" ? JSON.stringify(ans, null, 2) : (ans || "No response generated.");
    }
  } catch (e) {
    answerBox.textContent = "Failed to get answer: " + e.message;
  } finally {
    askBtn.disabled = false;
  }
}

document.getElementById("askBtn").addEventListener("click", handleAsk);
document.getElementById("questionInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    handleAsk();
  }
});
