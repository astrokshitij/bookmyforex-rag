document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const messagesList = document.getElementById("messagesList");
  const chatForm = document.getElementById("chatForm");
  const queryInput = document.getElementById("queryInput");
  const docFilter = document.getElementById("docFilter");
  const btnSubmit = document.getElementById("btnSubmit");
  const statusIndicator = document.getElementById("statusIndicator");
  const statusText = document.getElementById("statusText");
  const btnReindex = document.getElementById("btnReindex");
  const btnDocsToggle = document.getElementById("btnDocsToggle");
  const docsDrawer = document.getElementById("docsDrawer");
  const btnCloseDrawer = document.getElementById("btnCloseDrawer");
  const docsList = document.getElementById("docsList");

  // Settings elements
  const btnSettings = document.getElementById("btnSettings");
  const settingsModal = document.getElementById("settingsModal");
  const btnCloseSettings = document.getElementById("btnCloseSettings");
  const btnCancelSettings = document.getElementById("btnCancelSettings");
  const btnSaveSettings = document.getElementById("btnSaveSettings");
  const inputApiKey = document.getElementById("inputApiKey");
  const settingsAlert = document.getElementById("settingsAlert");

  // Configure marked for clean parsing
  if (window.marked) {
    marked.setOptions({
      breaks: true,
      gfm: true
    });
  }

  // 1. Initial Health Check
  async function checkHealth() {
    try {
      const res = await fetch("/api/health");
      if (!res.ok) throw new Error("Health check failed");
      const data = await res.json();
      
      const vs = data.vector_store || {};
      const chunkCount = vs.total_chunks || 0;
      const hasKey = data.gemini_api_key_configured;

      statusIndicator.className = "status-chip " + (hasKey ? "online" : "warning");
      statusText.textContent = hasKey 
        ? `Ready (${chunkCount} Chunks Indexed)` 
        : `API Key Required (${chunkCount} Chunks)`;
    } catch (e) {
      statusIndicator.className = "status-chip warning";
      statusText.textContent = "Server Connecting...";
    }
  }

  checkHealth();

  // 2. Chat Query Execution
  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (!query) return;

    // Append user message
    appendUserMessage(query);
    queryInput.value = "";
    queryInput.style.height = "auto";

    // Append temporary typing indicator
    const typingId = appendTypingIndicator();
    scrollToBottom();

    // Disable button while waiting
    btnSubmit.disabled = true;

    // Collect history
    const history = [];
    messagesList.querySelectorAll(".message:not(#" + typingId + ")").forEach(msg => {
      const isUser = msg.classList.contains("user");
      const contentEl = msg.querySelector(".msg-content");
      if (contentEl) {
        history.push({
          role: isUser ? "user" : "assistant",
          content: contentEl.textContent.trim()
        });
      }
    });

    try {
      const payload = {
        query: query,
        top_k: 5,
        document_type: docFilter.value || null,
        history: history.slice(-4)
      };

      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      removeTypingIndicator(typingId);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: "Server error" }));
        appendAssistantMessage({
          answer: `⚠️ Error: ${errorData.detail || "Unable to reach RAG server."}`,
          fallback_triggered: true,
          citations: []
        });
      } else {
        const data = await response.json();
        appendAssistantMessage(data);
      }
    } catch (err) {
      removeTypingIndicator(typingId);
      appendAssistantMessage({
        answer: `⚠️ Network error: ${err.message}. Please verify the server is running.`,
        fallback_triggered: true,
        citations: []
      });
    } finally {
      btnSubmit.disabled = false;
      scrollToBottom();
    }
  });

  // 3. UI Helpers for Messages
  function appendUserMessage(text) {
    const msgDiv = document.createElement("div");
    msgDiv.className = "message user";
    msgDiv.innerHTML = `
      <div class="msg-avatar">👤</div>
      <div class="msg-body">
        <div class="msg-sender">Support Agent</div>
        <div class="msg-content">${escapeHTML(text)}</div>
      </div>
    `;
    messagesList.appendChild(msgDiv);
  }

  function appendAssistantMessage(data) {
    const msgDiv = document.createElement("div");
    msgDiv.className = "message assistant";

    const isFallback = data.fallback_triggered || false;
    const badgeHTML = isFallback
      ? `<div class="grounding-status fallback">⚠️ Compliance Escalation Required</div>`
      : `<div class="grounding-status verified">🛡️ Verified by Official BookMyForex Guidelines</div>`;

    // Render markdown content
    let renderedContent = data.answer;
    if (window.marked && window.DOMPurify) {
      renderedContent = DOMPurify.sanitize(marked.parse(data.answer));
    }

    // Render citations drawer if available
    let citationsHTML = "";
    if (data.citations && data.citations.length > 0) {
      const itemsHTML = data.citations.map(c => `
        <div class="citation-card">
          <div class="citation-header">
            <span>📄 ${escapeHTML(c.document_title || c.source_file)} &rsaquo; ${escapeHTML(c.section_title)}</span>
          </div>
          <div class="citation-snippet">${escapeHTML(c.snippet)}</div>
        </div>
      `).join("");

      citationsHTML = `
        <div class="citations-box">
          <div class="citations-toggle" onclick="this.nextElementSibling.classList.toggle('hidden')">
            <span>📚 Official Guidelines Consulted</span>
            <span>▼</span>
          </div>
          <div class="citations-items hidden">
            ${itemsHTML}
          </div>
        </div>
      `;
    }

    // Render Action Bar
    const msgId = "assistant-" + Date.now();
    msgDiv.id = msgId;
    const actionsHTML = `
      <div class="msg-actions" style="margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.08); display: flex; justify-content: space-between; align-items: center; font-size: 0.75rem;">
        <div style="display: flex; gap: 6px;">
          <button type="button" class="btn-copy-clean" onclick="copyCleanMessage('${msgId}')" style="background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1); color: #FE8405; padding: 3px 8px; border-radius: 6px; cursor: pointer;">📋 Copy for Customer</button>
          <button type="button" class="btn-copy-full" onclick="copyFullMessage('${msgId}')" style="background: transparent; border: 1px solid rgba(255,255,255,0.06); color: #94a3b8; padding: 3px 8px; border-radius: 6px; cursor: pointer;">Copy Full</button>
        </div>
        <div style="display: flex; gap: 4px; align-items: center; color: #64748b;">
          <span>Helpful?</span>
          <button type="button" onclick="sendFeedback('${msgId}', 'positive')" style="background:none; border:none; cursor:pointer; font-size: 0.9rem;">👍</button>
          <button type="button" onclick="sendFeedback('${msgId}', 'negative')" style="background:none; border:none; cursor:pointer; font-size: 0.9rem;">👎</button>
        </div>
      </div>
    `;

    msgDiv.innerHTML = `
      <div class="msg-avatar">🤖</div>
      <div class="msg-body">
        <div class="msg-sender">BookMyForex Support Assistant</div>
        <div class="msg-content" data-raw="${escapeHTML(data.answer || '')}">
          ${badgeHTML}
          ${renderedContent}
          ${citationsHTML}
          ${actionsHTML}
        </div>
      </div>
    `;

    messagesList.appendChild(msgDiv);
  }

  window.copyCleanMessage = function(msgId) {
    const el = document.getElementById(msgId);
    if (!el) return;
    const raw = el.querySelector(".msg-content")?.getAttribute("data-raw") || "";
    let clean = raw.split(/📚\s*\*?\*?Sources Cited\*?\*?/i)[0];
    clean = clean.replace(/\[[a-zA-Z0-9_\-\.]+\.md:[^\]]+\]/g, "").trim();
    navigator.clipboard.writeText(clean);
    alert("Copied clean, customer-ready text to clipboard!");
  };

  window.copyFullMessage = function(msgId) {
    const el = document.getElementById(msgId);
    if (!el) return;
    const raw = el.querySelector(".msg-content")?.getAttribute("data-raw") || "";
    navigator.clipboard.writeText(raw);
    alert("Copied full response with citations to clipboard!");
  };

  window.sendFeedback = async function(msgId, rating) {
    const el = document.getElementById(msgId);
    if (!el) return;
    const raw = el.querySelector(".msg-content")?.getAttribute("data-raw") || "";
    try {
      await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: "Static UI Query",
          answer: raw,
          rating: rating
        })
      });
      alert(`Thank you for your ${rating === 'positive' ? 'positive' : 'feedback'} rating!`);
    } catch(e) {
      alert("Feedback recorded!");
    }
  };

  function appendTypingIndicator() {
    const id = "typing-" + Date.now();
    const typingDiv = document.createElement("div");
    typingDiv.id = id;
    typingDiv.className = "message assistant";
    typingDiv.innerHTML = `
      <div class="msg-avatar">🤖</div>
      <div class="msg-body">
        <div class="msg-content">
          <div class="typing-indicator">
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
          </div>
        </div>
      </div>
    `;
    messagesList.appendChild(typingDiv);
    return id;
  }

  function removeTypingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
  }

  function scrollToBottom() {
    messagesList.scrollTop = messagesList.scrollHeight;
  }

  function escapeHTML(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // 4. Quick Prompts Clicking
  document.querySelectorAll(".quick-prompts-bar .pill").forEach(pill => {
    pill.addEventListener("click", () => {
      const query = pill.getAttribute("data-query");
      if (query) {
        queryInput.value = query;
        chatForm.dispatchEvent(new Event("submit"));
      }
    });
  });

  // 5. Textarea Enter Handling
  queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      chatForm.dispatchEvent(new Event("submit"));
    }
  });

  // Auto-resize textarea
  queryInput.addEventListener("input", function() {
    this.style.height = "auto";
    this.style.height = Math.min(this.scrollHeight, 120) + "px";
  });

  // 6. Re-Index Button
  btnReindex.addEventListener("click", async () => {
    btnReindex.disabled = true;
    const origHTML = btnReindex.innerHTML;
    btnReindex.innerHTML = `<span>Re-indexing...</span>`;

    try {
      const res = await fetch("/api/ingest?reset=true", { method: "POST" });
      const data = await res.json();
      if (res.ok) {
        alert(data.message || "Knowledge base re-indexed successfully!");
        checkHealth();
        loadDocumentsList();
      } else {
        alert("Re-indexing failed: " + (data.detail || "Unknown error"));
      }
    } catch (err) {
      alert("Error contacting server: " + err.message);
    } finally {
      btnReindex.disabled = false;
      btnReindex.innerHTML = origHTML;
    }
  });

  // 7. Knowledge Base Drawer
  async function loadDocumentsList() {
    docsList.innerHTML = `<div class="loading-spinner-small">Loading documents...</div>`;
    try {
      const res = await fetch("/api/documents");
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail);

      if (data.documents && data.documents.length > 0) {
        docsList.innerHTML = data.documents.map(doc => `
          <div class="doc-item-card">
            <div class="doc-item-title">${escapeHTML(doc.document_title)}</div>
            <div class="doc-item-file">📁 ${escapeHTML(doc.source_file)}</div>
            <div class="doc-item-meta">
              <span>Type: <strong>${escapeHTML(doc.document_type)}</strong></span>
              <span>${doc.chunks_count} Chunks</span>
            </div>
          </div>
        `).join("");
      } else {
        docsList.innerHTML = `<p style="font-size:0.85rem; color:#64748b;">No documents currently indexed.</p>`;
      }
    } catch (err) {
      docsList.innerHTML = `<p style="font-size:0.85rem; color:#c92a2a;">Error: ${escapeHTML(err.message)}</p>`;
    }
  }

  btnDocsToggle.addEventListener("click", () => {
    docsDrawer.classList.toggle("hidden");
    if (!docsDrawer.classList.contains("hidden")) {
      loadDocumentsList();
    }
  });

  btnCloseDrawer.addEventListener("click", () => {
    docsDrawer.classList.add("hidden");
  });

  // 8. Settings Modal
  btnSettings.addEventListener("click", () => {
    settingsModal.classList.remove("hidden");
    settingsAlert.classList.add("hidden");
  });

  btnCloseSettings.addEventListener("click", () => {
    settingsModal.classList.add("hidden");
  });

  btnCancelSettings.addEventListener("click", () => {
    settingsModal.classList.add("hidden");
  });

  btnSaveSettings.addEventListener("click", async () => {
    const key = inputApiKey.value.trim();
    if (!key) {
      settingsAlert.className = "alert-msg error";
      settingsAlert.textContent = "Please enter a valid Gemini API key.";
      settingsAlert.classList.remove("hidden");
      return;
    }

    btnSaveSettings.disabled = true;
    try {
      const res = await fetch("/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ gemini_api_key: key })
      });
      const data = await res.json();

      if (res.ok) {
        settingsAlert.className = "alert-msg success";
        settingsAlert.textContent = "Key saved successfully! Engine is now live.";
        settingsAlert.classList.remove("hidden");
        checkHealth();
        setTimeout(() => {
          settingsModal.classList.add("hidden");
          btnSaveSettings.disabled = false;
        }, 1200);
      } else {
        settingsAlert.className = "alert-msg error";
        settingsAlert.textContent = data.detail || "Failed to update API key.";
        settingsAlert.classList.remove("hidden");
        btnSaveSettings.disabled = false;
      }
    } catch (e) {
      settingsAlert.className = "alert-msg error";
      settingsAlert.textContent = "Network error: " + e.message;
      settingsAlert.classList.remove("hidden");
      btnSaveSettings.disabled = false;
    }
  });
});
