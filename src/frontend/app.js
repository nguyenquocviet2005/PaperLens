const API = (window.PAPERLENS_CONFIG?.apiUrl || "http://127.0.0.1:8000").replace(/\/$/, "");
const AUTH_URL = String(window.PAPERLENS_CONFIG?.supabaseAuthUrl || "").replace(/\/$/, "");
const AUTH_ANON_KEY = String(window.PAPERLENS_CONFIG?.supabaseAnonKey || "");
const AUTH_CONFIGURED = Boolean(AUTH_URL && AUTH_ANON_KEY && !AUTH_URL.includes("${"));
const AGENT_CONVERSATION_KEY = "paperlens.agentConversationId";
const AUTH_SESSION_KEY = "paperlens.authSession.v1";
const INGESTION_JOBS_KEY = "paperlens.ingestionJobs.v1";
function readStoredJson(key, fallback) {
  try { return JSON.parse(localStorage.getItem(key) || JSON.stringify(fallback)); }
  catch { return fallback; }
}
const state = {
  papers: [],
  activePaperId: localStorage.getItem("paperlens.activePaper") || "",
  document: null,
  assets: null,
  chunks: null,
  readerTab: "text",
  deleteId: null,
  conversationIds: {},
  agentConversationId: localStorage.getItem(AGENT_CONVERSATION_KEY) || null,
  agentArtifacts: [],
  agentTrace: [],
  agentLiveTool: null,
  agentActive: false,
  agentImage: null,
  agentSessions: [],
  authSession: readStoredJson(AUTH_SESSION_KEY, null),
  guestQuota: null,
  guestTrial: null,
  ingestionJobs: readStoredJson(INGESTION_JOBS_KEY, {})
};
const $ = (q, root = document) => root.querySelector(q);
const $$ = (q, root = document) => [...root.querySelectorAll(q)];
const activeJobPolls = new Set();

function escapeHtml(value = "") {
  return String(value).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[c]));
}
function titleOf(p) { return p.title || p.filename || "Untitled paper"; }
function activePaper() { return state.papers.find(p => p.paper_id === state.activePaperId); }
function toast(message, type = "") {
  const el = document.createElement("div"); el.className = `toast ${type}`; el.textContent = message;
  $("#toast-region").append(el); setTimeout(() => el.remove(), 4200);
}
async function api(path, options = {}) {
  await ensureAuthSession();
  const token = state.authSession?.access_token;
  const response = await fetch(`${API}${path}`, { ...options, headers: { ...(options.body instanceof FormData ? {} : { "Content-Type": "application/json" }), ...(token ? { "Authorization": `Bearer ${token}` } : {}), ...(options.headers || {}) } });
  const text = await response.text(); let data;
  try { data = text ? JSON.parse(text) : {}; } catch { data = { detail: text || response.statusText }; }
  if (!response.ok) {
    const detail = data.detail;
    const message = (detail && typeof detail === "object")
      ? (detail.message || detail.error || JSON.stringify(detail))
      : (detail || `Request failed (${response.status})`);
    const error = new Error(message);
    error.status = response.status;
    error.detail = detail;
    if (detail && typeof detail === "object" && detail.quota) {
      applyGuestQuota(detail.quota);
    }
    if (response.status === 401 && isGuestSession()) {
      storeAuthSession(null);
      renderAccount();
      $("#auth-modal").classList.remove("hidden");
    }
    throw error;
  }
  return data;
}
function isGuestSession() {
  return Boolean(state.authSession?.user?.is_guest || String(state.authSession?.access_token || "").startsWith("guest."));
}
function applyGuestQuota(quota) {
  if (!quota) return;
  state.guestQuota = quota;
  if (state.authSession) {
    state.authSession = { ...state.authSession, guest_quota: quota };
    localStorage.setItem(AUTH_SESSION_KEY, JSON.stringify(state.authSession));
  }
  renderGuestQuota();
}
function renderGuestQuota() {
  const chip = $("#guest-quota");
  const hint = $("#guest-quota-hint");
  const quota = state.guestQuota || state.authSession?.guest_quota || null;
  const show = isGuestSession() && quota;
  if (chip) {
    chip.classList.toggle("hidden", !show);
    if (show) {
      chip.textContent = `Guest · ${quota.queries_remaining}/${quota.queries_limit} queries · ${quota.papers_remaining}/${quota.papers_limit} papers · ${quota.images_remaining}/${quota.images_limit} images`;
    }
  }
  if (hint) {
    hint.classList.toggle("hidden", !show);
    if (show) {
      hint.textContent = `Trial remaining: ${quota.queries_remaining} chat queries, ${quota.papers_remaining} paper uploads, ${quota.images_remaining} image uploads. Create an account for full access.`;
    }
  }
}
function storeAuthSession(session) {
  state.authSession = session || null;
  state.guestQuota = session?.guest_quota || null;
  if (session) localStorage.setItem(AUTH_SESSION_KEY, JSON.stringify(session));
  else localStorage.removeItem(AUTH_SESSION_KEY);
  renderGuestQuota();
}
async function authRequest(path, body) {
  const response = await fetch(`${AUTH_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "apikey": AUTH_ANON_KEY },
    body: JSON.stringify(body)
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.msg || data.error_description || data.message || `Authentication failed (${response.status})`);
  return data;
}
async function ensureAuthSession() {
  if (!state.authSession) return;
  if (isGuestSession()) {
    const expiresAt = Number(state.authSession.expires_at || 0);
    if (expiresAt && expiresAt * 1000 <= Date.now()) {
      storeAuthSession(null);
      $("#auth-modal").classList.remove("hidden");
      throw new Error("Your guest trial expired. Sign in or start a new guest session.");
    }
    return;
  }
  if (!AUTH_CONFIGURED) return;
  const expiresAt = Number(state.authSession.expires_at || 0);
  if (!expiresAt || expiresAt * 1000 > Date.now() + 60000) return;
  try {
    const refreshed = await authRequest("/auth/v1/token?grant_type=refresh_token", { refresh_token: state.authSession.refresh_token });
    storeAuthSession(refreshed);
    renderAccount();
  } catch {
    storeAuthSession(null);
    $("#auth-modal").classList.remove("hidden");
    throw new Error("Your session expired. Please sign in again.");
  }
}
function renderAccount() {
  const button = $("#account-button");
  if (!AUTH_CONFIGURED && !isGuestSession()) { button.classList.add("hidden"); renderGuestQuota(); return; }
  const guest = isGuestSession();
  const email = state.authSession?.user?.email || "";
  button.classList.toggle("hidden", !state.authSession);
  $("#account-label").textContent = guest ? "Guest trial" : (email || "Account");
  $("#account-avatar").textContent = guest ? "G" : (email[0] || "A").toUpperCase();
  renderGuestQuota();
}
async function submitAuth(mode = "signin") {
  const email = $("#auth-email").value.trim(), password = $("#auth-password").value;
  $("#auth-error").textContent = "";
  try {
    const path = mode === "signup" ? "/auth/v1/signup" : "/auth/v1/token?grant_type=password";
    const result = await authRequest(path, { email, password });
    if (!result.access_token) {
      $("#auth-error").textContent = "Check your email to confirm the account, then sign in.";
      return;
    }
    storeAuthSession(result);
    $("#auth-modal").classList.add("hidden");
    renderAccount();
    await Promise.all([loadLibrary(), loadAgentSessions()]);
    await restoreAgentConversation();
    resumeIngestionJobs();
  } catch (error) { $("#auth-error").textContent = error.message; }
}
async function continueAsGuest() {
  $("#auth-error").textContent = "";
  const button = $("#auth-guest");
  if (button) button.disabled = true;
  try {
    const response = await fetch(`${API}/auth/guest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}"
    });
    const result = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = result.detail;
      throw new Error(
        (detail && typeof detail === "object")
          ? (detail.message || detail.error || JSON.stringify(detail))
          : (detail || `Guest trial unavailable (${response.status})`)
      );
    }
    storeAuthSession({
      access_token: result.access_token,
      token_type: result.token_type || "bearer",
      expires_at: result.expires_at,
      user: result.user,
      guest_quota: result.guest_quota
    });
    applyGuestQuota(result.guest_quota);
    $("#auth-modal").classList.add("hidden");
    renderAccount();
    await Promise.all([loadLibrary(), loadAgentSessions()]);
    await restoreAgentConversation();
    resumeIngestionJobs();
    toast("Guest trial started. Limits apply until you create an account.", "ok");
  } catch (error) {
    $("#auth-error").textContent = error.message;
  } finally {
    if (button) button.disabled = false;
  }
}
async function loadCapabilities() {
  try {
    const caps = await fetch(`${API}/capabilities`).then(r => r.json());
    state.guestTrial = caps.guest_trial || null;
    const note = $("#auth-guest-note");
    const guestButton = $("#auth-guest");
    if (state.guestTrial?.enabled) {
      if (note) {
        note.textContent = `Guest trial: ${state.guestTrial.max_queries} chat queries, ${state.guestTrial.max_papers} paper uploads, and ${state.guestTrial.max_images} image uploads.`;
      }
      if (guestButton) guestButton.classList.remove("hidden");
    } else if (guestButton) {
      guestButton.classList.add("hidden");
      if (note) note.classList.add("hidden");
    }
  } catch {
    /* Capabilities are optional for local bootstrapping. */
  }
}
// Supabase email links (signup confirmation, magic link, recovery) redirect back with the
// session or an error in the URL fragment. Consume it before the hash router sees it.
async function consumeAuthRedirect() {
  const params = new URLSearchParams(location.hash.slice(1));
  if (!params.has("access_token") && !params.has("error_description")) return;
  history.replaceState(null, "", location.pathname + location.search);
  if (params.has("error_description")) {
    toast(params.get("error_description").replace(/\+/g, " "), "error");
    return;
  }
  if (!AUTH_CONFIGURED) return;
  const expiresIn = Number(params.get("expires_in") || 3600);
  const session = {
    access_token: params.get("access_token"),
    refresh_token: params.get("refresh_token"),
    token_type: params.get("token_type") || "bearer",
    expires_in: expiresIn,
    expires_at: Number(params.get("expires_at")) || Math.floor(Date.now() / 1000) + expiresIn,
  };
  try {
    const response = await fetch(`${AUTH_URL}/auth/v1/user`, {
      headers: { "apikey": AUTH_ANON_KEY, "Authorization": `Bearer ${session.access_token}` }
    });
    if (!response.ok) throw new Error(`Could not load your account (${response.status})`);
    session.user = await response.json();
    storeAuthSession(session);
    toast(params.get("type") === "signup" ? "Email confirmed. You're signed in." : "You're signed in.", "ok");
  } catch (error) {
    toast(error.message, "error");
  }
}
function initializeAuth() {
  renderAccount();
  if (AUTH_CONFIGURED && !state.authSession) {
    $("#auth-modal").classList.remove("hidden");
    return false;
  }
  return true;
}
function protectMath(value = "") {
  const math = [], code = [];
  const addMath = (expression, display) => {
    const source = String(expression).trim().replace(/^\$\$|\$\$$/g, "").trim();
    const encoded = escapeHtml(encodeURIComponent(source));
    return display
      ? `\n<div class="math-slot math-display" data-math="${encoded}" data-display="true"></div>\n`
      : `<span class="math-slot" data-math="${encoded}" data-display="false"></span>`;
  };
  let text = String(value).replace(/\r\n?/g, "\n");
  text = text.replace(/```(?:latex|tex|math)\s*\n([\s\S]*?)```/gi, (_, expression) => addMath(expression, true));
  text = text.replace(/```[\s\S]*?```/g, block => {
    const token = `PAPERLENSCODEBLOCK${code.length}TOKEN`;
    code.push(block);
    return token;
  });
  text = text.replace(/`[^`\n]+`/g, block => {
    const token = `PAPERLENSCODEBLOCK${code.length}TOKEN`;
    code.push(block);
    return token;
  });
  text = text.replace(/\$\$([\s\S]+?)\$\$/g, (_, expression) => addMath(expression, true));
  text = text.replace(/\\\[([\s\S]+?)\\\]/g, (_, expression) => addMath(expression, true));
  text = text.replace(/\\begin\{(equation\*?|align\*?|gather\*?)\}([\s\S]+?)\\end\{\1\}/g, (match) => addMath(match, true));
  text = text.replace(/\\\((.+?)\\\)/g, (_, expression) => addMath(expression, false));
  text = text.replace(/(^|[^\\$])\$([^$\n]+?)\$/g, (_, prefix, expression) => `${prefix}${addMath(expression, false)}`);
  code.forEach((block, index) => { text = text.replace(`PAPERLENSCODEBLOCK${index}TOKEN`, block); });
  return text;
}
function markdownHtml(value = "") {
  const protectedValue = protectMath(value);
  if (!window.marked || !window.DOMPurify) return escapeHtml(value).replace(/\n/g, "<br>");
  return DOMPurify.sanitize(marked.parse(protectedValue, { gfm: true, breaks: true }));
}
function renderMath(root) {
  if (!root) return;
  const slots = $$(".math-slot[data-math]", root);
  slots.forEach(slot => {
    let expression = "";
    try { expression = decodeURIComponent(slot.dataset.math || ""); } catch { expression = slot.dataset.math || ""; }
    if (!window.katex) { slot.textContent = expression; return; }
    try {
      katex.render(expression, slot, { displayMode: slot.dataset.display === "true", throwOnError: true, strict: "warn" });
    } catch {
      slot.classList.add("math-fallback");
      slot.textContent = expression;
      slot.title = "This expression could not be rendered by KaTeX";
    }
  });
  if (slots.length) return;
  if (!window.renderMathInElement) return;
  renderMathInElement(root, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "\\[", right: "\\]", display: true },
      { left: "$", right: "$", display: false },
      { left: "\\(", right: "\\)", display: false }
    ],
    throwOnError: false
  });
}
function renderMarkdown(root, value) {
  root.innerHTML = markdownHtml(value);
  renderMath(root);
}
function scrollMessages(target = $("#agent-messages"), behavior = "smooth") {
  if (!target) return;
  requestAnimationFrame(() => target.scrollTo({ top: target.scrollHeight, behavior }));
}
function setAgentActivity(active, detail = "Ready to research") {
  state.agentActive = active;
  $(".agent-chat")?.classList.toggle("streaming", active);
  $("#agent-send").disabled = active;
  $("#agent-image-input").disabled = active;
  $("#agent-pdf-file").disabled = active;
  $("#agent-tools-toggle").disabled = active;
  $$("#agent-tools-menu button").forEach(button => { button.disabled = active; });
  $("#agent-state span").textContent = active ? "Working" : "Ready";
  $("#agent-live-status").textContent = detail;
}
async function streamApi(path, payload, onEvent) {
  await ensureAuthSession();
  const token = state.authSession?.access_token;
  const response = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "Accept": "text/event-stream", ...(token ? { "Authorization": `Bearer ${token}` } : {}) },
    body: JSON.stringify(payload)
  });
  if (!response.ok) {
    const text = await response.text();
    let detail;
    try { detail = JSON.parse(text).detail; } catch { detail = text; }
    if (detail && typeof detail === "object") detail = detail.message || detail.error || JSON.stringify(detail);
    throw new Error(detail || `Request failed (${response.status})`);
  }
  const reader = response.body.getReader(), decoder = new TextDecoder();
  let buffer = "";
  const emitBlock = block => {
    const data = block.split(/\r?\n/).filter(line => line.startsWith("data:")).map(line => line.slice(5).trim()).join("");
    if (data) onEvent(JSON.parse(data));
  };
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const blocks = buffer.split(/\r?\n\r?\n/); buffer = blocks.pop() || "";
    blocks.forEach(emitBlock);
    if (done) {
      if (buffer.trim()) emitBlock(buffer);
      break;
    }
  }
}
function paperOptions(selected = "") {
  return `<option value="">Choose from your library</option>` + state.papers.map(p => `<option value="${escapeHtml(p.paper_id)}" ${p.paper_id === selected ? "selected" : ""}>${escapeHtml(titleOf(p))} · ${p.page_count || 0} pages</option>`).join("");
}
function statusBadge(status = "unknown") { return `<span class="status ${escapeHtml(status)}">${escapeHtml(status)}</span>`; }
function paperCard(p) {
  return `<article class="paper-card">${statusBadge(p.status)}
    <h3>${escapeHtml(titleOf(p))}</h3><p>${escapeHtml((p.authors || []).map(a => typeof a === "string" ? a : a.name).filter(Boolean).join(", ") || p.filename)}</p>
    <footer><button class="button secondary small" data-open-paper="${escapeHtml(p.paper_id)}">Open workspace</button><button class="button ghost small" data-delete-paper="${escapeHtml(p.paper_id)}">Delete</button></footer></article>`;
}
function navigate(route) {
  $$(".page").forEach(el => el.classList.toggle("active", el.id === `page-${route}`));
  $$(".nav-item").forEach(el => el.classList.toggle("active", el.dataset.route === route));
  $(".sidebar").classList.remove("open"); history.replaceState(null, "", `#${route}`);
  if (route === "reader") loadReader();
  if (route === "chat") updateChatPaper();
  if (route === "library") renderLibrary();
}
async function loadLibrary(showToast = false) {
  const result = await api("/papers?limit=200");
  state.papers = result.papers || [];
  if (state.activePaperId && !state.papers.some(p => p.paper_id === state.activePaperId)) state.activePaperId = "";
  $("#global-paper-select").innerHTML = paperOptions(state.activePaperId);
  $("#stat-papers").textContent = result.count ?? state.papers.length;
  $("#stat-completed").textContent = state.papers.filter(p => p.status === "completed").length;
  $("#recent-papers").innerHTML = state.papers.slice(0, 6).map(paperCard).join("") || `<div class="empty-inline">No papers yet — upload your first PDF.</div>`;
  renderLibrary(); renderPaperChecks(); updateChatPaper();
  if (showToast) toast("Library refreshed");
}
function renderLibrary() {
  const grid = $("#library-grid"); if (!grid) return;
  const term = ($("#library-search")?.value || "").toLowerCase();
  const status = $("#library-status")?.value || "";
  const filtered = state.papers.filter(p => (!status || p.status === status) && (!term || JSON.stringify([p.title, p.filename, p.authors, p.publication_year]).toLowerCase().includes(term)));
  grid.innerHTML = filtered.map(paperCard).join("") || `<div class="empty-state compact"><span>▤</span><h2>No matching papers</h2><p>Try a different search or upload a PDF.</p></div>`;
}
function renderPaperChecks() {
  const html = state.papers.filter(p => p.status === "completed").map(p => `<label class="check-item"><input type="checkbox" value="${escapeHtml(p.paper_id)}"><span>${escapeHtml(titleOf(p))}<small>${p.page_count || 0} pages · ${escapeHtml(p.filename)}</small></span></label>`).join("") || "<p>No completed papers available.</p>";
  $("#compare-papers").innerHTML = html; $("#research-papers").innerHTML = html;
}
function selectPaper(id, route) {
  state.activePaperId = id; localStorage.setItem("paperlens.activePaper", id); $("#global-paper-select").value = id;
  state.document = state.assets = state.chunks = null; updateChatPaper(); if (route) navigate(route);
}

async function loadReader() {
  if (!state.activePaperId) { $("#reader-empty").classList.remove("hidden"); $("#reader-content").classList.add("hidden"); return; }
  $("#reader-empty").classList.add("hidden"); $("#reader-content").classList.remove("hidden");
  $("#reader-panel").innerHTML = `<div class="operation"><div class="spinner"></div><div><strong>Loading structured document…</strong></div></div>`;
  try {
    const [doc, assets, chunks] = await Promise.all([
      api(`/papers/${state.activePaperId}/document`),
      api(`/papers/${state.activePaperId}/assets`),
      api(`/papers/${state.activePaperId}/chunks`).catch(() => ({ chunks: [], count: 0 }))
    ]);
    state.document = doc; state.assets = assets; state.chunks = chunks;
    $("#reader-subtitle").textContent = `${doc.title || doc.filename} · ${doc.page_count} pages`;
    $("#reader-summary").innerHTML = [
      ["Pages", doc.page_count], ["Text elements", (doc.text_elements || []).length],
      ["Visual assets", (assets.figures || []).length + (assets.tables || []).length + (assets.formulas || []).length],
      ["Retrieval chunks", chunks.count || 0]
    ].map(([k,v]) => `<div class="summary-cell"><small>${k}</small><strong>${v}</strong></div>`).join("");
    renderReaderPanel();
  } catch (e) { $("#reader-panel").innerHTML = `<div class="empty-state compact"><h2>Reader unavailable</h2><p>${escapeHtml(e.message)}</p></div>`; }
}
function assetCard(item, type) {
  const url = `${API}/papers/${encodeURIComponent(state.activePaperId)}/assets/${type}/${encodeURIComponent(item.element_id)}/content`;
  return `<article class="asset-card card">${item.image_uri ? `<img loading="lazy" data-asset-src="${escapeHtml(url)}" alt="${escapeHtml(type)} ${escapeHtml(item.element_id)}">` : `<div class="asset-placeholder">No image preview</div>`}
    <h3>${escapeHtml(item.caption || item.element_id)}</h3><p>Page ${item.page || "—"}${item.docling_text ? ` · ${escapeHtml(item.docling_text)}` : ""}</p>${item.needs_enrichment ? `<span class="status accepted">Needs transcription</span>` : ""}</article>`;
}
async function loadProtectedImages(root) {
  await ensureAuthSession();
  const token = state.authSession?.access_token;
  $$(".asset-card img[data-asset-src]", root).forEach(async image => {
    try {
      const response = await fetch(image.dataset.assetSrc, { headers: token ? { "Authorization": `Bearer ${token}` } : {} });
      if (!response.ok) throw new Error("preview unavailable");
      image.src = URL.createObjectURL(await response.blob());
      image.onload = () => URL.revokeObjectURL(image.src);
    } catch {
      image.replaceWith(Object.assign(document.createElement("div"), { className:"asset-placeholder", textContent:"Preview unavailable" }));
    }
  });
}
function renderReaderPanel() {
  if (!state.document) return;
  const panel = $("#reader-panel"), tab = state.readerTab;
  if (tab === "text") {
    panel.innerHTML = `<div class="document-flow">${(state.document.text_elements || []).map(el => `<div class="doc-element"><small>PAGE ${el.page || "—"} · ${escapeHtml((el.section_path || []).join(" › "))}</small><div>${escapeHtml(el.text)}</div></div>`).join("")}</div>`;
  } else if (tab === "chunks") {
    panel.innerHTML = `<div class="result-list">${(state.chunks?.chunks || []).map(c => `<article class="result-item"><span class="status">${escapeHtml(c.chunk_type)}</span><h3>${escapeHtml((c.section_path || []).join(" › ") || "Document")}</h3><p>Pages ${c.page_start || "—"}–${c.page_end || "—"} · ${c.token_count} tokens</p><p>${escapeHtml(c.content_preview)}</p></article>`).join("") || "<p>No chunks available. Rebuild the index.</p>"}</div>`;
  } else {
    const type = tab === "figures" ? "figure" : tab === "tables" ? "table" : "formula";
    const items = state.assets?.[tab] || [];
    panel.innerHTML = `<div class="asset-grid">${items.map(i => assetCard(i, type)).join("") || `<div class="empty-inline">No ${tab} were detected.</div>`}</div>`;
    loadProtectedImages(panel);
  }
}
function updateChatPaper() {
  const p = activePaper(), empty = $("#chat-empty"), shell = $("#chat-shell");
  if (!p) { empty.classList.remove("hidden"); shell.classList.add("hidden"); return; }
  empty.classList.add("hidden"); shell.classList.remove("hidden"); $("#chat-paper").textContent = titleOf(p);
}
function appendMessage(text, kind, extra = "") {
  const div = document.createElement("div"); div.className = `message ${kind}`; div.innerHTML = `${escapeHtml(text)}${extra}`; $("#messages").append(div); $("#messages").scrollTop = $("#messages").scrollHeight; return div;
}
function citeLabel(c) { return c.label || `[Page ${c.page || "—"}]`; }

function selectedAgentPapers() {
  return state.activePaperId ? [state.activePaperId] : [];
}
function setToolsMenu(open) {
  const menu = $("#agent-tools-menu"), toggle = $("#agent-tools-toggle");
  menu.classList.toggle("hidden", !open);
  toggle.setAttribute("aria-expanded", String(open));
}
function plotSeries(artifact) {
  if (Array.isArray(artifact.series)) {
    return artifact.series.map((series, index) => ({
      name: series.name || series.label || `Series ${index + 1}`,
      points: (series.points || series.data || []).map((point, pointIndex) => Array.isArray(point)
        ? { x: point[0], y: point[1] }
        : { x: point.x ?? pointIndex, y: point.y ?? point.value })
    }));
  }
  if (Array.isArray(artifact.data?.datasets)) {
    const labels = artifact.data.labels || [];
    return artifact.data.datasets.map((dataset, index) => ({
      name: dataset.label || `Series ${index + 1}`,
      points: (dataset.data || []).map((value, pointIndex) => ({
        x: labels[pointIndex] ?? pointIndex,
        y: typeof value === "object" ? value.y : value
      }))
    }));
  }
  const values = artifact.points || artifact.data || artifact.values || [];
  return [{
    name: artifact.label || "Values",
    points: Array.isArray(values) ? values.map((point, index) => Array.isArray(point)
      ? { x: point[0], y: point[1] }
      : typeof point === "object"
        ? { x: point.x ?? point.label ?? index, y: point.y ?? point.value }
        : { x: index, y: point }) : []
  }];
}
function plotArtifactHtml(artifact) {
  const series = plotSeries(artifact);
  const allPoints = series.flatMap(item => item.points).filter(point => Number.isFinite(Number(point.y)));
  if (!allPoints.length) return `<div class="agent-result"><span class="eyebrow">PLOT</span><h3>${escapeHtml(artifact.title || "Visualization")}</h3><p>No numeric plot data was returned.</p></div>`;
  const width = 720, height = 360, left = 62, right = 24, top = 28, bottom = 58;
  const xValues = allPoints.map((point, index) => Number.isFinite(Number(point.x)) ? Number(point.x) : index);
  const yValues = allPoints.map(point => Number(point.y));
  let xMin = Math.min(...xValues), xMax = Math.max(...xValues), yMin = Math.min(0, ...yValues), yMax = Math.max(...yValues);
  if (xMin === xMax) { xMin -= 1; xMax += 1; }
  if (yMin === yMax) { yMin -= 1; yMax += 1; }
  const px = value => left + ((value - xMin) / (xMax - xMin)) * (width - left - right);
  const py = value => top + (1 - (value - yMin) / (yMax - yMin)) * (height - top - bottom);
  const colors = ["#2f6b4f", "#da7245", "#4676a9", "#8b5ba7"];
  const ticks = Array.from({ length: 5 }, (_, index) => {
    const ratio = index / 4, value = yMax - ratio * (yMax - yMin), y = py(value);
    return `<line class="plot-grid" x1="${left}" y1="${y}" x2="${width-right}" y2="${y}"/><text x="${left-9}" y="${y+4}" text-anchor="end">${escapeHtml(Number(value.toPrecision(3)))}</text>`;
  }).join("");
  const xTicks = Array.from({ length: 5 }, (_, index) => {
    const value = xMin + (index / 4) * (xMax - xMin), x = px(value);
    return `<line class="plot-grid" x1="${x}" y1="${top}" x2="${x}" y2="${height-bottom}"/><text x="${x}" y="${height-bottom+19}" text-anchor="middle">${escapeHtml(Number(value.toPrecision(3)))}</text>`;
  }).join("");
  let pointIndex = 0;
  const lines = series.map((item, index) => {
    const points = item.points.filter(point => Number.isFinite(Number(point.y))).map(point => {
      const x = Number.isFinite(Number(point.x)) ? Number(point.x) : pointIndex++;
      return `${px(x)},${py(Number(point.y))}`;
    }).join(" ");
    return points ? `<polyline points="${points}" fill="none" stroke="${colors[index % colors.length]}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>` : "";
  }).join("");
  const legend = series.length > 1 ? series.map((item, index) => `<span style="color:${colors[index % colors.length]}">● ${escapeHtml(item.name)}</span>`).join(" · ") : "";
  return `<div class="agent-result"><span class="eyebrow">PLOT</span><div class="plot-card"><h3 class="plot-title">${escapeHtml(artifact.title || "Visualization")}</h3>
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(artifact.title || "Data plot")}">${ticks}${xTicks}
      <line class="plot-axis" x1="${left}" y1="${top}" x2="${left}" y2="${height-bottom}"/><line class="plot-axis" x1="${left}" y1="${height-bottom}" x2="${width-right}" y2="${height-bottom}"/>
      ${lines}<text x="${(left+width-right)/2}" y="${height-13}" text-anchor="middle">${escapeHtml(artifact.x_label || artifact.xLabel || "X")}</text>
      <text x="16" y="${(top+height-bottom)/2}" text-anchor="middle" transform="rotate(-90 16 ${(top+height-bottom)/2})">${escapeHtml(artifact.y_label || artifact.yLabel || "Y")}</text>
    </svg>${legend ? `<div class="artifact-summary">${legend}</div>` : ""}</div></div>`;
}
function renderAgentArtifacts() {
  const target = $("#agent-results");
  if (!state.agentArtifacts.length) { target.innerHTML = `<div class="empty-inline">No tool results yet.</div>`; return; }
  target.innerHTML = state.agentArtifacts.slice().reverse().map(artifact => {
    if (artifact.kind === "discovery") return `<div class="agent-result"><span class="eyebrow">${escapeHtml(artifact.source || "DISCOVERY")}</span><h3>Related papers</h3>${(artifact.results || []).map(item => `<div class="finding"><strong>${escapeHtml(item.title)}</strong><p>${escapeHtml((item.authors || []).join(", "))}${item.year ? ` · ${item.year}` : ""}</p>${item.source_url ? `<a href="${escapeHtml(item.source_url)}" target="_blank" rel="noopener">Open source ↗</a>` : ""}</div>`).join("")}</div>`;
    if (artifact.kind === "paper_answer") return `<div class="agent-result"><span class="eyebrow">PAPER EVIDENCE</span><h3>Citations and evidence</h3><p class="artifact-summary">${(artifact.citations || []).length} cited source${(artifact.citations || []).length === 1 ? "" : "s"} supporting the answer in chat.</p><div>${(artifact.citations || []).map(c => `<span class="citation">${escapeHtml(citeLabel(c))}</span>`).join(" ")}</div>${(artifact.evidence || []).slice(0,4).map(e => `<div class="evidence">${escapeHtml((e.text || "").slice(0,280))}</div>`).join("")}</div>`;
    if (artifact.kind === "paper_reader") return `<div class="agent-result"><span class="eyebrow">READER</span><h3>${escapeHtml(artifact.title)}</h3><p>${artifact.page_count} pages · ${(artifact.counts?.figures || 0)} figures · ${(artifact.counts?.tables || 0)} tables</p>${(artifact.sections || []).slice(0,12).map(section => `<div class="finding"><strong>${escapeHtml(section.heading || (section.section_path || []).join(" › "))}</strong><small>Page ${section.page_start || "—"}</small></div>`).join("")}</div>`;
    if (artifact.kind === "research_report") return `<div class="agent-result"><span class="eyebrow">LANGGRAPH REPORT</span><h3>Research evidence summary</h3><p class="artifact-summary">The complete synthesis is shown in chat. Review the supporting evidence and workflow trace here.</p><div>${(artifact.citations || []).map(c => `<span class="citation">${escapeHtml(citeLabel(c))}</span>`).join(" ")}</div>${(artifact.critic_feedback || []).slice(0,4).map(item => `<div class="evidence">${escapeHtml(typeof item === "string" ? item : JSON.stringify(item))}</div>`).join("")}</div>`;
    if (artifact.kind === "comparison") return `<div class="agent-result"><span class="eyebrow">COMPARISON</span><h3>${escapeHtml(artifact.question)}</h3>${(artifact.paper_findings || []).map(f => `<div class="finding"><strong>${escapeHtml(f.title || "Paper")}</strong><p>${escapeHtml(f.summary)}</p></div>`).join("")}</div>`;
    if (artifact.kind === "code_assist") return `<div class="agent-result"><span class="eyebrow">CODE · ${escapeHtml(artifact.language || "text")}</span><h3>${escapeHtml(artifact.task || "Coding assistance")}</h3><div class="markdown-body">${markdownHtml(artifact.answer || "")}</div><p class="artifact-summary">Text-only analysis · code was not executed.</p></div>`;
    if (artifact.kind === "plot") return plotArtifactHtml(artifact);
    if (artifact.kind === "math" || artifact.kind === "math_analysis" || artifact.kind === "formula") {
      const expression = artifact.latex || artifact.expression || artifact.formula || artifact.content || "";
      return `<div class="agent-result"><span class="eyebrow">MATH</span><h3>${escapeHtml(artifact.title || "Formula")}</h3><div class="markdown-body">${markdownHtml(expression.match(/^\s*(\$\$|\\\[)/) ? expression : `$$${expression}$$`)}</div>${artifact.explanation ? `<p>${escapeHtml(artifact.explanation)}</p>` : ""}</div>`;
    }
    return `<div class="agent-result"><pre>${escapeHtml(JSON.stringify(artifact,null,2))}</pre></div>`;
  }).join("");
  renderMath(target);
}
function renderAgentTrace() {
  const target = $("#agent-trace");
  const calls = state.agentLiveTool ? [...state.agentTrace, state.agentLiveTool] : state.agentTrace;
  target.innerHTML = calls.length ? calls.map((call,index) => `<div class="tool-step"><span>${index+1}</span><div><strong>${escapeHtml(call.name || "tool")}</strong><small>${escapeHtml(call.status || "")}${call.arguments ? ` · ${escapeHtml(JSON.stringify(call.arguments))}` : ""}</small></div></div>`).join("") : `<div class="empty-inline">No tools called — this was a conversational response.</div>`;
}
async function askAgent(question) {
  if (state.agentActive) return;
  const attachment = state.agentImage;
  appendAgentMessage(question, "user", attachment);
  clearAgentImage();
  const pending = appendAgentMessage("Planning the next step…", "assistant");
  pending.innerHTML = `<div class="stream-status">Planning the next step…</div><div class="markdown-body"><span class="stream-cursor" aria-label="Response streaming"></span></div>`;
  const status = $(".stream-status", pending), content = $(".markdown-body", pending);
  let streamed = "";
  setAgentActivity(true, "Planning the next step…");
  try {
    const payload = { message:question, selected_papers:selectedAgentPapers(), conversation_id:state.agentConversationId };
    if (attachment?.dataUrl) payload.image = attachment.dataUrl;
    await streamApi("/agent/stream", payload, event => {
      if (event.type === "start") {
        state.agentConversationId = event.conversation_id;
        if (event.conversation_id) localStorage.setItem(AGENT_CONVERSATION_KEY, event.conversation_id);
      }
      if (event.type === "tool") {
        const toolName = String(event.name || "research tool").replaceAll("_"," ");
        const detail = `${event.status === "running" ? "Using" : "Finished"} ${toolName}…`;
        status.textContent = detail;
        setAgentActivity(true, detail);
        state.agentLiveTool = { name: event.name || "research tool", status: event.status || "running", arguments: event.arguments };
        renderAgentTrace();
      }
      if (event.type === "token") {
        streamed += event.content || "";
        status.textContent = "Writing response…";
        setAgentActivity(true, "Writing response…");
        renderMarkdown(content, streamed);
        content.insertAdjacentHTML("beforeend", `<span class="stream-cursor" aria-label="Response streaming"></span>`);
        scrollMessages();
      }
      if (event.type === "error") throw new Error(event.message);
      if (event.type === "done") {
        state.agentConversationId = event.conversation_id;
        if (event.conversation_id) localStorage.setItem(AGENT_CONVERSATION_KEY, event.conversation_id);
        state.agentArtifacts.push(...(event.artifacts || []));
        state.agentTrace.push(...(event.tool_calls || []));
        if (event.guest_quota) applyGuestQuota(event.guest_quota);
        // The completed server answer is authoritative. Streaming can include
        // partial text, but must never override the validated final response.
        renderMarkdown(content, event.answer || streamed || "");
        status.remove();
        const citations = (event.citations || []).map(c => `<span class="citation">${escapeHtml(citeLabel(c))}</span>`).join(" ");
        if (citations) pending.insertAdjacentHTML("beforeend", `<div>${citations}</div>`);
        if (event.grounded) pending.insertAdjacentHTML("beforeend", `<div class="grounded-note">✓ Grounded in retrieved evidence</div>`);
        state.agentLiveTool = null;
        renderAgentArtifacts(); renderAgentTrace();
        loadAgentSessions().catch(() => {});
      }
    });
  } catch(e) {
    pending.textContent = `I couldn't complete that request: ${e.message}`;
    state.agentLiveTool = null;
    renderAgentTrace();
  } finally {
    setAgentActivity(false);
    scrollMessages();
    $("#agent-question").focus();
  }
}
function appendAgentMessage(text, kind, attachment = null) {
  const div=document.createElement("div");
  div.className=`message ${kind}`;
  if (kind === "assistant") {
    div.classList.add("markdown-body");
    renderMarkdown(div, text);
  } else {
    if (attachment?.dataUrl) {
      div.innerHTML = `<div class="message-attachment"><img src="${escapeHtml(attachment.dataUrl)}" alt=""><span>${escapeHtml(attachment.name)}</span></div><div>${escapeHtml(text)}</div>`;
    } else div.textContent=text;
  }
  $("#agent-messages").append(div);
  scrollMessages();
  return div;
}
function renderAgentSessions() {
  const target = $("#agent-sessions");
  target.innerHTML = state.agentSessions.length ? state.agentSessions.map(session => `
    <button class="session-item ${session.conversation_id === state.agentConversationId ? "active" : ""}" type="button" data-agent-session="${escapeHtml(session.conversation_id)}">
      <div><strong>${escapeHtml(session.title || "New conversation")}</strong><small>${escapeHtml(new Date(session.updated_at).toLocaleString())}</small></div>
      <span class="session-delete" role="button" aria-label="Delete conversation" data-delete-session="${escapeHtml(session.conversation_id)}">×</span>
    </button>`).join("") : `<div class="session-empty">Your conversations will appear here.</div>`;
}
async function loadAgentSessions() {
  const result = await api("/agent/conversations?limit=100");
  state.agentSessions = result.conversations || [];
  renderAgentSessions();
}
function resetAgentConversationView() {
  $("#agent-messages").innerHTML = `<div class="message assistant markdown-body"></div>`;
  renderMarkdown($("#agent-messages .message"), "Hi — ask about the active paper, find related work, compare methods, or request a research report.");
  state.agentArtifacts = [];
  state.agentTrace = [];
  state.agentLiveTool = null;
  renderAgentArtifacts();
  renderAgentTrace();
}
function startNewAgentConversation() {
  state.agentConversationId = null;
  localStorage.removeItem(AGENT_CONVERSATION_KEY);
  resetAgentConversationView();
  renderAgentSessions();
  $("#agent-live-status").textContent = "New conversation";
  $("#agent-question").focus();
}
async function selectAgentConversation(conversationId) {
  if (state.agentActive || !conversationId) return;
  state.agentConversationId = conversationId;
  localStorage.setItem(AGENT_CONVERSATION_KEY, conversationId);
  resetAgentConversationView();
  renderAgentSessions();
  await restoreAgentConversation();
}
async function deleteAgentConversation(conversationId) {
  await api(`/agent/conversations/${encodeURIComponent(conversationId)}`, { method: "DELETE" });
  state.agentSessions = state.agentSessions.filter(item => item.conversation_id !== conversationId);
  if (state.agentConversationId === conversationId) startNewAgentConversation();
  renderAgentSessions();
}
async function restoreAgentConversation() {
  if (!state.agentConversationId) return;
  try {
    const result = await api(`/agent/conversations/${encodeURIComponent(state.agentConversationId)}`);
    const messages = Array.isArray(result.turns) ? result.turns : (Array.isArray(result.messages) ? result.messages : []);
    if (!messages.length) return;
    const target = $("#agent-messages");
    target.innerHTML = "";
    messages.forEach(message => {
      if (!message || !["user", "assistant"].includes(message.role)) return;
      appendAgentMessage(String(message.content || ""), message.role);
    });
    scrollMessages(target, "auto");
    $("#agent-live-status").textContent = "Conversation restored";
    renderAgentSessions();
  } catch (error) {
    if (error.status === 404) {
      state.agentConversationId = null;
      localStorage.removeItem(AGENT_CONVERSATION_KEY);
      renderAgentSessions();
      return;
    }
    toast(`Could not restore conversation: ${error.message}`, "error");
  }
}
function saveIngestionJobs() {
  localStorage.setItem(INGESTION_JOBS_KEY, JSON.stringify(state.ingestionJobs));
}
function trackIngestionJob(jobId, paperId, filename = "") {
  if (!jobId) return;
  state.ingestionJobs[jobId] = { paperId, filename, startedAt: Date.now() };
  saveIngestionJobs();
  pollIngestionJob(jobId);
}
async function pollIngestionJob(jobId) {
  if (activeJobPolls.has(jobId) || !state.ingestionJobs[jobId]) return;
  activeJobPolls.add(jobId);
  try {
    while (state.ingestionJobs[jobId]) {
      const job = await api(`/jobs/${encodeURIComponent(jobId)}`);
      const percent = Math.round(Number(job.progress || 0) * 100);
      $("#agent-upload-state").textContent = `Parsing in background · ${percent}% · safe to refresh`;
      if (["completed", "failed"].includes(job.status)) {
        const tracked = state.ingestionJobs[jobId];
        delete state.ingestionJobs[jobId];
        saveIngestionJobs();
        await loadLibrary();
        if (job.status === "completed") {
          if (job.paper_id) selectPaper(job.paper_id);
          $("#agent-upload-state").textContent = "Paper ready";
          toast(`${tracked.filename || "Paper"} finished processing`);
        } else {
          $("#agent-upload-state").textContent = `Upload failed: ${job.error || "processing failed"}`;
          toast("Background ingestion failed", "error");
        }
        break;
      }
      await new Promise(resolve => setTimeout(resolve, 2000));
    }
  } catch (error) {
    if (error.status === 404) {
      delete state.ingestionJobs[jobId];
      saveIngestionJobs();
    } else {
      setTimeout(() => { activeJobPolls.delete(jobId); pollIngestionJob(jobId); }, 5000);
      return;
    }
  } finally {
    activeJobPolls.delete(jobId);
  }
}
function resumeIngestionJobs() {
  Object.keys(state.ingestionJobs).forEach(jobId => pollIngestionJob(jobId));
}
function clearAgentImage() {
  state.agentImage = null;
  $("#agent-image-input").value = "";
  $("#agent-image-preview").classList.add("hidden");
  $("#agent-image-thumbnail").removeAttribute("src");
}
function selectAgentImage(file) {
  if (!file) return;
  const validTypes = new Set(["image/png", "image/jpeg", "image/webp"]);
  if (!validTypes.has(file.type)) {
    clearAgentImage();
    return toast("Attach a PNG, JPEG, or WebP image", "error");
  }
  if (file.size > 5 * 1024 * 1024) {
    clearAgentImage();
    return toast("Image attachments must be 5 MB or smaller", "error");
  }
  const reader = new FileReader();
  reader.onload = () => {
    state.agentImage = { dataUrl: String(reader.result), name: file.name, size: file.size };
    $("#agent-image-thumbnail").src = state.agentImage.dataUrl;
    $("#agent-image-name").textContent = file.name;
    $("#agent-image-size").textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB · ready to send`;
    $("#agent-image-preview").classList.remove("hidden");
  };
  reader.onerror = () => toast("Could not read that image", "error");
  reader.readAsDataURL(file);
}
async function uploadFromAgent(file) {
  if (!file) return;
  $("#agent-upload-state").textContent = `Parsing ${file.name} with Docling…`;
  const form=new FormData(); form.append("file",file);
  try {
    const result=await api("/papers",{method:"POST",body:form});
    if (result.status === "accepted" && result.job_id) {
      $("#agent-upload-state").textContent="Queued for background parsing · safe to refresh";
      trackIngestionJob(result.job_id, result.paper_id, file.name);
      await loadLibrary();
    } else {
      $("#agent-upload-state").textContent=`Ready · ${result.pages} pages`;
      await loadLibrary(); selectPaper(result.paper_id);
      toast("Paper uploaded and added to agent context");
    }
  } catch(e) { $("#agent-upload-state").textContent=`Upload failed: ${e.message}`; }
}

async function askPaper(question) {
  if (!state.activePaperId) return toast("Choose a paper first", "error");
  appendMessage(question, "user"); const pending = appendMessage("Searching the paper and checking evidence…", "assistant");
  try {
    const payload = { question };
    if (state.conversationIds[state.activePaperId]) payload.conversation_id = state.conversationIds[state.activePaperId];
    const result = await api(`/papers/${state.activePaperId}/qa`, { method: "POST", body: JSON.stringify(payload) });
    if (result.conversation_id) state.conversationIds[state.activePaperId] = result.conversation_id;
    const a = result.answer || {}; const cites = (a.citations || []).map(c => `<span class="citation">${escapeHtml(citeLabel(c))}</span>`).join(" ");
    const evidence = (a.evidence || []).slice(0,3).map(e => `<div class="evidence">${escapeHtml(e.text?.slice(0,260))}</div>`).join("");
    pending.innerHTML = `<div class="markdown-body">${markdownHtml(a.answer || "The available evidence is insufficient.")}</div><div>${cites}</div>${evidence}${(a.citations || []).length ? `<div class="grounded-note">✓ Grounded in retrieved evidence</div>` : ""}`;
    renderMath(pending);
  } catch (e) { pending.textContent = `Could not answer: ${e.message}`; }
}
async function compare() {
  const ids = $$("#compare-papers input:checked").map(x => x.value); if (ids.length < 2) return toast("Select at least two papers", "error");
  const target = $("#compare-result"); target.innerHTML = `<div class="operation"><div class="spinner"></div><strong>Comparing retrieved evidence…</strong></div>`;
  try {
    const r = await api("/compare", { method: "POST", body: JSON.stringify({ paper_ids: ids, question: $("#compare-question").value }) });
    target.innerHTML = `<div class="result-block card"><span class="eyebrow">COMPARISON</span><h2>${escapeHtml(r.question)}</h2>
      ${(r.paper_findings || []).map(f => `<div class="finding"><h3>${escapeHtml(f.title || titleOf(state.papers.find(p=>p.paper_id===f.paper_id) || {}))}</h3><p>${escapeHtml(f.summary)}</p><div>${(f.citations || []).map(c=>`<span class="citation">${escapeHtml(citeLabel(c))}</span>`).join("")}</div></div>`).join("")}
      <h3>Agreements</h3><ul>${(r.agreements || []).map(x=>`<li>${escapeHtml(x)}</li>`).join("") || "<li>None established from available evidence.</li>"}</ul>
      <h3>Differences</h3><ul>${(r.differences || []).map(x=>`<li>${escapeHtml(x)}</li>`).join("") || "<li>None established from available evidence.</li>"}</ul></div>`;
  } catch(e) { target.innerHTML = `<div class="operation"><strong>Comparison failed:</strong> ${escapeHtml(e.message)}</div>`; }
}
async function discover(event) {
  event.preventDefault(); const target=$("#discover-results"); target.innerHTML=`<div class="operation"><div class="spinner"></div><strong>Searching scholarly APIs…</strong></div>`;
  try {
    const r=await api("/discover",{method:"POST",body:JSON.stringify({query:$("#discover-query").value,source:$("#discover-source").value,limit:10})});
    target.innerHTML=(r.results||[]).map(x=>`<article class="result-item"><span class="status">${escapeHtml(x.source||r.source||"paper")}</span><h3>${escapeHtml(x.title)}</h3><p>${escapeHtml((x.authors||[]).join(", "))}${x.year?` · ${x.year}`:""}</p><p>${escapeHtml((x.abstract||"").slice(0,400))}</p>${x.source_url?`<a target="_blank" rel="noopener" href="${escapeHtml(x.source_url)}">Open source ↗</a>`:""}${(x.library_matches||[]).length?` <span class="status">In library</span>`:""}</article>`).join("")||`<div class="empty-inline">No results found.</div>`;
  } catch(e){target.innerHTML=`<div class="operation"><strong>Search failed:</strong> ${escapeHtml(e.message)}</div>`;}
}
async function research() {
  const ids=$$("#research-papers input:checked").map(x=>x.value), target=$("#research-result"); target.innerHTML=`<div class="operation"><div class="spinner"></div><strong>Running bounded research workflow…</strong></div>`;
  try {
    const r=await api("/research",{method:"POST",body:JSON.stringify({research_question:$("#research-question").value,selected_papers:ids,enable_external:false,max_external_searches:0})});
    target.innerHTML=`<span class="eyebrow">VERIFIED REPORT</span><h2>Research synthesis</h2><div>${escapeHtml(r.final_report||"No report generated.").replace(/\n/g,"<br>")}</div>
      <h3>Evidence critic</h3><ul>${(r.critic_feedback||[]).map(x=>`<li>${escapeHtml(typeof x==="string"?x:JSON.stringify(x))}</li>`).join("")||"<li>No critic feedback.</li>"}</ul>
      <details><summary>Workflow trace (${(r.tool_calls||[]).length} tool calls)</summary><pre>${escapeHtml(JSON.stringify(r.tool_calls||[],null,2))}</pre></details>`;
  } catch(e){target.innerHTML=`<strong>Workflow failed:</strong> ${escapeHtml(e.message)}`;}
}
async function upload() {
  const file=$("#pdf-file").files[0]; if(!file)return; const op=$("#upload-operation"), result=$("#upload-result");
  op.classList.remove("hidden"); result.innerHTML=""; $("#upload-button").disabled=true; $("#upload-status").textContent="Parsing your paper…";
  const form=new FormData(); form.append("file",file);
  try {
    const r=await api("/papers",{method:"POST",body:form});
    if(r.status==="accepted"&&r.job_id){$("#upload-status").textContent="Parsing in background";$("#upload-detail").textContent="You can leave or refresh this page. Progress will resume automatically.";trackIngestionJob(r.job_id,r.paper_id,file.name);}
    else{$("#upload-status").textContent="Paper ready";$("#upload-detail").textContent=`${r.pages} pages · ${r.text_elements} text elements · ${r.tables} tables · ${r.pictures} figures · ${r.formulas} formulas`;}
    result.innerHTML=`<div class="card result-block"><span class="status">${escapeHtml(r.status)}</span><h2>${escapeHtml(r.filename)}</h2><p>Parsing completed successfully. Open it by title — no ID to remember.</p><button class="button primary" data-open-paper="${escapeHtml(r.paper_id)}">Open in reader</button></div>`;
    await loadLibrary(); selectPaper(r.paper_id); toast(r.status==="accepted"?"Paper queued for processing":"Paper added to your library");
  } catch(e){$("#upload-status").textContent="Upload failed";$("#upload-detail").textContent=e.message;toast(e.message,"error");}
  finally{$("#upload-button").disabled=false;}
}
async function deletePaper(id) {
  try { await api(`/papers/${id}`,{method:"DELETE"}); if(state.activePaperId===id)selectPaper(""); await loadLibrary(); toast("Paper deleted"); }
  catch(e){toast(e.message,"error");} finally{$("#confirm-modal").classList.add("hidden");state.deleteId=null;}
}

document.addEventListener("click", e => {
  const route=e.target.closest("[data-route]")?.dataset.route;if(route){e.preventDefault();navigate(route);}
  const open=e.target.closest("[data-open-paper]")?.dataset.openPaper;if(open)selectPaper(open,"workspace");
  const chat=e.target.closest("[data-chat-paper]")?.dataset.chatPaper;if(chat)selectPaper(chat,"chat");
  const del=e.target.closest("[data-delete-paper]")?.dataset.deletePaper;if(del){state.deleteId=del;$("#confirm-modal").classList.remove("hidden");}
  const deleteSession=e.target.closest("[data-delete-session]")?.dataset.deleteSession;
  if(deleteSession){e.preventDefault();e.stopPropagation();deleteAgentConversation(deleteSession).catch(error=>toast(error.message,"error"));return;}
  const session=e.target.closest("[data-agent-session]")?.dataset.agentSession;
  if(session){selectAgentConversation(session).catch(error=>toast(error.message,"error"));return;}
  const prompt=e.target.closest("[data-agent-prompt]")?.dataset.agentPrompt;if(prompt){setToolsMenu(false);$("#agent-question").value=prompt;$("#agent-form").requestSubmit();}
  const action=e.target.closest("[data-agent-action]")?.dataset.agentAction;
  if(action){setToolsMenu(false);if(action==="upload-image")$("#agent-image-input").click();if(action==="upload-pdf")$("#agent-pdf-file").click();}
  if(!e.target.closest(".agent-tools"))setToolsMenu(false);
});
$("#global-paper-select").addEventListener("change", e=>selectPaper(e.target.value));
$("#refresh-library").addEventListener("click",()=>loadLibrary(true));
$("#library-search").addEventListener("input",renderLibrary);$("#library-status").addEventListener("change",renderLibrary);
$("#reader-tabs").addEventListener("click",e=>{if(e.target.dataset.tab){state.readerTab=e.target.dataset.tab;$$('#reader-tabs button').forEach(b=>b.classList.toggle("active",b===e.target));renderReaderPanel();}});
$("#reader-index").addEventListener("click",async()=>{if(!state.activePaperId)return toast("Choose a paper first","error");try{await api(`/papers/${state.activePaperId}/index`,{method:"POST",body:JSON.stringify({force:true})});toast("Retrieval index rebuilt");state.chunks=await api(`/papers/${state.activePaperId}/chunks`);renderReaderPanel();}catch(e){toast(e.message,"error");}});
$("#chat-form").addEventListener("submit",e=>{e.preventDefault();const q=$("#chat-question").value.trim();if(q){$("#chat-question").value="";askPaper(q);}});
$("#agent-form").addEventListener("submit",e=>{
  e.preventDefault();
  if(state.agentActive)return;
  const q=$("#agent-question").value.trim();
  if(q||state.agentImage){
    $("#agent-question").value="";
    $("#agent-question").style.height="";
    askAgent(q||"Analyze the attached image.");
  }
});
["#agent-question","#chat-question"].forEach(selector=>$(selector).addEventListener("keydown",e=>{
  if(e.key==="Enter"&&!e.shiftKey&&!e.isComposing){
    e.preventDefault();
    e.currentTarget.form.requestSubmit();
  }
}));
$("#agent-question").addEventListener("input",e=>{e.target.style.height="auto";e.target.style.height=`${Math.min(e.target.scrollHeight,160)}px`;});
$("#agent-image-input").addEventListener("change",e=>selectAgentImage(e.target.files[0]));
$("#agent-image-remove").addEventListener("click",clearAgentImage);
$("#agent-pdf-file").addEventListener("change",e=>uploadFromAgent(e.target.files[0]));
$("#agent-tools-toggle").addEventListener("click",e=>{e.stopPropagation();setToolsMenu($("#agent-tools-menu").classList.contains("hidden"));});
$("#agent-new-chat").addEventListener("click",startNewAgentConversation);
document.addEventListener("keydown",e=>{if(e.key==="Escape")setToolsMenu(false);});
$("#agent-output-tabs").addEventListener("click",e=>{if(!e.target.dataset.agentTab)return;$$('#agent-output-tabs button').forEach(b=>b.classList.toggle("active",b===e.target));const trace=e.target.dataset.agentTab==="trace";$("#agent-results").classList.toggle("hidden",trace);$("#agent-trace").classList.toggle("hidden",!trace);});
$("#compare-button").addEventListener("click",compare);$("#discover-form").addEventListener("submit",discover);$("#research-button").addEventListener("click",research);
$("#dropzone").addEventListener("click",()=>$("#pdf-file").click());$("#dropzone").addEventListener("dragover",e=>{e.preventDefault();$("#dropzone").classList.add("drag");});$("#dropzone").addEventListener("dragleave",()=>$("#dropzone").classList.remove("drag"));
$("#dropzone").addEventListener("drop",e=>{e.preventDefault();$("#dropzone").classList.remove("drag");if(e.dataTransfer.files[0]){const dt=new DataTransfer();dt.items.add(e.dataTransfer.files[0]);$("#pdf-file").files=dt.files;$("#pdf-file").dispatchEvent(new Event("change"));}});
$("#pdf-file").addEventListener("change",e=>{const f=e.target.files[0],chip=$("#file-chip");if(f){chip.textContent=`${f.name} · ${(f.size/1024/1024).toFixed(2)} MB`;chip.classList.remove("hidden");$("#upload-button").disabled=false;}});
$("#upload-button").addEventListener("click",upload);$("#cancel-delete").addEventListener("click",()=>$("#confirm-modal").classList.add("hidden"));$("#confirm-delete").addEventListener("click",()=>state.deleteId&&deletePaper(state.deleteId));$("#mobile-menu").addEventListener("click",()=>$(".sidebar").classList.toggle("open"));
$("#auth-form").addEventListener("submit",e=>{e.preventDefault();submitAuth("signin");});
$("#auth-signup").addEventListener("click",()=>submitAuth("signup"));
$("#auth-guest").addEventListener("click",()=>continueAsGuest());
$("#account-button").addEventListener("click",()=>{storeAuthSession(null);state.papers=[];state.agentSessions=[];state.guestQuota=null;startNewAgentConversation();renderAccount();$("#auth-modal").classList.remove("hidden");});

(async function init(){
  await consumeAuthRedirect();
  navigate(location.hash.slice(1)||"home");
  await loadCapabilities();
  const authenticated = initializeAuth();
  try { await api("/health"); $("#health-dot").classList.add("ok"); $("#health-label").textContent="API connected"; if(authenticated){await Promise.all([loadLibrary(),loadAgentSessions()]);await restoreAgentConversation();resumeIngestionJobs();renderGuestQuota();} }
  catch(e){$("#health-dot").classList.add("bad");$("#health-label").textContent="API unavailable";toast(`Backend unavailable: ${e.message}`,"error");}
})();
