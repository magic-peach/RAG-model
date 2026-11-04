// Thesis RAG front end: plain JavaScript, no build step.
// All text from the server is inserted with textContent / DOM nodes, never
// innerHTML, so text from the PDF or the model cannot inject markup.

"use strict";

const $ = (sel, root = document) => root.querySelector(sel);

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

// ---------------------------------------------------------------------------
// Icons (inline SVG paths, Lucide-style)
// ---------------------------------------------------------------------------

const ICONS = {
  plus: "M12 5v14M5 12h14",
  search: "M11 19a8 8 0 1 0 0-16 8 8 0 0 0 0 16zM21 21l-4.3-4.3",
  chart: "M3 3v18h18M7 15l4-4 3 3 5-6",
  panel: "M3 5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2zM9 3v18",
  menu: "M4 6h16M4 12h16M4 18h16",
  moon: "M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z",
  sun: "M12 3v2M12 19v2M5.6 5.6l1.4 1.4M17 17l1.4 1.4M3 12h2M19 12h2M5.6 18.4 7 17M17 7l1.4-1.4M12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8z",
  book: "M4 19.5A2.5 2.5 0 0 1 6.5 17H20V3H6.5A2.5 2.5 0 0 0 4 5.5zM4 19.5A2.5 2.5 0 0 0 6.5 22H20v-5",
  send: "M22 2 11 13M22 2l-7 20-4-9-9-4z",
  stop: "M7 7h10v10H7z",
  close: "M18 6 6 18M6 6l12 12",
  upload: "M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12",
  chat: "M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z",
  more: "M12 13a1 1 0 1 0 0-2 1 1 0 0 0 0 2zM19 13a1 1 0 1 0 0-2 1 1 0 0 0 0 2zM5 13a1 1 0 1 0 0-2 1 1 0 0 0 0 2z",
  edit: "M12 20h9M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z",
  trash: "M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6",
  copy: "M9 9h11v11H9zM5 15H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v1",
  check: "M20 6 9 17l-5-5",
  bulb: "M9 18h6M10 22h4M12 2a7 7 0 0 0-4 12.7V17h8v-2.3A7 7 0 0 0 12 2z",
  globe: "M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20zM2 12h20M12 2a15 15 0 0 1 0 20M12 2a15 15 0 0 0 0 20",
  chev: "m6 9 6 6 6-6",
  sparkle: "M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9zM19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z",
  flask: "M9 3h6M10 3v6L4.5 18.5A2 2 0 0 0 6.2 21.5h11.6a2 2 0 0 0 1.7-3L14 9V3",
  filter: "M22 3H2l8 9.5V19l4 2v-8.5z",
  leaf: "M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.5 19 2c1 2 2 4.2 2 8 0 5.5-4.8 10-10 10zM2 21c0-3 1.9-5.4 5.1-6",
  info: "M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20zM12 16v-4M12 8h.01",
};

function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("class", "i");
  svg.setAttribute("aria-hidden", "true");
  const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
  path.setAttribute("d", ICONS[name] || "");
  svg.append(path);
  return svg;
}

function hydrateIcons(root = document) {
  root.querySelectorAll("[data-icon]").forEach((node) => {
    if (!node.querySelector("svg")) node.prepend(icon(node.dataset.icon));
  });
}

// ---------------------------------------------------------------------------
// Client id, theme, layout
// ---------------------------------------------------------------------------

function storageGet(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}
function storageSet(key, value) {
  try { localStorage.setItem(key, value); } catch { /* private mode: fine */ }
}

// A random id per browser scopes saved chats (see src/thesis_rag/store.py).
const CLIENT_ID = (() => {
  let id = storageGet("thesis-rag-client");
  if (!id || !/^[A-Za-z0-9-]{8,64}$/.test(id)) {
    const raw = crypto.randomUUID ? crypto.randomUUID() : `c-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    id = raw.replace(/[^A-Za-z0-9-]/g, "");
    storageSet("thesis-rag-client", id);
  }
  return id;
})();

function api(path, options = {}) {
  const headers = { "X-Client-Id": CLIENT_ID, ...(options.headers || {}) };
  return fetch(path, { ...options, headers });
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const btn = $("#theme-btn");
  btn.replaceChildren(icon(theme === "dark" ? "sun" : "moon"));
  btn.setAttribute("aria-label", theme === "dark" ? "Switch to light mode" : "Switch to dark mode");
}
const prefersDark = window.matchMedia("(prefers-color-scheme: dark)");
applyTheme(storageGet("thesis-rag-theme") || (prefersDark.matches ? "dark" : "light"));
prefersDark.addEventListener("change", (e) => {
  if (!storageGet("thesis-rag-theme")) applyTheme(e.matches ? "dark" : "light");
});
$("#theme-btn").addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  storageSet("thesis-rag-theme", next);
  applyTheme(next);
});

const app = $("#app");
const narrow = () => window.matchMedia("(max-width: 820px)").matches;
$("#collapse-btn").addEventListener("click", () => (narrow() ? app.classList.remove("sidebar-open") : app.classList.add("collapsed")));
$("#open-sidebar").addEventListener("click", () => (narrow() ? app.classList.add("sidebar-open") : app.classList.remove("collapsed")));

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

const state = { chatId: null, chats: [], busy: false, controller: null, view: "ask", docTitle: "" };
let statusTimer = null;

function prettyModel(name) {
  if (/llama[-_.]?3\.1[-_:]?8b/i.test(name)) return "Llama 3.1 8B";
  if (/llama[-_.]?3\.2[-_:]?3b/i.test(name)) return "Llama 3.2 3B";
  if (/gpt-oss-20b/i.test(name)) return "gpt-oss 20B";
  if (/gpt-oss-120b/i.test(name)) return "gpt-oss 120B";
  return name || "Local model";
}

async function refreshStatus() {
  let next = 20000;
  const dot = $("#status-dot");
  try {
    const res = await fetch("/status");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const s = await res.json();
    const ingest = s.ingest || {};
    state.docTitle = s.document ? s.document.title || s.document.source : "";
    $("#doc-title").textContent = state.docTitle || "No document indexed";
    $("#hero-doc").textContent = state.docTitle ? `Answering from “${state.docTitle}”` : "Upload a PDF to get started.";
    $("#model-name").textContent = prettyModel((s.settings && s.settings.llm_model) || "");
    if (ingest.state === "running") {
      dot.dataset.state = "busy";
      $("#doc-sub").textContent = `${ingest.message || "Indexing"} · ${Math.round((ingest.progress || 0) * 100)}%`;
      next = 2000;
    } else if (ingest.state === "error") {
      dot.dataset.state = "error";
      $("#doc-sub").textContent = "Indexing failed";
      $("#doc-sub").title = ingest.error;
    } else if (s.ollama === "offline") {
      dot.dataset.state = "error";
      $("#doc-sub").textContent = "Model offline";
    } else if (s.indexed) {
      dot.dataset.state = "ok";
      $("#doc-sub").textContent = s.ollama === "busy" ? "Indexed · model busy" : "Indexed";
    } else {
      dot.dataset.state = "";
      $("#doc-sub").textContent = "No document";
    }
    $("#upload-label").hidden = !s.ingest_enabled || ingest.state === "running";
  } catch {
    dot.dataset.state = "error";
    $("#doc-sub").textContent = "Server unreachable";
  }
  clearTimeout(statusTimer);
  statusTimer = setTimeout(refreshStatus, next);
}

$("#pdf-input").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  const body = new FormData();
  body.append("file", file);
  $("#doc-sub").textContent = "Uploading…";
  try {
    const res = await api("/ingest", { method: "POST", body });
    if (!res.ok) throw new Error((await res.json()).detail || `HTTP ${res.status}`);
  } catch (err) {
    $("#status-dot").dataset.state = "error";
    $("#doc-sub").textContent = `Upload failed: ${err.message}`;
  }
  event.target.value = "";
  refreshStatus();
});

// ---------------------------------------------------------------------------
// Saved chats (sidebar)
// ---------------------------------------------------------------------------

async function loadChats() {
  try {
    const res = await api("/chats");
    state.chats = res.ok ? await res.json() : [];
  } catch {
    state.chats = [];
  }
  renderChatList();
}

function renderChatList() {
  const list = $("#chat-list");
  const q = $("#chat-search").value.trim().toLowerCase();
  const chats = state.chats.filter((c) => !q || c.title.toLowerCase().includes(q));
  list.replaceChildren();
  if (!chats.length) {
    list.append(el("div", { class: "chat-empty" }, q ? "No chats match." : "Your chats will appear here."));
    return;
  }
  for (const chat of chats) list.append(chatItem(chat));
}

function chatItem(chat) {
  const item = el("div", { class: "chat-item", "aria-current": String(chat.id === state.chatId), dataset: { id: chat.id } });
  const open = el(
    "button",
    { type: "button", class: "open", title: chat.title, onclick: () => openChat(chat.id) },
    icon("chat"),
    el("span", { class: "title" }, chat.title),
  );
  const more = el(
    "button",
    {
      type: "button",
      class: "icon-btn more",
      "aria-label": `Options for ${chat.title}`,
      "aria-haspopup": "menu",
      onclick: (e) => { e.stopPropagation(); toggleMenu(item, chat); },
    },
    icon("more"),
  );
  item.append(open, more);
  return item;
}

function closeMenus() {
  document.querySelectorAll(".menu").forEach((m) => m.remove());
}

function toggleMenu(item, chat) {
  const existing = item.querySelector(".menu");
  closeMenus();
  if (existing) return;
  const menu = el(
    "div",
    { class: "menu", role: "menu" },
    el("button", { type: "button", role: "menuitem", onclick: () => { closeMenus(); startRename(item, chat); } }, icon("edit"), "Rename"),
    el("button", { type: "button", role: "menuitem", class: "danger-text", onclick: () => { closeMenus(); confirmDelete(chat); } }, icon("trash"), "Delete"),
  );
  item.append(menu);
  menu.querySelector("button").focus();
}
document.addEventListener("click", (e) => { if (!e.target.closest(".menu")) closeMenus(); });
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") { closeMenus(); closeDrawer(); }
});

function startRename(item, chat) {
  const field = el("input", { class: "rename", value: chat.title, maxlength: "80", "aria-label": "Chat title" });
  item.replaceChildren(field);
  field.focus();
  field.select();
  let done = false;
  const finish = async (save) => {
    if (done) return;
    done = true;
    const title = field.value.trim();
    if (save && title && title !== chat.title) {
      const res = await api(`/chats/${chat.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title }),
      });
      if (res.ok) chat.title = (await res.json()).title;
      if (chat.id === state.chatId) $("#topbar-title").textContent = chat.title;
    }
    renderChatList();
  };
  field.addEventListener("keydown", (e) => {
    if (e.key === "Enter") finish(true);
    if (e.key === "Escape") finish(false);
  });
  field.addEventListener("blur", () => finish(true));
}

function confirmDelete(chat) {
  const modal = $("#confirm");
  $("#confirm-text").textContent = `“${chat.title}” and its answers will be removed.`;
  modal.hidden = false;
  $("#confirm-cancel").focus();
  const close = () => { modal.hidden = true; $("#confirm-ok").onclick = null; };
  $("#confirm-cancel").onclick = close;
  modal.onclick = (e) => { if (e.target === modal) close(); };
  $("#confirm-ok").onclick = async () => {
    close();
    const res = await api(`/chats/${chat.id}`, { method: "DELETE" });
    if (res.ok || res.status === 404) {
      state.chats = state.chats.filter((c) => c.id !== chat.id);
      if (state.chatId === chat.id) newChat();
      renderChatList();
    }
  };
}

$("#chat-search").addEventListener("input", renderChatList);

function newChat() {
  stopStreaming();
  state.chatId = null;
  $("#thread").replaceChildren();
  $("#hero").hidden = false;
  $("#topbar-title").textContent = "";
  showView("ask");
  renderChatList();
  setSources(null);
  $("#composer-input").focus();
  if (narrow()) app.classList.remove("sidebar-open");
}
$("#new-chat").addEventListener("click", newChat);

async function openChat(id) {
  stopStreaming();
  const res = await api(`/chats/${id}`);
  if (!res.ok) { await loadChats(); return; }
  const chat = await res.json();
  state.chatId = id;
  showView("ask");
  $("#hero").hidden = true;
  $("#topbar-title").textContent = chat.title;
  const thread = $("#thread");
  thread.replaceChildren();
  let last = null;
  for (const m of chat.messages) {
    if (m.role === "user") thread.append(el("div", { class: "msg-user" }, m.content));
    else if (m.answer) {
      const { root, bubble } = botMessage();
      thread.append(root);
      renderAnswer(bubble, m.answer);
      last = bubble;
    }
  }
  setSources(last ? last.dataset.key : null);
  renderChatList();
  $("#scroll").scrollTop = $("#scroll").scrollHeight;
  if (narrow()) app.classList.remove("sidebar-open");
}

// ---------------------------------------------------------------------------
// Rendering answers: citation chips, light Markdown, reasoning, sources
// ---------------------------------------------------------------------------

const TOKEN = /\[[^\[\]]{2,80}\]|\bPassage (\d)\b|\*\*([^*]+)\*\*|\((general knowledge)\)/gi;
const answers = new Map(); // key -> answer object, for the sources drawer
let answerSeq = 0;

// Inline text -> nodes: known citation tags and "Passage N" become chips, **x** bold.
function inline(text, sources, key) {
  const nodes = [];
  let last = 0;
  for (const m of text.matchAll(TOKEN)) {
    let node = null;
    const [whole, passageNo, bold, general] = m;
    if (general) node = el("span", { class: "gk", title: "Background from general knowledge, not from the thesis" }, "general knowledge");
    else if (bold) node = el("strong", {}, bold);
    else {
      const source = passageNo ? sources.find((s) => s.n === Number(passageNo)) : sources.find((s) => s.citation === whole);
      if (source) {
        const label = passageNo ? `Passage ${source.n}` : whole.slice(1, -1);
        node = el(
          "button",
          { type: "button", class: "chip", "aria-label": `Show source ${source.n}: ${source.citation}`, onclick: () => highlightSource(key, source.n) },
          label,
        );
      }
    }
    if (!node) continue;
    nodes.push(text.slice(last, m.index), node);
    last = m.index + whole.length;
  }
  nodes.push(text.slice(last));
  return nodes;
}

const BULLET = /^\s*([-*•]|\d+[.)])\s+/;
function rich(text, sources, key) {
  const frag = document.createDocumentFragment();
  for (const block of text.split(/\n{2,}/)) {
    const lines = block.split("\n").filter((l) => l.trim());
    if (!lines.length) continue;
    const bullets = lines.filter((l) => BULLET.test(l));
    const prose = lines.filter((l) => !BULLET.test(l));
    if (prose.length) frag.append(el("p", {}, inline(prose.join(" "), sources, key)));
    if (bullets.length) frag.append(el("ul", {}, bullets.map((l) => el("li", {}, inline(l.replace(BULLET, ""), sources, key)))));
  }
  return frag;
}

function botMessage() {
  const key = `a${++answerSeq}`;
  const bubble = el("div", { class: "bubble", dataset: { key } });
  const avatar = el("div", { class: "orb small avatar", "aria-hidden": "true" }, el("span"));
  return { root: el("div", { class: "msg-bot" }, avatar, bubble), bubble, key };
}

const GENERAL_BADGE = () => el("span", { class: "badge general" }, icon("globe"), "Not from the thesis · general knowledge");

function renderAnswer(bubble, answer) {
  const key = bubble.dataset.key;
  answers.set(key, answer);
  bubble.classList.remove("streaming");
  const parts = [];
  if (answer.mode === "general") parts.push(GENERAL_BADGE());
  else if (answer.broad) parts.push(el("span", { class: "badge overview" }, icon("sparkle"), "Overview of the whole thesis"));
  parts.push(el("div", { class: "answer" }, rich(answer.answer, answer.sources || [], key)));
  if (answer.reasoning) {
    parts.push(
      el(
        "details",
        { class: "reasoning" },
        el("summary", {}, icon("bulb"), "How I got this", el("span", { class: "chev" }, icon("chev"))),
        el("div", { class: "body" }, rich(answer.reasoning, answer.sources || [], key)),
      ),
    );
  }
  const meta = el("div", { class: "meta" });
  if (answer.sources && answer.sources.length && answer.mode !== "general") {
    meta.append(
      el("button", { type: "button", class: "meta-btn", onclick: () => { setSources(key); openDrawer(); } }, icon("book"), `${answer.sources.length} sources`),
    );
  }
  const copy = el("button", { type: "button", class: "meta-btn", "aria-label": "Copy answer" }, icon("copy"), "Copy");
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(answer.answer + (answer.reasoning ? `\n\nReasoning:\n${answer.reasoning}` : ""));
      copy.replaceChildren(icon("check"), "Copied");
      setTimeout(() => copy.replaceChildren(icon("copy"), "Copy"), 1500);
    } catch { /* clipboard blocked */ }
  });
  meta.append(copy);
  if (answer.mode === "thesis" && !(answer.cited || []).length) {
    meta.append(el("span", { class: "meta-note meta-warn", title: "The model did not cite a passage; check the sources." }, "No citation given"));
  }
  const t = answer.timings || {};
  if (t.first_token) meta.append(el("span", { class: "meta-note" }, `first words in ${t.first_token.toFixed(1)} s · done in ${(t.answer_total || 0).toFixed(1)} s`));
  parts.push(meta);
  bubble.replaceChildren(...parts);
}

// ---------------------------------------------------------------------------
// Sources drawer
// ---------------------------------------------------------------------------

let drawerKey = null;

function setSources(key) {
  drawerKey = key;
  const list = $("#sources-list");
  list.replaceChildren();
  const answer = key ? answers.get(key) : null;
  if (!answer || !answer.sources || !answer.sources.length || answer.mode === "general") {
    list.append(
      el("p", { class: "muted" },
        answer && answer.mode === "general"
          ? "This answer did not come from the thesis, so it has no sources."
          : "Passages behind an answer appear here. Select a citation to highlight its passage."),
    );
    return;
  }
  for (const s of answer.sources) {
    const pages = s.page_start === s.page_end ? `p. ${s.page_start}` : `pp. ${s.page_start}–${s.page_end}`;
    const cited = (answer.cited || []).includes(s.n);
    const score = Math.max(0, Math.min(1, Number(s.score) || 0));
    list.append(
      el(
        "article",
        { class: "source", id: `src-${key}-${s.n}`, tabindex: "-1" },
        el("div", { class: "source-head" }, el("span", {}, el("span", { class: "source-num" }, s.n), s.citation), el("span", { class: "muted" }, pages)),
        el("div", { class: "source-title" }, s.header),
        el("p", { class: "source-snippet" }, s.snippet),
        el(
          "div",
          { class: "source-foot" },
          el("span", { class: `tag${cited ? " cited" : ""}` }, cited ? "cited" : s.chunk_type),
          el("span", { class: "bar", title: `Relevance ${score.toFixed(2)}` }, el("i", { style: `width:${Math.round(score * 100)}%` })),
          el("span", {}, score.toFixed(2)),
        ),
      ),
    );
  }
}

function openDrawer() {
  $("#drawer").hidden = false;
  $("#sources-btn").setAttribute("aria-expanded", "true");
}
function closeDrawer() {
  $("#drawer").hidden = true;
  $("#sources-btn").setAttribute("aria-expanded", "false");
}
$("#sources-btn").addEventListener("click", () => ($("#drawer").hidden ? openDrawer() : closeDrawer()));
$("#close-drawer").addEventListener("click", closeDrawer);

function highlightSource(key, n) {
  if (drawerKey !== key) setSources(key);
  openDrawer();
  const card = document.getElementById(`src-${key}-${n}`);
  if (!card) return;
  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
  card.focus({ preventScroll: true });
  card.classList.add("flash");
  setTimeout(() => card.classList.remove("flash"), 1600);
}

// ---------------------------------------------------------------------------
// Asking (Server-Sent Events over a POST response)
// ---------------------------------------------------------------------------

async function* readSSE(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      let event = "message";
      let data = "";
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      yield { event, data: data ? JSON.parse(data) : {} };
    }
  }
}

const sendBtn = $("#send-btn");
const input = $("#composer-input");

function updateSend() {
  sendBtn.disabled = !state.busy && !input.value.trim();
}
function setBusy(busy) {
  state.busy = busy;
  sendBtn.classList.toggle("stop", busy);
  sendBtn.replaceChildren(icon(busy ? "stop" : "send"));
  sendBtn.setAttribute("aria-label", busy ? "Stop answering" : "Send");
  updateSend();
}
function stopStreaming() {
  if (state.controller) state.controller.abort();
}

function keepScrolled() {
  const scroll = $("#scroll");
  if (scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight < 160) scroll.scrollTop = scroll.scrollHeight;
}

function moveChatToTop(id) {
  const i = state.chats.findIndex((c) => c.id === id);
  if (i > 0) state.chats.unshift(...state.chats.splice(i, 1));
  renderChatList();
}

async function ask(question) {
  if (state.busy) return;
  showView("ask");
  $("#hero").hidden = true;
  const thread = $("#thread");
  const userMsg = el("div", { class: "msg-user" }, question);
  thread.append(userMsg);
  const { root, bubble } = botMessage();
  bubble.classList.add("streaming");
  bubble.append(el("div", { class: "thinking" }, el("span", { class: "dots" }, el("i"), el("i"), el("i")), "Reading the thesis…"));
  thread.append(root);
  root.scrollIntoView({ block: "end", behavior: "smooth" });
  setBusy(true);
  state.controller = new AbortController();

  let text = "";
  let answerNode = null;
  const ensureAnswerNode = () => {
    if (!answerNode) {
      bubble.replaceChildren();
      answerNode = el("div", { class: "answer" });
      bubble.append(answerNode);
    }
  };
  try {
    const res = await api("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, chat_id: state.chatId }),
      signal: state.controller.signal,
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(typeof detail.detail === "string" ? detail.detail : `Request failed (HTTP ${res.status})`);
    }
    for await (const { event, data } of readSSE(res)) {
      if (event === "chat") {
        state.chatId = data.id;
        state.chats.unshift({ id: data.id, title: data.title, updated_at: Date.now() / 1000 });
        $("#topbar-title").textContent = data.title;
        renderChatList();
      } else if (event === "mode") {
        bubble.replaceChildren();
        if (data.mode === "general") bubble.append(GENERAL_BADGE());
        answerNode = el("div", { class: "answer" });
        bubble.append(answerNode);
      } else if (event === "token") {
        text += data.text;
        ensureAnswerNode();
        answerNode.textContent = text; // plain while streaming; chips and lists at the end
        keepScrolled();
      } else if (event === "answer") {
        renderAnswer(bubble, data);
        setSources(bubble.dataset.key);
        moveChatToTop(state.chatId);
      } else if (event === "error") {
        throw new Error(data.message);
      }
    }
  } catch (err) {
    bubble.classList.remove("streaming");
    if (err.name === "AbortError") {
      if (answerNode) answerNode.append(el("span", { class: "meta-note" }, " (stopped)"));
      else bubble.replaceChildren(el("span", { class: "meta-note" }, "Stopped."));
    } else {
      bubble.replaceChildren(
        el("div", { class: "msg-error", role: "alert" },
          el("div", {}, `Something went wrong: ${err.message}`),
          el("button", { type: "button", class: "meta-btn", onclick: () => { userMsg.remove(); root.remove(); ask(question); } }, "Try again")),
      );
    }
  } finally {
    state.controller = null;
    setBusy(false);
  }
}

function autosize() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
}
input.addEventListener("input", () => { autosize(); updateSend(); });
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $("#ask-form").requestSubmit();
  }
});
$("#ask-form").addEventListener("submit", (event) => {
  event.preventDefault();
  if (state.busy) { stopStreaming(); return; }
  const question = input.value.trim();
  if (!question) return;
  input.value = "";
  autosize();
  updateSend();
  ask(question);
});
document.querySelectorAll(".suggestion").forEach((b) => b.addEventListener("click", () => ask(b.textContent.trim())));

// ---------------------------------------------------------------------------
// Analyze view
// ---------------------------------------------------------------------------

function showView(view) {
  state.view = view;
  $("#ask-view").hidden = view !== "ask";
  $("#analyze-view").hidden = view !== "analyze";
  $("#analyze-nav").setAttribute("aria-current", view === "analyze" ? "page" : "false");
  if (view === "analyze") $("#topbar-title").textContent = "Analyze new data";
}
$("#analyze-nav").addEventListener("click", () => {
  showView("analyze");
  if (narrow()) app.classList.remove("sidebar-open");
});
$("#attach-btn").addEventListener("click", () => {
  showView("analyze");
  $("#csv-input").click();
});

const csvInput = $("#csv-input");
function loadColumns(file) {
  $("#csv-name").textContent = file.name;
  const reader = new FileReader();
  reader.onload = () => {
    const header = String(reader.result).split(/\r?\n/)[0] || "";
    const sep = [",", ";", "\t"].sort((a, b) => header.split(b).length - header.split(a).length)[0];
    const select = $("#group-col");
    select.replaceChildren(el("option", { value: "" }, "Detect automatically"));
    for (const name of header.split(sep).map((c) => c.trim().replace(/^"|"$/g, "")).filter(Boolean)) {
      select.append(el("option", { value: name }, name));
    }
  };
  reader.readAsText(file.slice(0, 64 * 1024));
}
csvInput.addEventListener("change", () => csvInput.files[0] && loadColumns(csvInput.files[0]));
const dropzone = $("#dropzone");
for (const type of ["dragenter", "dragover"]) {
  dropzone.addEventListener(type, (e) => { e.preventDefault(); dropzone.classList.add("drag"); });
}
for (const type of ["dragleave", "drop"]) dropzone.addEventListener(type, () => dropzone.classList.remove("drag"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  if (e.dataTransfer.files.length) {
    csvInput.files = e.dataTransfer.files;
    loadColumns(csvInput.files[0]);
  }
});

const fmt = (v) => (v === null || v === undefined ? "–" : String(v));
function table(columns, rows) {
  return el("div", { class: "table-wrap" },
    el("table", {},
      el("thead", {}, el("tr", {}, columns.map(([label]) => el("th", { scope: "col" }, label)))),
      el("tbody", {}, rows.map((row) => el("tr", {}, columns.map(([, key]) => {
        const value = typeof key === "function" ? key(row) : row[key];
        return el("td", { class: typeof value === "number" ? "num" : null }, value instanceof Node ? value : fmt(value));
      }))))));
}

function renderAnalysis(result) {
  const out = $("#analyze-out");
  const s = result.stats;
  out.replaceChildren(el("div", { class: "banner", role: "note" }, icon("info"), el("span", {}, result.label)));
  if (result.note) out.append(el("p", { class: "muted" }, result.note));
  if (result.explanation) {
    const { root, bubble } = botMessage();
    out.append(el("div", { class: "glass card" }, el("h3", {}, "What this means, according to the thesis"), root));
    renderAnswer(bubble, { answer: result.explanation, sources: result.sources, cited: result.cited, mode: "thesis", reasoning: "" });
    setSources(bubble.dataset.key);
  }
  out.append(
    el("div", { class: "glass card" }, el("h3", {}, `Descriptive statistics · ${s.rows} rows`),
      s.descriptive.length
        ? table([["Column", "column"], ["n", "n"], ["Missing", "missing"], ["Mean", "mean"], ["SD", "std"], ["Min", "min"], ["Median", "median"], ["Max", "max"]], s.descriptive)
        : el("p", { class: "muted" }, "No numeric columns found.")),
  );
  if (s.comparisons.length) {
    out.append(
      el("div", { class: "glass card" }, el("h3", {}, `Group comparisons by ${s.group_column}`),
        table([
          ["Column", "column"],
          ["Test", "test"],
          ["p", "p_value"],
          ["p (Holm)", "p_adjusted"],
          ["Effect", (r) => (r.cohens_d !== undefined ? `d = ${fmt(r.cohens_d)}` : `η² = ${fmt(r.eta_sq)}`)],
          ["", (r) => (r.significant ? el("span", { class: "sig" }, "significant") : "")],
          ["Group means", (r) => Object.entries(r.group_means).map(([g, m]) => `${g}: ${fmt(m)}`).join(" · ")],
        ], s.comparisons)),
    );
  }
  if (s.thresholds.length) {
    out.append(
      el("div", { class: "glass card" }, el("h3", {}, "Thesis thresholds applied to your data"),
        table([
          ["Rule", "rule"],
          ["Pass", (r) => `${r.passing} / ${r.n}`],
          ["Share", (r) => (r.passing_share === null ? "–" : `${Math.round(r.passing_share * 100)}%`)],
          ["Source", "citation"],
        ], s.thresholds)),
    );
  }
  if (s.correlations.length) {
    out.append(
      el("div", { class: "glass card" }, el("h3", {}, "Correlations (|r| ≥ 0.3)"),
        table([["X", "x"], ["Y", "y"], ["r", "r"], ["p", "p_value"], ["n", "n"]], s.correlations)),
    );
  }
}

$("#analyze-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = csvInput.files[0];
  if (!file) { csvInput.click(); return; }
  const body = new FormData();
  body.append("file", file);
  body.append("group_col", $("#group-col").value);
  body.append("explain", String($("#explain").checked));
  const button = $("#analyze-btn");
  button.disabled = true;
  const out = $("#analyze-out");
  out.replaceChildren(
    el("div", { class: "glass loading", role: "status" }, el("span", { class: "dots" }, el("i"), el("i"), el("i")),
      $("#explain").checked ? "Computing statistics, then explaining them against the thesis…" : "Computing statistics…"),
  );
  try {
    const res = await api("/analyze", { method: "POST", body });
    const data = await res.json();
    if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : `HTTP ${res.status}`);
    renderAnalysis(data);
  } catch (err) {
    out.replaceChildren(el("div", { class: "msg-error", role: "alert" }, `Analysis failed: ${err.message}`));
  } finally {
    button.disabled = false;
  }
});

// ---------------------------------------------------------------------------

hydrateIcons();
updateSend();
refreshStatus();
loadChats();
