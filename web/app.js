// Thesis RAG front end: plain JavaScript, no build step.
// All server text is inserted with textContent / DOM nodes, never innerHTML,
// so text from the PDF or the model cannot inject markup.

"use strict";

const $ = (sel, root = document) => root.querySelector(sel);

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "dataset") Object.assign(node.dataset, value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

const state = { indexed: false, busy: false, statusTimer: null };

async function refreshStatus() {
  let next = 15000;
  try {
    const res = await fetch("/status");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const s = await res.json();
    state.indexed = s.indexed;
    const pill = $("#status-pill");
    const ingest = s.ingest || {};
    if (ingest.state === "running") {
      pill.dataset.state = "busy";
      $("#status-text").textContent = `${ingest.message || "Indexing"} (${Math.round((ingest.progress || 0) * 100)}%)`;
      next = 2000;
    } else if (ingest.state === "error") {
      pill.dataset.state = "error";
      $("#status-text").textContent = "Indexing failed";
      pill.title = ingest.error;
    } else if (!s.ollama) {
      pill.dataset.state = "error";
      $("#status-text").textContent = s.indexed ? "Indexed · Ollama offline" : "Ollama offline";
    } else if (s.indexed) {
      pill.dataset.state = "ok";
      $("#status-text").textContent = "Indexed";
    } else {
      pill.dataset.state = "";
      $("#status-text").textContent = "No document";
    }
    $("#doc-name").textContent = s.document ? s.document.title || s.document.source : "No document indexed";
    document.title = s.document ? `${s.document.title} · Thesis RAG` : "Thesis RAG";
    $("#upload-label").hidden = s.indexed || ingest.state === "running";
  } catch (err) {
    $("#status-pill").dataset.state = "error";
    $("#status-text").textContent = "Server unreachable";
  }
  clearTimeout(state.statusTimer);
  state.statusTimer = setTimeout(refreshStatus, next);
}

$("#pdf-input").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  const body = new FormData();
  body.append("file", file);
  $("#status-text").textContent = "Uploading…";
  try {
    const res = await fetch("/ingest", { method: "POST", body });
    if (!res.ok) throw new Error((await res.json()).detail || `HTTP ${res.status}`);
  } catch (err) {
    $("#status-pill").dataset.state = "error";
    $("#status-text").textContent = `Upload failed: ${err.message}`;
  }
  event.target.value = "";
  refreshStatus();
});

// ---------------------------------------------------------------------------
// Tabs (WAI-ARIA tabs pattern: arrow keys move between tabs)
// ---------------------------------------------------------------------------

const tabs = [...document.querySelectorAll('[role="tab"]')];

function selectTab(tab, focus = true) {
  for (const t of tabs) {
    const selected = t === tab;
    t.setAttribute("aria-selected", String(selected));
    t.tabIndex = selected ? 0 : -1;
    document.getElementById(t.getAttribute("aria-controls")).hidden = !selected;
  }
  if (focus) tab.focus();
}

for (const tab of tabs) {
  tab.addEventListener("click", () => selectTab(tab));
  tab.addEventListener("keydown", (event) => {
    const i = tabs.indexOf(tab);
    const target = {
      ArrowRight: tabs[(i + 1) % tabs.length],
      ArrowLeft: tabs[(i - 1 + tabs.length) % tabs.length],
      Home: tabs[0],
      End: tabs[tabs.length - 1],
    }[event.key];
    if (target) {
      event.preventDefault();
      selectTab(target);
    }
  });
}

// ---------------------------------------------------------------------------
// Answers with citation chips
// ---------------------------------------------------------------------------

const TAG = /\[[^\[\]]{2,80}\]/g;

// Inline text -> nodes, turning known citation tags into chip buttons.
function inline(text, sources, panelId) {
  const nodes = [];
  let last = 0;
  for (const match of text.matchAll(TAG)) {
    const source = sources.find((s) => s.citation === match[0]);
    if (!source) continue;
    nodes.push(text.slice(last, match.index));
    nodes.push(
      el(
        "button",
        { type: "button", class: "chip", dataset: { n: source.n, panel: panelId }, "aria-label": `Show source ${source.n}: ${source.citation}` },
        match[0].slice(1, -1),
      ),
    );
    last = match.index + match[0].length;
  }
  nodes.push(text.slice(last));
  return nodes;
}

// Very small Markdown subset: paragraphs and "-"/"*" bullet lists. **bold** is unwrapped.
function renderRich(text, sources, panelId) {
  const frag = document.createDocumentFragment();
  const blocks = text.replace(/\*\*(.+?)\*\*/g, "$1").split(/\n{2,}/);
  for (const block of blocks) {
    const lines = block.split("\n").filter((l) => l.trim());
    if (!lines.length) continue;
    if (lines.every((l) => /^\s*([-*•]|\d+[.)])\s+/.test(l))) {
      frag.append(el("ul", {}, lines.map((l) => el("li", {}, inline(l.replace(/^\s*([-*•]|\d+[.)])\s+/, ""), sources, panelId)))));
    } else {
      frag.append(el("p", {}, inline(lines.join(" "), sources, panelId)));
    }
  }
  return frag;
}

// The Sources panel shows the sources of one selected answer at a time.
const sourceSets = new Map(); // panelId -> {sources, cited}

function showSources(panelId) {
  const set = sourceSets.get(panelId);
  const list = $("#sources-list");
  list.replaceChildren();
  document.querySelectorAll(".msg-bot").forEach((m) => (m.dataset.selected = String(m.id === panelId)));
  if (!set || !set.sources.length) {
    list.append(el("p", { class: "muted" }, "No passages were retrieved for this answer."));
    return;
  }
  for (const s of set.sources) {
    const pages = s.page_start === s.page_end ? `p. ${s.page_start}` : `pp. ${s.page_start}–${s.page_end}`;
    list.append(
      el(
        "article",
        { class: "source", id: `src-${s.n}`, tabindex: "-1", dataset: { cited: String(set.cited.includes(s.n)) } },
        el("div", { class: "source-head" }, el("span", {}, `${s.n}. ${s.citation}`), el("span", { class: "muted" }, pages)),
        el("div", { class: "source-title" }, s.header),
        el("p", { class: "source-snippet" }, s.snippet),
        el(
          "div",
          { class: "source-foot" },
          el("span", { class: "tag" }, s.chunk_type),
          el("span", { class: "tag" }, `score ${Number(s.score).toFixed(3)}`),
          set.cited.includes(s.n) ? el("span", { class: "tag" }, "cited") : null,
        ),
      ),
    );
  }
}

function highlightSource(panelId, n) {
  showSources(panelId);
  const card = document.getElementById(`src-${n}`);
  if (!card) return;
  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
  card.focus({ preventScroll: true });
  card.classList.add("flash");
  setTimeout(() => card.classList.remove("flash"), 1600);
}

document.addEventListener("click", (event) => {
  const chip = event.target.closest(".chip");
  if (chip) highlightSource(chip.dataset.panel, Number(chip.dataset.n));
  const bot = event.target.closest(".msg-bot");
  if (bot && !chip && sourceSets.has(bot.id)) showSources(bot.id);
});

// ---------------------------------------------------------------------------
// Ask (Server-Sent Events over a POST response)
// ---------------------------------------------------------------------------

let answerCount = 0;

// Parse "event: x\ndata: {...}\n\n" frames from a streamed fetch body.
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

function setBusy(busy) {
  state.busy = busy;
  $("#ask-btn").disabled = busy;
  $("#ask-btn").textContent = busy ? "Answering…" : "Ask";
}

async function ask(question) {
  if (state.busy) return;
  $("#chat-empty")?.remove();
  const chat = $("#chat");
  chat.append(el("div", { class: "msg msg-user" }, question));
  const id = `answer-${++answerCount}`;
  const bubble = el("div", { class: "msg msg-bot", id, "aria-busy": "true" }, el("span", { class: "typing", "aria-label": "Searching the thesis" }, el("span"), el("span"), el("span")));
  chat.append(bubble);
  bubble.scrollIntoView({ block: "end" });
  setBusy(true);

  let text = "";
  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(typeof detail.detail === "string" ? detail.detail : `Request failed (HTTP ${res.status})`);
    }
    for await (const { event, data } of readSSE(res)) {
      if (event === "token") {
        text += data.text;
        bubble.textContent = text; // plain text while streaming; chips are added at the end
      } else if (event === "answer") {
        renderAnswer(bubble, data);
      } else if (event === "error") {
        throw new Error(data.message);
      }
    }
  } catch (err) {
    bubble.classList.add("msg-error");
    bubble.replaceChildren(
      el("p", {}, `Something went wrong: ${err.message}`),
      el("button", { type: "button", class: "button button-quiet", "data-retry": question }, "Try again"),
    );
  } finally {
    bubble.removeAttribute("aria-busy");
    setBusy(false);
  }
}

function renderAnswer(bubble, answer) {
  sourceSets.set(bubble.id, { sources: answer.sources, cited: answer.cited });
  bubble.replaceChildren(renderRich(answer.answer, answer.sources, bubble.id));
  const meta = el("div", { class: "msg-meta" });
  if (answer.not_found) meta.append(el("span", { class: "tag" }, "not in the thesis"));
  else if (!answer.cited.length) meta.append(el("span", { class: "tag tag-warn", title: "The model did not cite a passage; check the sources." }, "no citation"));
  if (answer.broad) meta.append(el("span", { class: "tag", title: "Broad question: summaries were searched too" }, "overview"));
  if (answer.timings && answer.timings.total) meta.append(el("span", {}, `retrieval ${answer.timings.total.toFixed(1)} s`));
  bubble.append(meta);
  showSources(bubble.id);
}

document.addEventListener("click", (event) => {
  const retry = event.target.closest("[data-retry]");
  if (retry) ask(retry.dataset.retry);
  const example = event.target.closest(".example");
  if (example) ask(example.textContent.trim());
});

$("#ask-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const box = $("#question");
  const question = box.value.trim();
  if (!question) return;
  box.value = "";
  ask(question);
});

$("#question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $("#ask-form").requestSubmit();
  }
});

// ---------------------------------------------------------------------------
// Analyze
// ---------------------------------------------------------------------------

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
  dropzone.addEventListener(type, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag");
  });
}
for (const type of ["dragleave", "drop"]) dropzone.addEventListener(type, () => dropzone.classList.remove("drag"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  if (e.dataTransfer.files.length) {
    csvInput.files = e.dataTransfer.files;
    loadColumns(csvInput.files[0]);
  }
});

const fmt = (v) => (v === null || v === undefined ? "–" : typeof v === "number" ? String(v) : v);

function table(columns, rows) {
  return el(
    "div",
    { class: "table-wrap" },
    el(
      "table",
      {},
      el("thead", {}, el("tr", {}, columns.map(([label]) => el("th", { scope: "col" }, label)))),
      el("tbody", {}, rows.map((row) => el("tr", {}, columns.map(([, key, cls]) => {
        const value = typeof key === "function" ? key(row) : row[key];
        return el("td", { class: cls || (typeof value === "number" ? "num" : null) }, value instanceof Node ? value : fmt(value));
      })))),
    ),
  );
}

function renderAnalysis(result) {
  const out = $("#analyze-out");
  const s = result.stats;
  out.replaceChildren(el("div", { class: "banner", role: "note" }, result.label));
  if (result.note) out.append(el("p", { class: "muted" }, result.note));
  out.append(
    el("div", { class: "card" },
      el("h3", {}, `Descriptive statistics · ${s.rows} rows`),
      s.descriptive.length
        ? table([["Column", "column"], ["n", "n"], ["Missing", "missing"], ["Mean", "mean"], ["SD", "std"], ["Min", "min"], ["Median", "median"], ["Max", "max"]], s.descriptive)
        : el("p", { class: "muted" }, "No numeric columns found."),
    ),
  );
  if (s.comparisons.length) {
    out.append(
      el("div", { class: "card" },
        el("h3", {}, `Group comparisons by ${s.group_column}`),
        table(
          [
            ["Column", "column"],
            ["Test", "test"],
            ["p", "p_value"],
            ["p (Holm)", "p_adjusted"],
            ["Non-param. p", "nonparametric_p"],
            ["Effect", (r) => (r.cohens_d !== undefined ? `d = ${fmt(r.cohens_d)}` : `η² = ${fmt(r.eta_sq)}`)],
            ["", (r) => (r.significant ? el("span", { class: "sig" }, "significant") : "")],
            ["Group means", (r) => Object.entries(r.group_means).map(([g, m]) => `${g}: ${fmt(m)}`).join(" · ")],
          ],
          s.comparisons,
        ),
      ),
    );
  } else if (!s.group_column) {
    out.append(el("p", { class: "muted" }, "No categorical column to compare groups by; choose one above to add group comparisons."));
  }
  if (s.correlations.length) {
    out.append(el("div", { class: "card" }, el("h3", {}, "Correlations (|r| ≥ 0.3)"), table([["X", "x"], ["Y", "y"], ["r", "r"], ["p", "p_value"], ["n", "n"]], s.correlations)));
  }
  if (s.thresholds.length) {
    out.append(el("div", { class: "card" }, el("h3", {}, "Thesis thresholds applied"), table([["Rule", "rule"], ["Pass", (r) => `${r.passing} / ${r.n}`], ["Share", (r) => (r.passing_share === null ? "–" : `${Math.round(r.passing_share * 100)}%`)], ["Source", "citation"]], s.thresholds)));
  }
  if (result.explanation) {
    const id = `answer-${++answerCount}`;
    const card = el("div", { class: "card msg-bot", id }, el("h3", {}, "What this means, according to the thesis"));
    card.append(renderRich(result.explanation, result.sources, id));
    out.append(card);
    sourceSets.set(id, { sources: result.sources, cited: result.cited });
    showSources(id);
  }
}

$("#analyze-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = csvInput.files[0];
  if (!file) return;
  const body = new FormData();
  body.append("file", file);
  body.append("group_col", $("#group-col").value);
  body.append("explain", String($("#explain").checked));
  const button = $("#analyze-btn");
  button.disabled = true;
  const out = $("#analyze-out");
  out.replaceChildren(
    el("div", { class: "loading", role: "status" }, el("span", { class: "spinner" }),
      $("#explain").checked ? "Computing statistics, then asking the model to explain them…" : "Computing statistics…"),
  );
  try {
    const res = await fetch("/analyze", { method: "POST", body });
    const data = await res.json();
    if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : `HTTP ${res.status}`);
    renderAnalysis(data);
  } catch (err) {
    out.replaceChildren(el("div", { class: "card msg-error", role: "alert" }, `Analysis failed: ${err.message}`));
  } finally {
    button.disabled = false;
  }
});

refreshStatus();
