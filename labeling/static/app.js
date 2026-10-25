"use strict";

const $ = (sel) => document.querySelector(sel);
const params = new URLSearchParams(location.search);

async function api(method, path, body, headers) {
  const res = await fetch(path, {
    method,
    headers: body instanceof Blob ? headers : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : body instanceof Blob ? body : JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

function setStatus(text, isError) {
  const el = $("#status");
  el.textContent = text;
  el.className = isError ? "error" : "muted";
}

// ---------- home ----------

async function showHome() {
  $("#home").hidden = false;
  const home = await api("GET", "/api/home");
  const form = $("#new-session");
  form.axis.innerHTML = home.axes.map((a) => `<option>${a}</option>`).join("");
  const built = home.pools.filter((p) => p.built && p.pool !== "trap_pairs_v1");
  form.pool.innerHTML = built.map((p) => `<option value="${p.pool}">${p.pool} (${p.n})</option>`).join("");
  const syncSubsets = () => {
    const pool = built.find((p) => p.pool === form.pool.value);
    form.subset.innerHTML = '<option value="">all</option>' +
      (pool ? pool.subsets.map((s) => `<option>${s}</option>`).join("") : "");
  };
  const syncKind = () => {
    document.querySelectorAll("[data-for=axis]").forEach((el) => (el.hidden = form.kind.value !== "axis"));
    document.querySelectorAll("[data-for=pool]").forEach((el) => (el.hidden = form.kind.value === "trap"));
  };
  form.pool.onchange = syncSubsets;
  form.kind.onchange = syncKind;
  syncSubsets();
  syncKind();
  form.onsubmit = async (ev) => {
    ev.preventDefault();
    try {
      const { session_id } = await api("POST", "/api/sessions", {
        kind: form.kind.value, axis: form.axis.value, pool: form.pool.value, subset: form.subset.value,
      });
      location.search = `?session=${session_id}`;
    } catch (err) {
      $("#form-error").hidden = false;
      $("#form-error").textContent = err.message;
    }
  };
  $("#sessions tbody").innerHTML = home.sessions.slice().reverse().map((s) =>
    `<tr><td><a href="?session=${s.session_id}">${s.session_id}</a></td>` +
    `<td>${s.kind === "axis" ? s.axis : s.kind}</td><td>${s.pool}${s.subset ? " / " + s.subset : ""}</td>` +
    `<td>${s.n_done} / ${s.n_items}</td></tr>`).join("");
}

// ---------- labeling ----------

const S = {
  payload: null, cards: [], idx: 0, shownAt: 0, effective: {}, notes: {}, kind: "axis",
  keyToValue: {}, showStratum: false,
};

async function showSession(sessionId) {
  $("#label").hidden = false;
  S.payload = await api("GET", `/api/session/${sessionId}`);
  S.cards = S.payload.cards;
  S.effective = S.payload.effective;
  S.kind = S.payload.session.kind;
  S.idx = Math.min(S.payload.cursor, S.cards.length);
  S.sessionId = sessionId;
  renderRubric();
  render();
  document.addEventListener("keydown", onKeyDown);
  document.addEventListener("keyup", onKeyUp);
}

function renderRubric() {
  const spec = S.payload.axis_spec;
  if (S.kind === "narrate") {
    $("#axis-name").textContent = "Free narration";
    $("#definition").textContent = "Hold space and talk about the post: what you'd score, what makes it hard, any distinction the rubric misses.";
    $("#help").textContent = "space (hold) record · → next · ← back · u undo last step";
    return;
  }
  $("#axis-name").textContent = S.kind === "trap" ? "Collision-trap pairs" : spec.axis;
  $("#definition").textContent = spec.definition;
  $("#keys").innerHTML = spec.keys.map((k) => `<li data-value="${k.value}"><kbd>${k.key}</kbd>${escapeHtml(k.text)}</li>`).join("");
  $("#rules").innerHTML = spec.boundary_rules.map((r) => `<li>${escapeHtml(r)}</li>`).join("");
  S.keyToValue = Object.fromEntries(spec.keys.map((k) => [k.key, k.value]));
  $("#help").textContent = "keys label & advance · ? can't tell · u undo · ← → navigate · space (hold) voice note · s stratum (after labeling)";
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function renderCtx(el, block) {
  el.hidden = !block;
  if (!block) return;
  el.classList.toggle("missing", !block.available);
  el.querySelector(".ctx-text").textContent = block.available
    ? block.text + (block.truncated ? " …" : "")
    : "(not available: deleted, hidden or outside the export window)";
}

function mediaLine(m) {
  const bits = [];
  if (m.image_count) bits.push(`[${m.image_count} image${m.image_count > 1 ? "s" : ""} — alt text not in export]`);
  if (m.has_video) bits.push("[video]");
  if (m.external_host) bits.push(`[link: ${m.external_host}]`);
  if (!bits.length && m.embed_type && !["record", ""].includes(m.embed_type)) bits.push(`[embed: ${m.embed_type}]`);
  return bits.join(" ");
}

function render() {
  const done = Object.keys(S.effective).length;
  $("#progress").textContent = `${Math.min(S.idx + 1, S.cards.length)} / ${S.cards.length} · ${done} done`;
  if (S.idx >= S.cards.length) {
    $("#card").innerHTML = `<p class="post-text">Session complete: ${done} of ${S.cards.length} items.</p><p class="muted">← to review, u to undo.</p>`;
    $("#card").dataset.done = "1";
    return;
  }
  if ($("#card").dataset.done) location.reload();
  const card = S.cards[S.idx];
  renderCtx($("#ctx-parent"), card.reply_parent);
  renderCtx($("#ctx-quote"), card.quoted);
  $("#post-text").textContent = card.text;
  $("#media").textContent = mediaLine(card.media);
  $("#post-type").textContent = card.post_type;
  const value = S.effective[card.key];
  const labeled = value !== undefined;
  $("#card").classList.toggle("labeled", labeled);
  $("#current-value").textContent = labeled ? (S.kind === "narrate" ? "narrated" : `labeled: ${value}`) : "";
  $("#stratum").hidden = !(labeled && S.showStratum);
  $("#stratum").textContent = `stratum: ${card.stratum}`;
  const opts = $("#options");
  opts.hidden = !card.options;
  if (card.options) {
    opts.innerHTML = card.options.map((o, i) =>
      `<div class="${value === i + 1 ? "chosen" : ""}"><b>${i + 1}</b>${escapeHtml(o)}</div>`).join("");
  }
  document.querySelectorAll("#keys li").forEach((li) => li.classList.toggle("chosen", labeled && li.dataset.value === String(value)));
  S.shownAt = performance.now();
}

function go(idx) {
  S.idx = Math.max(0, Math.min(idx, S.cards.length));
  render();
}

// Saves go through one ordered queue so fast keypresses are never dropped and undo
// always applies after every label typed before it.
let queue = Promise.resolve();
let pending = 0;
function enqueue(task) {
  pending += 1;
  setStatus(`saving… (${pending})`);
  queue = queue.then(task).finally(() => {
    pending -= 1;
    if (pending === 0 && $("#status").className !== "error") setStatus("saved");
  });
}

function commit(value) {
  const card = S.cards[S.idx];
  const ms = Math.round(performance.now() - S.shownAt);
  const noteIds = S.notes[card.key] || [];
  const previous = S.effective[card.key];
  S.effective[card.key] = S.kind === "narrate" ? null : value;
  go(S.idx + 1);
  enqueue(async () => {
    try {
      if (S.kind === "narrate") {
        await api("POST", `/api/session/${S.sessionId}/view`, { key: card.key, ms_on_card: ms, note_ids: noteIds });
      } else {
        await api("POST", `/api/session/${S.sessionId}/label`, { key: card.key, value, ms_to_decide: ms, note_ids: noteIds });
      }
    } catch (err) {
      if (previous === undefined) delete S.effective[card.key]; else S.effective[card.key] = previous;
      setStatus(`NOT SAVED (${err.message}); returned to that card`, true);
      go(S.cards.indexOf(card));
    }
  });
}

function undo() {
  enqueue(async () => {
    try {
      const res = await api("POST", `/api/session/${S.sessionId}/undo`);
      if (!res.row) { setStatus("nothing to undo"); return; }
      S.effective = res.effective;
      go(res.row.position);
    } catch (err) {
      setStatus(`undo failed: ${err.message}`, true);
    }
  });
}

function onKeyDown(ev) {
  if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
  if (ev.key === " ") { ev.preventDefault(); if (!ev.repeat) startRecording(); return; }
  if (ev.repeat) return;
  if (ev.key === "ArrowLeft") { ev.preventDefault(); go(S.idx - 1); return; }
  if (ev.key === "u") { undo(); return; }
  if (S.idx >= S.cards.length) return;
  if (ev.key === "ArrowRight") {
    ev.preventDefault();
    if (S.kind === "narrate") commit(null); else go(S.idx + 1);
    return;
  }
  if (ev.key === "s") { S.showStratum = !S.showStratum; render(); return; }
  if (S.kind !== "narrate" && Object.hasOwn(S.keyToValue, ev.key)) commit(S.keyToValue[ev.key]);
}

function onKeyUp(ev) {
  if (ev.key === " ") { ev.preventDefault(); stopRecording(); }
}

// ---------- push-to-talk ----------

const R = { stream: null, recorder: null, chunks: [], startedAt: 0, key: null };

async function startRecording() {
  if (R.recorder) return;
  try {
    R.stream = R.stream || await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    setStatus(`microphone unavailable: ${err.message}`, true);
    return;
  }
  if (S.idx >= S.cards.length) return;
  R.key = S.cards[S.idx].key;
  R.chunks = [];
  R.recorder = new MediaRecorder(R.stream);
  R.recorder.ondataavailable = (e) => e.data.size && R.chunks.push(e.data);
  R.recorder.start();
  R.startedAt = performance.now();
  $("#rec").hidden = false;
}

function stopRecording() {
  const rec = R.recorder;
  if (!rec) return;
  R.recorder = null;
  $("#rec").hidden = true;
  const key = R.key;
  const durationMs = Math.round(performance.now() - R.startedAt);
  rec.onstop = async () => {
    const blob = new Blob(R.chunks, { type: rec.mimeType });
    if (durationMs < 300) { setStatus("note too short, discarded"); return; }
    try {
      const q = new URLSearchParams({ key, duration_ms: durationMs });
      const { note_id } = await api("POST", `/api/session/${S.sessionId}/note?${q}`, blob, { "Content-Type": rec.mimeType });
      (S.notes[key] = S.notes[key] || []).push(note_id);
      setStatus(`note saved (${(durationMs / 1000).toFixed(1)} s)`);
    } catch (err) {
      setStatus(`note not saved: ${err.message}`, true);
    }
  };
  rec.stop();
}

if (params.get("session")) showSession(params.get("session")).catch((e) => setStatus(e.message, true));
else showHome().catch((e) => { $("#home").hidden = false; $("#form-error").hidden = false; $("#form-error").textContent = e.message; });
