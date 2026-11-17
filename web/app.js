// app.js - the dashboard. It only DISPLAYS: every number comes from the Python
// Mesa model running in worker.js. No plant logic is written in JavaScript.

// ---------------------------------------------------------------- Python link
const params = new URLSearchParams(location.search);
const worker = new Worker("worker.js");
let nextId = 1;
const pending = new Map();

worker.onmessage = (e) => {
  const d = e.data;
  if (d.progress) return markLoad(d.progress);
  const p = pending.get(d.id);
  if (!p) return;
  pending.delete(d.id);
  d.error ? p.reject(new Error(d.error)) : p.resolve(d.result);
};

function call(fn, ...args) {
  return new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, fn, args, base: params.get("pyodide") || undefined });
  }).then((txt) => JSON.parse(txt));
}

// ---------------------------------------------------------------- state
const $ = (id) => document.getElementById(id);
const S = {
  opts: null, hours: [], snap: null, running: false, busy: false, total: 0,
  effParam: "BOD", infParam: "flow",
};

const EFF_PARAMS = ["BOD", "COD", "TSS", "TN", "FC", "pH"];
const INF_PARAMS = ["flow", "BOD", "COD", "TSS", "NH4", "pH"];
const PRETTY = {
  industrial_discharge: "Industrial discharge", acid_dump: "Acid dump", alkaline_dump: "Alkaline dump",
  storm_inflow: "Storm inflow", septage_dumping: "Septage dumping", high_strength_food: "Food-processing waste",
  unusual_load: "Unusual load", normal: "Normal sewage", unknown: "Unlabelled",
};
const pretty = (k) => PRETTY[k] || k;
const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const fmtFC = (v) => (v >= 1000 ? Number(v).toExponential(1).replace("e+", "×10^") : fmt(v, 0));
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// ---------------------------------------------------------------- loading
const LOAD_ORDER = ["py", "pkgs", "mesa", "code", "train", "ready"];
function markLoad(step) {
  const idx = LOAD_ORDER.indexOf(step);
  document.querySelectorAll("#loadSteps li").forEach((li) => {
    const i = LOAD_ORDER.indexOf(li.dataset.step);
    li.classList.toggle("done", i < idx);
    li.classList.toggle("active", i === idx);
  });
}

async function start() {
  markLoad("py");
  try {
    await call("__boot");
    S.opts = await call("options");
  } catch (err) {
    $("loadError").textContent = "Could not start Python: " + err.message;
    $("loadError").classList.remove("hidden");
    return;
  }
  fillSelect($("mode"), S.opts.modes);
  fillSelect($("norms"), Object.fromEntries(Object.entries(S.opts.norms).map(([k, v]) => [k, v.label])));
  fillSelect($("injectSel"), Object.fromEntries(Object.entries(S.opts.events).map(([k]) => [k, pretty(k)])));
  makeTabs("effTabs", EFF_PARAMS, "effParam");
  makeTabs("infTabs", INF_PARAMS, "infParam");
  buildPlant();
  await resetModel();
  $("loader").classList.add("hidden");
  ["btnRun", "btnStep", "btnDay", "btnReset", "btnInject", "btnEval"].forEach((id) => ($(id).disabled = false));
}

function fillSelect(sel, obj) {
  sel.innerHTML = Object.entries(obj).map(([k, v]) => `<option value="${k}">${v}</option>`).join("");
}

function makeTabs(id, list, key) {
  const box = $(id);
  box.innerHTML = list.map((p) => `<button data-p="${p}" class="${p === S[key] ? "on" : ""}">${p}</button>`).join("");
  box.onclick = (e) => {
    const p = e.target.dataset.p;
    if (!p) return;
    S[key] = p;
    box.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.p === p));
    drawCharts();
  };
}

// ---------------------------------------------------------------- running
async function resetModel() {
  stop();
  const info = await call("reset", $("mode").value, $("norms").value);
  S.total = info.total_hours;
  S.hours = [];
  S.snap = null;
  $("dataLabel").textContent = "Data: " + info.data_label;
  $("msgLog").innerHTML = "";
  $("evalOut").innerHTML = "";
  await advance(1);
}

async function advance(n) {
  if (S.busy) return;
  S.busy = true;
  try {
    const r = await call("advance", n);
    S.hours.push(...r.hours);
    if (r.snapshot) S.snap = r.snapshot;
    r.hours.forEach((h) => addMessages(h));
    render();
    if (r.finished) {
      stop();
      $("btnRun").textContent = "End of data";
      $("btnRun").disabled = true;
    }
  } catch (err) {
    stop();
    alert("Python error: " + err.message);
  } finally {
    S.busy = false;
  }
}

function loop() {
  if (!S.running) return;
  advance(Number($("speed").value)).then(() => {
    if (S.running) S.timer = setTimeout(loop, 120);
  });
}
function play() { S.running = true; $("btnRun").textContent = "❚❚ Pause"; loop(); }
function stop() { S.running = false; clearTimeout(S.timer); $("btnRun").textContent = "▶ Run"; $("btnRun").disabled = false; }

$("btnRun").onclick = () => (S.running ? stop() : play());
$("btnStep").onclick = () => advance(1);
$("btnDay").onclick = () => advance(24);
$("btnReset").onclick = resetModel;
$("mode").onchange = resetModel;
$("norms").onchange = resetModel;
$("btnInject").onclick = async () => {
  const ev = $("injectSel").value;
  await call("inject", ev, 4);
  addMessageRow({ t: S.snap ? S.snap.time.slice(0, 16) : "", sender: "You", kind: "INJECT",
    text: `${pretty(ev)} injected into the next 4 hours of data` });
};
$("dataSel").onchange = async () => {
  if ($("dataSel").value === "upload") {
    $("fileInput").click();
  } else {
    await call("use_demo");
    await resetModel();
  }
};
$("fileInput").onchange = async () => {
  const f = $("fileInput").files[0];
  if (!f) { $("dataSel").value = "demo"; return; }
  try {
    await call("use_csv", await f.text(), f.name);
    await resetModel();
  } catch (err) {
    alert("Could not read that CSV:\n" + err.message.split("\n").slice(-2).join("\n"));
    $("dataSel").value = "demo";
  }
  $("fileInput").value = "";
};
$("btnEval").onclick = async () => {
  stop();
  $("btnEval").disabled = true;
  $("evalOut").innerHTML = `<p class="muted">Running the whole dataset in all three modes… (the Python is busy, the page stays usable)</p>`;
  try {
    renderEval(await call("evaluate", $("norms").value));
  } catch (err) {
    $("evalOut").innerHTML = `<p class="error">${err.message}</p>`;
  }
  $("btnEval").disabled = false;
};

// ---------------------------------------------------------------- render
function render() {
  const s = S.snap;
  if (!s) return;
  $("simTime").textContent = s.time.slice(0, 16).replace("T", " ");
  $("simHour").textContent = `hour ${s.hour} of ${s.total_hours}`;
  const ev = $("eventBadge");
  ev.textContent = "Data label: " + pretty(s.event);
  ev.className = "badge " + (s.event === "normal" ? "ok" : s.event === "storm_inflow" ? "warn" : "bad");
  renderPlant(s);
  renderCompliance(s);
  renderKpis(s.summary);
  renderDetection(s);
  drawCharts();
}

function renderCompliance(s) {
  const limits = S.opts.norms[s.norms].limits;
  const eff = s.streams.effluent;
  const c = s.compliance;
  $("passBadge").textContent = c.passed ? "MEETS NORMS" : "FAILS: " + c.failing.join(", ");
  $("passBadge").className = "badge " + (c.passed ? "ok" : "bad");
  let rows = `<tr><th>Parameter</th><th class="num">Now</th><th class="num">Limit</th><th></th></tr>`;
  for (const [p, [lo, hi]] of Object.entries(limits)) {
    const v = p === "TN" ? eff.TN : eff[p];
    const ok = (lo === null || v >= lo) && (hi === null || v <= hi);
    const lim = lo !== null && hi !== null ? `${lo}–${hi}` : `≤ ${hi}`;
    const unit = S.opts.units[p] || "";
    rows += `<tr><td>${p} <span class="muted small">${unit}</span></td><td class="num">${p === "FC" ? fmtFC(v) : fmt(v, p === "pH" ? 2 : 1)}</td>
      <td class="num">${lim}</td><td class="${ok ? "pass" : "fail"}">${ok ? "✓" : "✗"}</td></tr>`;
  }
  $("complianceTable").innerHTML = rows;
  $("discharge").textContent = c.discharged_m3 > 0
    ? `${fmt(c.discharged_m3, 0)} m³ discharged this hour` + (c.recirculated_m3 > 0 ? `, ${fmt(c.recirculated_m3, 0)} m³ sent back for re-treatment` : "")
    : `Nothing discharged this hour (${fmt(c.recirculated_m3, 0)} m³ recirculated)`;
}

function renderKpis(sm) {
  const v = sm.violation_hours_by_parameter;
  const worst = Object.keys(v).length ? Object.entries(v).sort((a, b) => b[1] - a[1])[0] : null;
  const items = [
    [sm.compliant_hours_pct + "%", "hours meeting norms"],
    [sm.compliant_volume_pct + "%", "volume meeting norms"],
    [sm.energy_kwh.toLocaleString() + " kWh", "blower energy"],
    [sm.chlorine_kg.toLocaleString() + " kg", "chlorine used"],
    [sm.diverted_m3.toLocaleString() + " m³", "shock load diverted"],
    [sm.overflow_m3.toLocaleString() + " m³", "raw sewage overflow"],
    [sm.alarms, "alarms raised"],
    [worst ? `${worst[0]} (${worst[1]} h)` : "none", "most-failed parameter"],
  ];
  $("kpis").innerHTML = items.map(([val, l]) => `<div class="kpi"><div class="v">${val}</div><div class="l">${l}</div></div>`).join("");
}

function renderDetection(s) {
  const d = s.detection;
  const f = s.forecast;
  const raw = s.streams.raw;
  const score = d.ml_score;
  const badge = d.alarm ? `<span class="badge bad">ALARM – ${pretty(d.type)}</span>` : `<span class="badge ok">No alarm</span>`;
  const rules = d.rules.length ? `<ul class="rules-list small">${d.rules.map((r) => `<li>${r}</li>`).join("")}</ul>` : `<p class="small muted">No rule broken</p>`;
  const fc = f ? `<p class="small">Forecast for this hour (${f.source}): flow <b>${fmt(f.flow, 0)}</b> m³/h · actual <b>${fmt(raw.flow, 0)}</b></p>` : "";
  $("detection").innerHTML = `
    <p>${badge}</p>
    <p class="small"><b>Rule check</b>${d.rules_alarm ? " – alarm" : ""}</p>${rules}
    <p class="small"><b>ML (Isolation Forest)</b> score ${fmt(score, 2)} ${d.ml_alarm ? "– alarm" : ""}
      ${s.mode !== "ai" ? '<span class="muted">(not used in this mode)</span>' : ""}</p>
    <div class="meter" title="0.5 = alarm threshold"><i style="width:${Math.min(100, score * 100)}%;background:${d.ml_alarm ? css("--bad") : css("--water")}"></i><b></b></div>
    <p class="small muted">Line = alarm threshold. Truth from the data: <b>${pretty(d.true_event)}</b></p>
    ${fc}`;
}

function addMessages(h) {
  h.msgs.forEach((m) => addMessageRow({ t: h.t, sender: m.sender, kind: m.kind, text: m.text }));
}
function addMessageRow(m) {
  const li = document.createElement("li");
  li.innerHTML = `<span class="t">${m.t.slice(5)}</span> <span class="who">${m.sender}</span><span class="kind">${m.kind}</span><br>${m.text}`;
  const log = $("msgLog");
  log.prepend(li);
  while (log.children.length > 250) log.lastChild.remove();
}

// ---------------------------------------------------------------- plant diagram
const UNITS = [
  { id: "raw", title: "Inlet", x: 10 },
  { id: "preliminary", title: "Screen & Grit", x: 135 },
  { id: "equalization", title: "Equalization", x: 260, tank: true },
  { id: "primary", title: "Primary clar.", x: 385 },
  { id: "aeration", title: "Aeration", x: 510 },
  { id: "secondary", title: "Secondary clar.", x: 635 },
  { id: "filter", title: "Filter", x: 760 },
  { id: "chlorination", title: "Chlorination", x: 885 },
  { id: "effluent", title: "Outlet → river", x: 1010 },
];
const BW = 104, BY = 30, BH = 128;

function buildPlant() {
  const svg = $("plantSvg");
  let h = "";
  // main pipes between boxes
  for (let i = 0; i < UNITS.length - 1; i++) {
    const a = UNITS[i], b = UNITS[i + 1];
    h += `<path class="pipe" d="M${a.x + BW} ${BY + 64} L${b.x} ${BY + 64}"/>`;
  }
  // divert pipe: screen -> holding; release: holding -> EQ; recirculation: outlet -> EQ
  h += `<path id="pDivert" class="pipe hold off" d="M${135 + BW / 2} ${BY + BH} L${135 + BW / 2} 255 L260 255"/>`;
  h += `<text class="flow-label" x="140" y="276">divert shock</text>`;
  h += `<path id="pRelease" class="pipe hold off" d="M${280} 215 L${280} ${BY + BH}"/>`;
  h += `<path id="pRecirc" class="pipe dashed off" d="M${1010 + BW / 2} ${BY + BH} L${1010 + BW / 2} 190 L${350} 190 L${350} ${BY + BH}"/>`;
  h += `<text class="flow-label" x="640" y="184">recirculate failed water</text>`;
  // holding tank
  h += `<rect id="b-hold" class="u-box" x="260" y="215" width="${BW}" height="60" rx="8"/>
        <rect id="f-hold" class="tank-fill hold" x="261" y="274" width="${BW - 2}" height="0"/>
        <text class="u-title" x="268" y="234">Holding tank</text>
        <text id="t-hold" class="u-line" x="268" y="252">0% full</text>
        <text id="t-hold2" class="u-line" x="268" y="267"></text>`;
  for (const u of UNITS) {
    h += `<rect id="b-${u.id}" class="u-box" x="${u.x}" y="${BY}" width="${BW}" height="${BH}" rx="8"/>`;
    if (u.tank) h += `<rect id="f-${u.id}" class="tank-fill" x="${u.x + 1}" y="${BY + BH - 1}" width="${BW - 2}" height="0" rx="6"/>`;
    h += `<text class="u-title" x="${u.x + 8}" y="${BY + 20}">${u.title}</text>`;
    for (let k = 0; k < 5; k++) h += `<text id="t-${u.id}-${k}" class="u-line" x="${u.x + 8}" y="${BY + 42 + k * 18}"></text>`;
  }
  h += `<rect class="bar-bg" x="${510 + 8}" y="${BY + 119}" width="${BW - 16}" height="7" rx="3"/>
        <rect id="healthBar" class="bar-fg" x="${510 + 8}" y="${BY + 119}" width="0" height="7" rx="3"/>
        <rect class="bar-bg" x="${760 + 8}" y="${BY + 119}" width="${BW - 16}" height="7" rx="3"/>
        <rect id="clogBar" class="bar-fg" x="${760 + 8}" y="${BY + 119}" width="0" height="7" rx="3"/>`;
  svg.innerHTML = h;
}

function setLines(id, lines) {
  for (let k = 0; k < 5; k++) {
    const el = $(`t-${id}-${k}`);
    const ln = lines[k];
    el.textContent = ln ? (Array.isArray(ln) ? ln[0] : ln) : "";
    el.setAttribute("class", "u-line" + (Array.isArray(ln) && ln[1] ? " strong" : ""));
  }
}
function setStatus(id, st) { $(`b-${id}`).setAttribute("class", "u-box " + st); }
function setPipe(id, on) { const el = $(id); el.classList.toggle("off", !on); }

function renderPlant(s) {
  const w = s.streams, u = s.units, d = s.detection, c = s.compliance;
  const eq = u.equalization, aer = u.aeration, fil = u.filter, chl = u.chlorination;

  setLines("raw", [[`${fmt(w.raw.flow, 0)} m³/h`, true], `BOD ${fmt(w.raw.BOD, 0)}`, `COD ${fmt(w.raw.COD, 0)}`, `pH ${fmt(w.raw.pH, 1)}`, `NH₄ ${fmt(w.raw.NH4, 1)}`]);
  setStatus("raw", d.alarm ? (d.type === "storm_inflow" ? "warn" : "bad") : "ok");

  setLines("preliminary", [`TSS ${fmt(w.preliminary.TSS, 0)}`, `BOD ${fmt(w.preliminary.BOD, 0)}`, eq.diverted > 0 ? [`diverting ${fmt(eq.diverted, 0)} m³`, true] : ""]);
  setStatus("preliminary", eq.diverted > 0 ? "warn" : "ok");

  setLines("equalization", [[`level ${fmt(eq.level_pct, 0)}%`, true], `out ${fmt(eq.outflow, 0)} m³/h`, `pH ${fmt(w.equalization.pH, 2)}`,
    eq.ph_shift ? [`pH dosing ${eq.ph_shift > 0 ? "+" : ""}${fmt(eq.ph_shift, 1)}`, true] : "", eq.overflow > 0 ? [`OVERFLOW ${fmt(eq.overflow, 0)} m³`, true] : ""]);
  setStatus("equalization", eq.overflow > 0 ? "bad" : eq.level_pct > 85 ? "warn" : "ok");
  const fh = (BH - 2) * eq.level_pct / 100;
  $("f-equalization").setAttribute("y", BY + BH - 1 - fh);
  $("f-equalization").setAttribute("height", fh);

  $("t-hold").textContent = `${fmt(eq.holding_pct, 0)}% full`;
  $("t-hold2").textContent = eq.released > 0 ? `releasing ${fmt(eq.released, 0)} m³/h` : "";
  const hh = 58 * eq.holding_pct / 100;
  $("f-hold").setAttribute("y", 274 - hh);
  $("f-hold").setAttribute("height", hh);
  setStatus("hold", eq.holding_pct > 90 ? "bad" : eq.holding_pct > 0 ? "warn" : "ok");
  setPipe("pDivert", eq.diverted > 0);
  setPipe("pRelease", eq.released > 0);
  setPipe("pRecirc", c.recirculated_m3 > 0);

  setLines("primary", [`TSS ${fmt(w.primary.TSS, 0)}`, `BOD ${fmt(w.primary.BOD, 0)}`, `removed ${fmt(u.primary.tss_removed_pct, 0)}% TSS`]);
  setStatus("primary", u.primary.tss_removed_pct < 50 ? "warn" : "ok");

  setLines("aeration", [[`DO ${fmt(aer.DO, 2)} mg/L`, true], `blowers ${fmt(aer.blower_pct, 0)}%`, `BOD ${fmt(w.aeration.BOD, 0)} · NH₄ ${fmt(w.aeration.NH4, 1)}`, `recycle ${fmt(aer.internal_recycle, 1)}×`, `bacteria health`]);
  setStatus("aeration", aer.health < 0.5 ? "bad" : aer.health < 0.85 || aer.DO < 0.8 ? "warn" : "ok");
  $("healthBar").setAttribute("width", (BW - 16) * aer.health);
  $("healthBar").style.fill = aer.health < 0.5 ? css("--bad") : aer.health < 0.85 ? css("--warn") : css("--ok");

  setLines("secondary", [`TSS ${fmt(w.secondary.TSS, 1)}`, `BOD ${fmt(w.secondary.BOD, 1)}`, `COD ${fmt(w.secondary.COD, 0)}`]);
  setStatus("secondary", w.secondary.TSS > 40 ? "bad" : w.secondary.TSS > 20 ? "warn" : "ok");

  setLines("filter", [`TSS ${fmt(w.filter.TSS, 1)}`, `clogged ${fmt(fil.clog_pct, 0)}%`, fil.backwash ? ["backwashing", true] : "", "", "media clogging"]);
  setStatus("filter", fil.clog_pct > 85 ? "warn" : "ok");
  $("clogBar").setAttribute("width", (BW - 16) * fil.clog_pct / 100);
  $("clogBar").style.fill = fil.clog_pct > 85 ? css("--warn") : css("--water");

  setLines("chlorination", [[`dose ${fmt(chl.dose, 1)} mg/L`, true], `FC ${fmtFC(w.chlorination.FC)}`, `residual ${fmt(chl.residual_chlorine, 1)}`]);
  setStatus("chlorination", w.chlorination.FC > 230 ? "warn" : "ok");

  setLines("effluent", [[c.passed ? "MEETS NORMS" : "FAILS", true], `BOD ${fmt(w.effluent.BOD, 1)}`, `COD ${fmt(w.effluent.COD, 0)}`, `TN ${fmt(w.effluent.TN, 1)}`, `TSS ${fmt(w.effluent.TSS, 1)}`]);
  setStatus("effluent", c.passed ? "ok" : "bad");
}

// ---------------------------------------------------------------- charts
function lineChart(canvas, { series, limits = [], log = false, shade = null, truth = null, fixedMin = null, fixedMax = null, hideAxis = false }) {
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = Number(canvas.getAttribute("height"));
  canvas.width = W * dpr; canvas.height = H * dpr; canvas.style.height = H + "px";
  const g = canvas.getContext("2d");
  g.scale(dpr, dpr);
  g.clearRect(0, 0, W, H);
  const L = 46, R = 10, T = 8, B = 22;
  const n = Math.max(S.total || 1, 1);
  const tf = (v) => (log ? Math.log10(Math.max(v, 1)) : v);
  let vals = series.flatMap((s) => s.data.filter((v) => v !== null).map(tf));
  limits.forEach((l) => vals.push(tf(l.value)));
  let lo = fixedMin !== null ? fixedMin : Math.min(...vals, log ? 0 : 0);
  let hi = fixedMax !== null ? fixedMax : Math.max(...vals, 1);
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) { lo = 0; hi = 1; }
  if (hi - lo < 1e-9) hi = lo + 1;
  hi += (hi - lo) * 0.06;
  const x = (i) => L + (W - L - R) * (i / (n - 1 || 1));
  const y = (v) => T + (H - T - B) * (1 - (tf(v) - lo) / (hi - lo));
  const yRaw = (v) => T + (H - T - B) * (1 - (v - lo) / (hi - lo));
  const muted = css("--muted"), line = css("--line");

  // shading: hours where an alarm was raised, and a strip for the true event label
  if (shade) {
    g.fillStyle = "rgba(194,54,47,0.12)";
    shade.forEach((on, i) => { if (on) g.fillRect(x(i) - (W - L - R) / n / 2, T, Math.max((W - L - R) / n, 1), H - T - B); });
  }
  if (truth) {
    truth.forEach((ev, i) => {
      if (ev && ev !== "normal" && ev !== "unknown") {
        g.fillStyle = ev === "storm_inflow" ? css("--water") : css("--warn");
        g.fillRect(x(i) - (W - L - R) / n / 2, H - B + 2, Math.max((W - L - R) / n, 1), 5);
      }
    });
  }
  // grid + axis labels
  g.strokeStyle = line; g.lineWidth = 1; g.fillStyle = muted; g.font = "11px system-ui, sans-serif";
  for (let k = 0; k <= 4; k++) {
    const v = lo + (hi - lo) * k / 4;
    const yy = yRaw(v);
    g.beginPath(); g.moveTo(L, yy); g.lineTo(W - R, yy); g.stroke();
    if (!hideAxis) g.fillText(log ? "10^" + v.toFixed(1) : v >= 100 ? v.toFixed(0) : v.toFixed(1), 4, yy + 4);
  }
  // day ticks
  S.hours.forEach((h, i) => {
    if (h.t.endsWith("00:00") && (n <= 240 || Number(h.t.slice(8, 10)) % 3 === 1)) {
      g.fillText(h.t.slice(5, 10), x(i) - 14, H - 4);
    }
  });
  // limit lines
  limits.forEach((l) => {
    g.strokeStyle = css("--bad"); g.setLineDash([5, 4]); g.beginPath();
    g.moveTo(L, y(l.value)); g.lineTo(W - R, y(l.value)); g.stroke(); g.setLineDash([]);
    g.fillStyle = css("--bad"); g.fillText(l.label, W - R - g.measureText(l.label).width - 2, y(l.value) - 4);
  });
  // series
  series.forEach((s) => {
    g.strokeStyle = s.color; g.lineWidth = s.width || 1.6; g.setLineDash(s.dash || []);
    g.beginPath();
    let started = false;
    s.data.forEach((v, i) => {
      if (v === null) return;
      const px = x(i), py = y(v);
      started ? g.lineTo(px, py) : g.moveTo(px, py);
      started = true;
    });
    g.stroke(); g.setLineDash([]);
  });
}

function drawCharts() {
  if (!S.hours.length || !S.snap) return;
  const limits = S.opts.norms[S.snap.norms].limits;
  const p = S.effParam;
  const lim = [];
  if (limits[p]) {
    const [lo, hi] = limits[p];
    if (hi !== null) lim.push({ value: hi, label: `limit ${hi}` });
    if (lo !== null) lim.push({ value: lo, label: `min ${lo}` });
  }
  lineChart($("chartEff"), {
    series: [{ data: S.hours.map((h) => h.eff[p]), color: css("--accent"), width: 1.8 }],
    limits: lim, log: p === "FC",
  });
  const ip = S.infParam;
  lineChart($("chartInf"), {
    series: [{ data: S.hours.map((h) => h.inf[ip]), color: css("--water") }],
    shade: S.hours.map((h) => h.alarm), truth: S.hours.map((h) => h.event),
  });
  lineChart($("chartAer"), {
    series: [
      { data: S.hours.map((h) => h.DO / 8), color: css("--water"), width: 1.8 },
      { data: S.hours.map((h) => h.blower / 100), color: css("--muted"), dash: [4, 3] },
      { data: S.hours.map((h) => h.health), color: css("--ok"), width: 2 },
    ],
    fixedMin: 0, fixedMax: 1, hideAxis: true,
  });
  $("aerLegend").innerHTML = `<span style="color:${css("--water")}">━ DO (0–8 mg/L)</span>
    <span>┅ blowers (0–100%)</span><span style="color:${css("--ok")}">━ bacteria health (0–100%)</span>
    <span>· orange/blue strip under the influent chart = true event in the data</span>`;
}
window.addEventListener("resize", () => drawCharts());

// ---------------------------------------------------------------- evaluation
function table(rows, cols, headers) {
  return `<table><tr>${headers.map((h) => `<th>${h}</th>`).join("")}</tr>${rows
    .map((r) => `<tr>${cols.map((c) => `<td>${typeof c === "function" ? c(r) : r[c]}</td>`).join("")}</tr>`).join("")}</table>`;
}
function renderEval(r) {
  const modes = table(r.modes,
    [(x) => `<b>${x.mode}</b>`, "hours_meeting_norms_pct", "volume_meeting_norms_pct", "violation_hours", "worst_parameter",
      (x) => x.energy_kwh.toLocaleString(), (x) => x.chlorine_kg.toLocaleString(), (x) => x.diverted_m3.toLocaleString(), (x) => x.overflow_m3.toLocaleString()],
    ["Mode", "% hours OK", "% volume OK", "Bad hours", "Worst param", "Energy kWh", "Chlorine kg", "Diverted m³", "Overflow m³"]);
  const det = table(r.detection, [(x) => pretty(x.true_event), "rules_caught_pct", "ml_caught_pct", "ai_mode_caught_pct"],
    ["Event in data", "Rules caught %", "ML caught %", "AI mode (both) %"]);
  const sc = table(Object.entries(r.detector_scores).map(([k, v]) => ({ k, ...v })),
    [(x) => (x.k === "used_by_plant" ? "AI mode (rules OR ML)" : x.k), "precision", "recall", "f1", "false_alarms", "missed"],
    ["Detector", "Precision", "Recall", "F1", "False alarms", "Missed hours"]);
  const fc = table(r.forecast, ["quantity", "ml_forecast_mae", "last_hour_mae", (x) => `${x.improvement_pct}%`],
    ["Predicting", "ML error (MAE)", "'Same as last hour' error", "Improvement"]);
  const fa = r.false_alarms;
  $("evalOut").innerHTML = `
    <h3>1 · Same data, three ways of running the plant</h3>${modes}
    <p class="small muted">DO control error (mean distance from 2.0 mg/L): ${Object.entries(r.do_tracking_error).map(([k, v]) => `${k} ${v}`).join(" · ")}</p>
    <div class="eval-cols">
      <div><h3>2 · Which shocks each detector caught (graded against the true labels)</h3>${det}
        <p class="small muted">False alarms on ${fa.normal_hours} normal hours: rules ${fa.rules}, ML ${fa.ml}.</p></div>
      <div><h3>Detector scores</h3>${sc}<h3>3 · ML forecast vs "same as last hour"</h3>${fc}</div>
    </div>
    <p class="small">Reading this: the agents that act (divert, dose, control oxygen and chlorine) make the biggest difference.
      Rules and ML catch different shocks, so AI mode uses both. Results are on simulated data with simplified process formulas.</p>`;
}

start();
