// ── DOM refs ──────────────────────────────────────────────────────────────
const svg = document.getElementById("graph");
const runMeta = document.getElementById("run-meta");
const workspaceSelect = document.getElementById("workspace-select");
const taskGraphSelect = document.getElementById("task-graph-select");
const kindFilter = document.getElementById("kind-filter");
const searchInput = document.getElementById("search-input");
const searchOverlay = document.getElementById("search-overlay");
const searchResults = document.getElementById("search-results");
const searchCount = document.getElementById("search-count");
const searchClose = document.getElementById("search-close");
const subgraphList = document.getElementById("subgraph-list");
const inspector = document.getElementById("inspector-body");
const stats = document.getElementById("stats");
const legend = document.getElementById("legend");
const fitBtn = document.getElementById("fit-btn");
const reloadBtn = document.getElementById("reload-btn");

// ── State ─────────────────────────────────────────────────────────────────
const colors = {
  intent: "#59636f",
  tool: "#0f766e",
  model: "#2563eb",
  final: "#7c3aed",
  compress: "#0891b2",
  expand: "#b45309",
  memory_query: "#0891b2",
  router: "#b45309",
  observation: "#6d28d9",
  goal: "#15803d",
  other: "#b45309",
};

const STATUS_COLORS = {
  succeeded: "#16a34a",
  failed: "#dc2626",
  running: "#2563eb",
  retryable: "#d97706",
  skipped: "#9ca3af",
  pending: "#e5e7eb",
};

let graphData = null;
let selected = null;
let activeSubgraph = null; // task_graph_id filter, null = all
let viewBox = { x: 0, y: 0, w: 1000, h: 700 };
let isPanning = false;
let panStart = null;
let dragNode = null;
let dbAvailable = false;

// ── Resize handles ────────────────────────────────────────────────────────
function initResizeHandles() {
  const sidebar = document.querySelector(".sidebar");
  const inspectorPanel = document.querySelector(".inspector");

  function makeDragger(handle, panel, direction) {
    // direction: "right" = dragging rightward grows the left panel
    //            "left"  = dragging rightward shrinks the right panel
    let startX, startW;

    handle.addEventListener("mousedown", (e) => {
      e.preventDefault();
      startX = e.clientX;
      startW = panel.offsetWidth;
      handle.classList.add("dragging");
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
      document.addEventListener("mousemove", onMove);
      document.addEventListener("mouseup", onUp);
    });

    function onMove(e) {
      const dx = e.clientX - startX;
      const newW =
        direction === "right"
          ? Math.max(140, Math.min(520, startW + dx))
          : Math.max(180, Math.min(520, startW - dx));
      panel.style.width = newW + "px";
      fitGraph();
    }

    function onUp() {
      handle.classList.remove("dragging");
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    }
  }

  makeDragger(document.getElementById("resize-left"), sidebar, "right");
  makeDragger(document.getElementById("resize-right"), inspectorPanel, "left");
}

// ── Init ──────────────────────────────────────────────────────────────────

async function init() {
  searchInput.value = "";           // clear any browser-restored search value
  searchOverlay.classList.remove("is-open");
  initResizeHandles();
  try {
    const workspaces = await apiFetch("/api/workspaces");
    dbAvailable = true;
    populateWorkspaceSelect(workspaces);
    if (workspaces.length > 0) {
      workspaceSelect.value = workspaces[0].id;
      await onWorkspaceChange();
    }
  } catch {
    // DB unavailable — fall back to static graph.json
    dbAvailable = false;
    workspaceSelect.disabled = true;
    taskGraphSelect.disabled = true;
    await loadStaticGraph();
  }
}

function populateWorkspaceSelect(workspaces) {
  workspaceSelect.innerHTML = '<option value="">— workspace —</option>';
  for (const ws of workspaces) {
    const opt = document.createElement("option");
    opt.value = ws.id;
    opt.textContent = `${ws.name} (${ws.run_count} runs)`;
    workspaceSelect.appendChild(opt);
  }
}

async function onWorkspaceChange() {
  const wsId = workspaceSelect.value;
  if (!wsId) return;
  activeSubgraph = null;

  // Load task-graphs into sidebar + task-graph-select
  const taskGraphs = await apiFetch(`/api/workspaces/${wsId}/task-graphs`);
  populateTaskGraphSelect(taskGraphs);
  populateSubgraphList(taskGraphs);

  await loadGraph(wsId, null);
}

async function onTaskGraphChange() {
  const wsId = workspaceSelect.value;
  if (!wsId) return;
  const tgId = taskGraphSelect.value || null;
  activeSubgraph = tgId;
  await loadGraph(wsId, tgId);
}

function populateTaskGraphSelect(taskGraphs) {
  taskGraphSelect.innerHTML = '<option value="">All subgraphs</option>';
  for (const tg of taskGraphs) {
    const opt = document.createElement("option");
    opt.value = tg.id;
    const badge = tg.run_status ? ` [${tg.run_status}]` : "";
    opt.textContent = `${truncate(tg.id, 28)}${badge}`;
    taskGraphSelect.appendChild(opt);
  }
}

function populateSubgraphList(taskGraphs) {
  subgraphList.innerHTML = '<div class="section-title">Subgraphs</div>';
  const allItem = document.createElement("div");
  allItem.className = "subgraph-item active";
  allItem.dataset.id = "";
  allItem.textContent = `All (${taskGraphs.length})`;
  allItem.addEventListener("click", () => selectSubgraph(null, allItem));
  subgraphList.appendChild(allItem);

  for (const tg of taskGraphs) {
    const item = document.createElement("div");
    item.className = "subgraph-item";
    item.dataset.id = tg.id;
    const statusDot = tg.run_status
      ? `<span class="status-dot" style="background:${STATUS_COLORS[tg.run_status] || "#ccc"}"></span>`
      : "";
    item.innerHTML = `${statusDot}<span class="subgraph-label">${escapeHtml(truncate(tg.id, 26))}</span><span class="subgraph-count">${tg.node_count}</span>`;
    item.addEventListener("click", () => selectSubgraph(tg.id, item));
    subgraphList.appendChild(item);
  }
}

function selectSubgraph(tgId, clickedItem) {
  activeSubgraph = tgId;
  for (const el of subgraphList.querySelectorAll(".subgraph-item")) {
    el.classList.toggle("active", el === clickedItem);
  }
  taskGraphSelect.value = tgId || "";
  applyFilters();
  fitGraph();
  render();
}

// ── Graph loading ─────────────────────────────────────────────────────────

async function loadGraph(wsId, tgId) {
  const url = tgId
    ? `/api/workspaces/${wsId}/graph?task_graph_id=${encodeURIComponent(tgId)}`
    : `/api/workspaces/${wsId}/graph`;
  graphData = await apiFetch(url);
  prepareGraph(graphData);
  populateControls(graphData);
  fitGraph();
  render();
}

async function loadStaticGraph() {
  graphData = await apiFetch("/api/graph");
  if (graphData.error) throw new Error(graphData.error);
  prepareGraph(graphData);
  populateControls(graphData);
  fitGraph();
  render();
}

function prepareGraph(data) {
  const byId = new Map(data.nodes.map((n) => [n.id, n]));
  data.edges.forEach((edge) => {
    edge.sourceNode = byId.get(edge.source);
    edge.targetNode = byId.get(edge.target);
  });
  applyLayeredLayout(data.nodes, data.edges);
}

function applyFilters() {
  if (!graphData) return;
  const visibleKind = kindFilter.value;
  graphData.nodes.forEach((node) => {
    const matchKind = visibleKind === "all" || node.kind === visibleKind;
    const matchSubgraph = !activeSubgraph || node.task_graph_id === activeSubgraph;
    node.visible = matchKind && matchSubgraph;
  });
}

// ── Layout (unchanged from original) ──────────────────────────────────────

function applyLayeredLayout(nodes, edges) {
  const layers = new Map();
  nodes.forEach((node) => {
    node.layer = computeLayer(node.id, edges);
    node.width = measureNodeWidth(node);
    node.height = 64;
    node.visible = true;
    if (!layers.has(node.layer)) layers.set(node.layer, []);
    layers.get(node.layer).push(node);
  });
  const sortedLayers = [...layers.keys()].sort((a, b) => a - b);
  const gapX = 96;
  const gapY = 36;
  let x = 80;
  for (const layer of sortedLayers) {
    const column = sortLayer(layers.get(layer), edges);
    const maxWidth = Math.max(...column.map((n) => n.width));
    const totalH = column.reduce((s, n) => s + n.height, 0) + Math.max(0, column.length - 1) * gapY;
    let y = Math.max(80, 260 - totalH / 2);
    for (const node of column) {
      node.x = x + (maxWidth - node.width) / 2;
      node.y = y;
      y += node.height + gapY;
    }
    x += maxWidth + gapX;
  }
}

function computeLayer(nodeId, edges) {
  let layer = 0;
  let changed = true;
  while (changed) {
    changed = false;
    for (const edge of edges) {
      if (edge.target === nodeId) {
        const next = computeLayer(edge.source, edges) + 1;
        if (next > layer) { layer = next; changed = true; }
      }
    }
  }
  return layer;
}

function sortLayer(nodes, edges) {
  const incomingRank = new Map();
  for (const edge of edges) {
    if (!edge.sourceNode || !edge.targetNode) continue;
    incomingRank.set(
      edge.target,
      Math.min(incomingRank.get(edge.target) ?? Infinity, edge.sourceNode.y ?? 0),
    );
  }
  return [...nodes].sort((a, b) => {
    const ra = incomingRank.get(a.id) ?? 0;
    const rb = incomingRank.get(b.id) ?? 0;
    return ra !== rb ? ra - rb : a.label.localeCompare(b.label);
  });
}

function measureNodeWidth(node) {
  const ll = String(node.label || "").length;
  const sl = String(`${node.kind} / ${node.status}`).length;
  return Math.max(176, Math.min(280, Math.max(ll, sl) * 8 + 34));
}

// ── Controls ──────────────────────────────────────────────────────────────

function populateControls(data) {
  const meta = data.metadata || {};
  runMeta.textContent = `${meta.workspace_name || "workspace"} / ${meta.task_graph_id || "task graph"} / ${meta.run_status || "unknown"}`;

  const kinds = [...new Set(data.nodes.map((n) => n.kind))].sort();
  kindFilter.innerHTML = '<option value="all">All</option>';
  kinds.forEach((kind) => {
    const opt = document.createElement("option");
    opt.value = kind;
    opt.textContent = kind;
    kindFilter.appendChild(opt);
  });

  legend.innerHTML = '<div class="section-title">Legend</div>';
  kinds.forEach((kind) => {
    const item = document.createElement("div");
    item.className = "legend-item";
    item.innerHTML = `<span><span class="swatch" style="background:${colorFor(kind)}"></span> ${escapeHtml(kind)}</span>`;
    legend.appendChild(item);
  });

  stats.innerHTML = `
    <div class="section-title">Run</div>
    <div class="stat-row"><span>Nodes</span><strong>${data.nodes.length}</strong></div>
    <div class="stat-row"><span>Edges</span><strong>${data.edges.length}</strong></div>
    <div class="stat-row"><span>Status</span><strong>${escapeHtml(meta.run_status || "unknown")}</strong></div>
  `;
}

// ── Render ────────────────────────────────────────────────────────────────

function render() {
  applyFilters();
  svg.innerHTML = "";
  svg.setAttribute("viewBox", `${viewBox.x} ${viewBox.y} ${viewBox.w} ${viewBox.h}`);

  const defs = createSvg("defs");
  defs.innerHTML = `
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#6b7280"></path>
    </marker>
  `;
  svg.appendChild(defs);

  const edgeGroup = createSvg("g");
  const nodeGroup = createSvg("g");
  svg.append(edgeGroup, nodeGroup);

  graphData.edges.forEach((edge) => {
    if (!edge.sourceNode?.visible || !edge.targetNode?.visible) return;
    edgeGroup.appendChild(renderEdge(edge));
  });
  graphData.nodes.forEach((node) => {
    if (!node.visible) return;
    nodeGroup.appendChild(renderNode(node));
  });
}

function renderEdge(edge) {
  const isCross = edge.type === "cross";
  const group = createSvg("g");
  group.classList.add("edge-wrap");
  if (isCross) group.classList.add("cross");
  if (selected?.id === edge.id) group.classList.add("selected");
  const { x: x1, y: y1, width: w1, height: h1 } = edge.sourceNode;
  const { x: x2, y: y2, height: h2 } = edge.targetNode;
  const sx = x1 + w1, sy = y1 + h1 / 2;
  const tx = x2, ty = y2 + h2 / 2;
  const midX = Math.max(sx + 36, (sx + tx) / 2);
  const path = createSvg("path");
  path.classList.add("edge");
  if (isCross) path.classList.add("cross");
  if (selected?.id === edge.id) path.classList.add("selected");
  path.setAttribute("d", `M${sx},${sy} C${midX},${sy} ${midX},${ty} ${tx},${ty}`);
  path.setAttribute("marker-end", "url(#arrow)");
  group.appendChild(path);
  if (!isCross && edge.label) {
    const label = createSvg("text");
    label.classList.add("edge-label");
    label.setAttribute("x", midX - Math.min(80, String(edge.label).length * 2.8));
    label.setAttribute("y", (sy + ty) / 2 - 6);
    label.textContent = edge.label;
    group.appendChild(label);
  }
  group.addEventListener("click", (e) => {
    e.stopPropagation();
    selected = edge;
    showEdge(edge);
    render();
  });
  return group;
}

function renderNode(node) {
  const group = createSvg("g");
  group.classList.add("node");
  if (selected?.id === node.id) group.classList.add("selected");
  group.setAttribute("transform", `translate(${node.x}, ${node.y})`);

  const rect = createSvg("rect");
  rect.setAttribute("width", node.width);
  rect.setAttribute("height", node.height);
  rect.setAttribute("rx", 8);
  rect.setAttribute("stroke", colorFor(node.kind));

  // Status indicator strip on left edge
  const statusBar = createSvg("rect");
  statusBar.setAttribute("x", 0);
  statusBar.setAttribute("y", 0);
  statusBar.setAttribute("width", 5);
  statusBar.setAttribute("height", node.height);
  statusBar.setAttribute("rx", 8);
  statusBar.setAttribute("fill", STATUS_COLORS[node.status] || STATUS_COLORS.pending);

  const title = createSvg("text");
  title.setAttribute("x", 14);
  title.setAttribute("y", 23);
  title.textContent = truncate(node.label, Math.floor((node.width - 24) / 7));

  const sub = createSvg("text");
  sub.classList.add("subtitle");
  sub.setAttribute("x", 14);
  sub.setAttribute("y", 44);
  sub.textContent = truncate(`${node.kind} / ${node.status}`, Math.floor((node.width - 24) / 6.5));

  group.append(rect, statusBar, title, sub);

  group.addEventListener("click", (e) => {
    e.stopPropagation();
    selected = node;
    showNode(node);
    render();
  });
  group.addEventListener("pointerdown", (e) => {
    dragNode = { node, x: e.clientX, y: e.clientY, ox: node.x, oy: node.y };
    group.setPointerCapture(e.pointerId);
  });
  group.addEventListener("pointermove", (e) => {
    if (!dragNode || dragNode.node !== node) return;
    const sx = viewBox.w / svg.clientWidth;
    const sy = viewBox.h / svg.clientHeight;
    node.x = dragNode.ox + (e.clientX - dragNode.x) * sx;
    node.y = dragNode.oy + (e.clientY - dragNode.y) * sy;
    render();
  });
  group.addEventListener("pointerup", () => { dragNode = null; });
  return group;
}

// ── Inspector ─────────────────────────────────────────────────────────────

function showNode(node) {
  inspector.innerHTML = `
    <div class="detail-row"><span>Kind</span><strong>${escapeHtml(node.kind)}</strong></div>
    <div class="detail-row"><span>Status</span><strong style="color:${STATUS_COLORS[node.status] || "inherit"}">${escapeHtml(node.status)}</strong></div>
    <div class="detail-row"><span>Input</span><strong>${escapeHtml(node.schema?.input || "-")}</strong></div>
    <div class="detail-row"><span>Output</span><strong>${escapeHtml(node.schema?.output || "-")}</strong></div>
    <div class="detail-row"><span>Subgraph</span><strong>${escapeHtml(node.task_graph_id || "-")}</strong></div>
    <div class="detail-block"><div class="section-title">Execution</div><pre>${escapeHtml(JSON.stringify(node.execution || {}, null, 2))}</pre></div>
    <div class="detail-block"><div class="section-title">Metadata</div><pre>${escapeHtml(JSON.stringify(node.metadata || {}, null, 2))}</pre></div>
    <div id="node-traces"><em class="muted">Loading traces…</em></div>
  `;

  if (dbAvailable) {
    const runParam = node.run_id ? `?run_id=${encodeURIComponent(node.run_id)}` : "";
    apiFetch(`/api/nodes/${encodeURIComponent(node.id)}/detail${runParam}`)
      .then((data) => {
        const el = document.getElementById("node-traces");
        if (el) el.outerHTML = renderTraces(data);
      })
      .catch((err) => {
        const el = document.getElementById("node-traces");
        if (el) el.innerHTML = `<em class="muted">Traces unavailable: ${escapeHtml(String(err))}</em>`;
      });
  } else {
    document.getElementById("node-traces").innerHTML = "";
  }
}

function renderTraces(data) {
  const parts = [];

  if (data.run_state) {
    parts.push(`
      <details class="trace-section" open>
        <summary class="section-title">Run State</summary>
        <div class="detail-row"><span>Status</span><strong style="color:${STATUS_COLORS[data.run_state.status] || "inherit"}">${escapeHtml(data.run_state.status || "-")}</strong></div>
        <div class="detail-row"><span>Attempts</span><strong>${escapeHtml(String(data.run_state.attempts ?? "-"))}</strong></div>
        ${data.run_state.last_error_json ? `<pre>${escapeHtml(JSON.stringify(data.run_state.last_error_json, null, 2))}</pre>` : ""}
      </details>`);
  }

  if (data.artifacts?.length) {
    const cards = data.artifacts.map((a) => `
      <div class="trace-card">
        <div class="trace-card-header"><span class="badge">${escapeHtml(a.port)}</span><span class="muted">${escapeHtml(a.content_type)}</span><span class="muted">${a.token_count} tok</span></div>
        <pre>${escapeHtml(JSON.stringify(a.content_json, null, 2)).slice(0, 600)}${JSON.stringify(a.content_json).length > 600 ? "\n…" : ""}</pre>
      </div>`).join("");
    parts.push(`<details class="trace-section"><summary class="section-title">Artifacts (${data.artifacts.length})</summary>${cards}</details>`);
  }

  if (data.model_calls?.length) {
    const cards = data.model_calls.map((m) => `
      <div class="trace-card">
        <div class="trace-card-header"><span class="badge">${escapeHtml(m.model)}</span><span class="muted status-${m.status}">${escapeHtml(m.status)}</span></div>
        <pre>${escapeHtml(JSON.stringify(m.token_json, null, 2))}</pre>
      </div>`).join("");
    parts.push(`<details class="trace-section"><summary class="section-title">Model Calls (${data.model_calls.length})</summary>${cards}</details>`);
  }

  if (data.tool_calls?.length) {
    const cards = data.tool_calls.map((t) => `
      <div class="trace-card">
        <div class="trace-card-header"><span class="badge">${escapeHtml(t.tool_name)}</span><span class="muted status-${t.status}">${escapeHtml(t.status)}</span><span class="muted">${t.latency_ms != null ? t.latency_ms + "ms" : ""}</span></div>
        <pre>${escapeHtml(JSON.stringify(t.input_json, null, 2)).slice(0, 400)}</pre>
      </div>`).join("");
    parts.push(`<details class="trace-section"><summary class="section-title">Tool Calls (${data.tool_calls.length})</summary>${cards}</details>`);
  }

  if (!parts.length) return "";
  return parts.join("");
}

function showEdge(edge) {
  inspector.innerHTML = `
    <div class="detail-row"><span>Type</span><strong>${escapeHtml(edge.type)}</strong></div>
    <div class="detail-row"><span>Source</span><strong>${escapeHtml(edge.source)}</strong></div>
    <div class="detail-row"><span>Target</span><strong>${escapeHtml(edge.target)}</strong></div>
    <div class="detail-row"><span>Label</span><strong>${escapeHtml(edge.label || "-")}</strong></div>
    <div class="detail-block"><div class="section-title">Metadata</div><pre>${escapeHtml(JSON.stringify(edge.metadata || {}, null, 2))}</pre></div>
  `;
}

// ── Search ────────────────────────────────────────────────────────────────

let searchTimer = null;

searchInput.addEventListener("input", () => {
  clearTimeout(searchTimer);
  const q = searchInput.value.trim();
  if (!q) { closeSearch(); return; }
  searchTimer = setTimeout(() => runSearch(q), 300);
});

searchInput.addEventListener("keydown", (e) => {
  if (e.key === "Escape") { searchInput.value = ""; closeSearch(); }
});

searchClose.addEventListener("click", () => { searchInput.value = ""; closeSearch(); });

function closeSearch() {
  searchOverlay.classList.remove("is-open");
}

async function runSearch(q) {
  const wsId = workspaceSelect.value;
  const url = wsId
    ? `/api/search?q=${encodeURIComponent(q)}&workspace_id=${encodeURIComponent(wsId)}&limit=20`
    : `/api/search?q=${encodeURIComponent(q)}&limit=20`;
  const data = await apiFetch(url);
  renderSearchResults(q, data);
}

function renderSearchResults(q, data) {
  const total = (data.nodes?.length || 0) + (data.memory_items?.length || 0) + (data.goals?.length || 0) + (data.artifacts?.length || 0);
  searchCount.textContent = `${total} results for "${escapeHtml(q)}"`;
  searchResults.innerHTML = "";

  const sections = [
    { key: "nodes", label: "Nodes", render: renderNodeResult },
    { key: "memory_items", label: "Memory", render: renderMemoryResult },
    { key: "goals", label: "Goals", render: renderGoalResult },
    { key: "artifacts", label: "Artifacts", render: renderArtifactResult },
  ];

  for (const { key, label, render } of sections) {
    const items = data[key] || [];
    if (!items.length) continue;
    const heading = document.createElement("div");
    heading.className = "result-heading";
    heading.textContent = `${label} (${items.length})`;
    searchResults.appendChild(heading);
    for (const item of items) {
      searchResults.appendChild(render(item));
    }
  }

  searchOverlay.classList.add("is-open");
}

function renderNodeResult(item) {
  const card = document.createElement("div");
  card.className = "result-card";
  card.innerHTML = `<span class="badge" style="background:${colorFor(item.kind)}">${escapeHtml(item.kind)}</span> <strong>${escapeHtml(item.label || item.id)}</strong><div class="muted result-sub">${escapeHtml(item.task_graph_id || "")}</div>`;
  card.addEventListener("click", () => {
    closeSearch();
    const node = graphData?.nodes.find((n) => n.id === item.id);
    if (node) { selected = node; zoomToNode(node); showNode(node); render(); }
  });
  return card;
}

function renderMemoryResult(item) {
  const card = document.createElement("div");
  card.className = "result-card";
  card.innerHTML = `<span class="badge badge-memory">${escapeHtml(item.kind)}</span> <span>${escapeHtml(truncate(item.text, 80))}</span><div class="muted result-sub">confidence: ${item.confidence ?? "-"}</div>`;
  card.addEventListener("click", () => {
    inspector.innerHTML = `
      <div class="detail-row"><span>Kind</span><strong>${escapeHtml(item.kind)}</strong></div>
      <div class="detail-row"><span>Confidence</span><strong>${item.confidence ?? "-"}</strong></div>
      <div class="detail-block"><div class="section-title">Text</div><pre>${escapeHtml(item.text)}</pre></div>
    `;
  });
  return card;
}

function renderGoalResult(item) {
  const card = document.createElement("div");
  card.className = "result-card";
  card.innerHTML = `<span class="badge badge-goal">${escapeHtml(item.status)}</span> <span>${escapeHtml(truncate(item.text, 80))}</span>`;
  card.addEventListener("click", () => {
    inspector.innerHTML = `
      <div class="detail-row"><span>Status</span><strong>${escapeHtml(item.status)}</strong></div>
      <div class="detail-block"><div class="section-title">Goal</div><pre>${escapeHtml(item.text)}</pre></div>
    `;
  });
  return card;
}

function renderArtifactResult(item) {
  const card = document.createElement("div");
  card.className = "result-card";
  card.innerHTML = `<span class="badge badge-artifact">artifact</span> <span class="muted">${escapeHtml(item.node_id)} / ${escapeHtml(item.port)}</span><div class="result-excerpt">${escapeHtml(truncate(item.excerpt || "", 120))}</div>`;
  card.addEventListener("click", () => {
    inspector.innerHTML = `
      <div class="detail-row"><span>Node</span><strong>${escapeHtml(item.node_id)}</strong></div>
      <div class="detail-row"><span>Port</span><strong>${escapeHtml(item.port)}</strong></div>
      <div class="detail-block"><div class="section-title">Content excerpt</div><pre>${escapeHtml(item.excerpt || "")}</pre></div>
    `;
  });
  return card;
}

function zoomToNode(node) {
  const cx = node.x + node.width / 2;
  const cy = node.y + node.height / 2;
  viewBox = { x: cx - 400, y: cy - 300, w: 800, h: 600 };
}

// ── Fit / pan / zoom ──────────────────────────────────────────────────────

function fitGraph() {
  if (!graphData?.nodes?.length) return;
  const visible = graphData.nodes.filter((n) => n.visible !== false);
  if (!visible.length) return;
  const xs = visible.map((n) => n.x).filter(Number.isFinite);
  const ys = visible.map((n) => n.y).filter(Number.isFinite);
  if (!xs.length || !ys.length) return;
  const minX = Math.min(...xs);
  const minY = Math.min(...ys);
  const maxX = Math.max(...visible.filter((n) => Number.isFinite(n.x)).map((n) => n.x + n.width));
  const maxY = Math.max(...visible.filter((n) => Number.isFinite(n.y)).map((n) => n.y + n.height));
  viewBox = {
    x: minX - 80,
    y: minY - 80,
    w: Math.max(600, maxX - minX + 160),
    h: Math.max(420, maxY - minY + 160),
  };
}

svg.addEventListener("click", () => {
  selected = null;
  inspector.textContent = "Select a node or edge.";
  render();
});

svg.addEventListener("pointerdown", (e) => {
  if (e.target.closest(".node")) return;
  isPanning = true;
  panStart = { x: e.clientX, y: e.clientY, viewBox: { ...viewBox } };
});

svg.addEventListener("pointermove", (e) => {
  if (!isPanning || !panStart) return;
  const sx = viewBox.w / svg.clientWidth;
  const sy = viewBox.h / svg.clientHeight;
  viewBox.x = panStart.viewBox.x - (e.clientX - panStart.x) * sx;
  viewBox.y = panStart.viewBox.y - (e.clientY - panStart.y) * sy;
  render();
});

svg.addEventListener("pointerup", () => { isPanning = false; panStart = null; });

svg.addEventListener("wheel", (e) => {
  e.preventDefault();
  const factor = e.deltaY > 0 ? 1.12 : 0.88;
  const mx = viewBox.x + (e.offsetX / svg.clientWidth) * viewBox.w;
  const my = viewBox.y + (e.offsetY / svg.clientHeight) * viewBox.h;
  viewBox.w *= factor;
  viewBox.h *= factor;
  viewBox.x = mx - (e.offsetX / svg.clientWidth) * viewBox.w;
  viewBox.y = my - (e.offsetY / svg.clientHeight) * viewBox.h;
  render();
});

// ── Event wiring ──────────────────────────────────────────────────────────

workspaceSelect.addEventListener("change", () => onWorkspaceChange().catch(showError));
taskGraphSelect.addEventListener("change", () => onTaskGraphChange().catch(showError));
kindFilter.addEventListener("change", () => { fitGraph(); render(); });
fitBtn.addEventListener("click", () => { fitGraph(); render(); });
reloadBtn.addEventListener("click", () => init().catch(showError));

// ── Utilities ─────────────────────────────────────────────────────────────

async function apiFetch(url) {
  const res = await fetch(url, { cache: "no-store" });
  const json = await res.json();
  if (json?.error) throw new Error(json.error);
  return json;
}

function createSvg(name) {
  return document.createElementNS("http://www.w3.org/2000/svg", name);
}

function colorFor(kind) {
  return colors[kind] || colors.other;
}

function truncate(value, length) {
  const text = String(value || "");
  return text.length > length ? `${text.slice(0, length - 1)}…` : text;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function showError(error) {
  runMeta.textContent = "Error";
  inspector.innerHTML = `<pre>${escapeHtml(error.stack || error.message || String(error))}</pre>`;
}

// ── Boot ──────────────────────────────────────────────────────────────────
init().catch(showError);
