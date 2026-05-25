const svg = document.getElementById("graph");
const runMeta = document.getElementById("run-meta");
const kindFilter = document.getElementById("kind-filter");
const inspector = document.getElementById("inspector-body");
const stats = document.getElementById("stats");
const legend = document.getElementById("legend");
const fitBtn = document.getElementById("fit-btn");
const reloadBtn = document.getElementById("reload-btn");

const colors = {
  intent: "#59636f",
  tool: "#0f766e",
  model: "#2563eb",
  final: "#7c3aed",
  compress: "#0891b2",
  expand: "#b45309",
  other: "#b45309",
};

let graphData = null;
let selected = null;
let viewBox = { x: 0, y: 0, w: 1000, h: 700 };
let isPanning = false;
let panStart = null;
let dragNode = null;

async function loadGraph() {
  const response = await fetch("/api/graph", { cache: "no-store" });
  graphData = await response.json();
  if (graphData.error) {
    throw new Error(graphData.error);
  }
  prepareGraph(graphData);
  populateControls(graphData);
  fitGraph();
  render();
}

function prepareGraph(data) {
  const byId = new Map(data.nodes.map((node) => [node.id, node]));
  data.edges.forEach((edge) => {
    edge.sourceNode = byId.get(edge.source);
    edge.targetNode = byId.get(edge.target);
  });
  applyLayeredLayout(data.nodes, data.edges);
}

function applyLayeredLayout(nodes, edges) {
  const layers = new Map();
  nodes.forEach((node) => {
    node.layer = computeLayer(node.id, edges);
    node.width = measureNodeWidth(node);
    node.height = 64;
    node.visible = true;
    if (!layers.has(node.layer)) {
      layers.set(node.layer, []);
    }
    layers.get(node.layer).push(node);
  });

  const sortedLayers = [...layers.keys()].sort((a, b) => a - b);
  const layerWidths = sortedLayers.map((layer) =>
    Math.max(...layers.get(layer).map((node) => node.width)),
  );
  const gapX = 96;
  const gapY = 36;
  let x = 80;
  for (const layer of sortedLayers) {
    const column = sortLayer(layers.get(layer), edges);
    const maxWidth = layerWidths[sortedLayers.indexOf(layer)];
    const totalHeight =
      column.reduce((sum, node) => sum + node.height, 0) + Math.max(0, column.length - 1) * gapY;
    let y = Math.max(80, 260 - totalHeight / 2);
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
        if (next > layer) {
          layer = next;
          changed = true;
        }
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
      Math.min(incomingRank.get(edge.target) ?? Number.POSITIVE_INFINITY, edge.sourceNode.y ?? 0),
    );
  }
  return [...nodes].sort((a, b) => {
    const rankA = incomingRank.get(a.id) ?? 0;
    const rankB = incomingRank.get(b.id) ?? 0;
    if (rankA !== rankB) return rankA - rankB;
    return a.label.localeCompare(b.label);
  });
}

function measureNodeWidth(node) {
  const labelLength = String(node.label || "").length;
  const subtitleLength = String(`${node.kind} / ${node.status}`).length;
  return Math.max(176, Math.min(280, Math.max(labelLength, subtitleLength) * 8 + 34));
}

function populateControls(data) {
  const meta = data.metadata || {};
  runMeta.textContent = `${meta.workspace_name || "workspace"} / ${meta.task_graph_id || "task graph"} / ${meta.run_status || "unknown"}`;
  const kinds = [...new Set(data.nodes.map((node) => node.kind))].sort();
  kindFilter.innerHTML = '<option value="all">All</option>';
  kinds.forEach((kind) => {
    const option = document.createElement("option");
    option.value = kind;
    option.textContent = kind;
    kindFilter.appendChild(option);
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

function render() {
  const visibleKind = kindFilter.value;
  graphData.nodes.forEach((node) => {
    node.visible = visibleKind === "all" || node.kind === visibleKind;
  });
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
  const group = createSvg("g");
  group.classList.add("edge-wrap");
  if (selected?.id === edge.id) group.classList.add("selected");
  const source = edge.sourceNode;
  const target = edge.targetNode;
  const x1 = source.x + source.width;
  const y1 = source.y + source.height / 2;
  const x2 = target.x;
  const y2 = target.y + target.height / 2;
  const midX = Math.max(x1 + 36, (x1 + x2) / 2);
  const path = createSvg("path");
  path.classList.add("edge");
  if (selected?.id === edge.id) path.classList.add("selected");
  path.setAttribute("d", `M${x1},${y1} C${midX},${y1} ${midX},${y2} ${x2},${y2}`);
  path.setAttribute("marker-end", "url(#arrow)");
  const label = createSvg("text");
  label.classList.add("edge-label");
  label.setAttribute("x", midX - Math.min(80, String(edge.label || edge.type).length * 2.8));
  label.setAttribute("y", (y1 + y2) / 2 - 6);
  label.textContent = edge.label || edge.type;
  group.append(path, label);
  group.addEventListener("click", (event) => {
    event.stopPropagation();
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
  const title = createSvg("text");
  title.setAttribute("x", 12);
  title.setAttribute("y", 23);
  title.textContent = truncate(node.label, Math.floor((node.width - 24) / 7));
  const sub = createSvg("text");
  sub.classList.add("subtitle");
  sub.setAttribute("x", 12);
  sub.setAttribute("y", 44);
  sub.textContent = truncate(`${node.kind} / ${node.status}`, Math.floor((node.width - 24) / 6.5));
  group.append(rect, title, sub);
  group.addEventListener("click", (event) => {
    event.stopPropagation();
    selected = node;
    showNode(node);
    render();
  });
  group.addEventListener("pointerdown", (event) => {
    dragNode = { node, x: event.clientX, y: event.clientY, ox: node.x, oy: node.y };
    group.setPointerCapture(event.pointerId);
  });
  group.addEventListener("pointermove", (event) => {
    if (!dragNode || dragNode.node !== node) return;
    const scaleX = viewBox.w / svg.clientWidth;
    const scaleY = viewBox.h / svg.clientHeight;
    node.x = dragNode.ox + (event.clientX - dragNode.x) * scaleX;
    node.y = dragNode.oy + (event.clientY - dragNode.y) * scaleY;
    render();
  });
  group.addEventListener("pointerup", () => {
    dragNode = null;
  });
  return group;
}

function showNode(node) {
  inspector.innerHTML = `
    <div class="detail-row"><span>Kind</span><strong>${escapeHtml(node.kind)}</strong></div>
    <div class="detail-row"><span>Status</span><strong>${escapeHtml(node.status)}</strong></div>
    <div class="detail-row"><span>Input</span><strong>${escapeHtml(node.schema?.input || "-")}</strong></div>
    <div class="detail-row"><span>Output</span><strong>${escapeHtml(node.schema?.output || "-")}</strong></div>
    <div class="detail-block"><div class="section-title">Metrics</div><pre>${escapeHtml(JSON.stringify(node.metrics || {}, null, 2))}</pre></div>
    <div class="detail-block"><div class="section-title">Metadata</div><pre>${escapeHtml(JSON.stringify(node.metadata || {}, null, 2))}</pre></div>
  `;
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

function fitGraph() {
  if (!graphData?.nodes?.length) return;
  const visible = graphData.nodes.filter((node) => node.visible !== false);
  const minX = Math.min(...visible.map((node) => node.x));
  const minY = Math.min(...visible.map((node) => node.y));
  const maxX = Math.max(...visible.map((node) => node.x + node.width));
  const maxY = Math.max(...visible.map((node) => node.y + node.height));
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

svg.addEventListener("pointerdown", (event) => {
  if (event.target.closest(".node")) return;
  isPanning = true;
  panStart = { x: event.clientX, y: event.clientY, viewBox: { ...viewBox } };
});

svg.addEventListener("pointermove", (event) => {
  if (!isPanning || !panStart) return;
  const scaleX = viewBox.w / svg.clientWidth;
  const scaleY = viewBox.h / svg.clientHeight;
  viewBox.x = panStart.viewBox.x - (event.clientX - panStart.x) * scaleX;
  viewBox.y = panStart.viewBox.y - (event.clientY - panStart.y) * scaleY;
  render();
});

svg.addEventListener("pointerup", () => {
  isPanning = false;
  panStart = null;
});

svg.addEventListener("wheel", (event) => {
  event.preventDefault();
  const factor = event.deltaY > 0 ? 1.12 : 0.88;
  const mx = viewBox.x + (event.offsetX / svg.clientWidth) * viewBox.w;
  const my = viewBox.y + (event.offsetY / svg.clientHeight) * viewBox.h;
  viewBox.w *= factor;
  viewBox.h *= factor;
  viewBox.x = mx - (event.offsetX / svg.clientWidth) * viewBox.w;
  viewBox.y = my - (event.offsetY / svg.clientHeight) * viewBox.h;
  render();
});

kindFilter.addEventListener("change", () => {
  fitGraph();
  render();
});

fitBtn.addEventListener("click", () => {
  fitGraph();
  render();
});

reloadBtn.addEventListener("click", () => {
  loadGraph().catch(showError);
});

function createSvg(name) {
  return document.createElementNS("http://www.w3.org/2000/svg", name);
}

function colorFor(kind) {
  return colors[kind] || colors.other;
}

function truncate(value, length) {
  const text = String(value || "");
  return text.length > length ? `${text.slice(0, length - 1)}...` : text;
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
  runMeta.textContent = "Failed to load graph";
  inspector.innerHTML = `<pre>${escapeHtml(error.stack || error.message || error)}</pre>`;
}

loadGraph().catch(showError);
