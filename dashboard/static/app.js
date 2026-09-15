"use strict";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const NS = "http://www.w3.org/2000/svg";
const ROUTE_COLORS = ["#b7ff4a", "#73d7ff", "#ff8c68", "#c59cff", "#ffd166", "#5ee6b8", "#ff6b9a", "#a6b4ff"];
const METRICS = {
  goal_hit_rate: "Goal hit rate",
  optimal_path_rate: "Optimal path rate",
  success_cost_ratio: "Success cost ratio",
  loop_rate: "Loop rate",
  broken_rate: "Broken rate",
  coverage_rate: "Coverage rate",
  optimal_coverage_rate: "Optimal coverage",
  // Weighted 扩展 / 多分支增强新增的口径
  weighted_optimal_coverage_rate: "加权最优覆盖率（表里至少一条最小 cost 路）",
  mean_goal_paths: "平均 Goal 路径数",
  mean_finished_paths: "平均终止路径数",
  mean_filtered_dead_branches: "平均被预筛选的必死 branch"
};

const KIND_LABEL = { weighted: "带权", ablated: "带权·无cost", unweighted: "无权" };

const state = {
  catalog: null,
  pathData: null,
  animation: null,
  startedAt: 0,
  elapsedBeforePause: 0,
  duration: 6000,
  paused: false,
  routeGraphics: [],
  view: "path",
  diffusionMode: "clean",
  diffusionFrame: 0,
  diffusionTimer: null,
  diffusionPlaying: false,
  diffusionNodeShapes: null,
  sampleCount: 0
};

function routeColor(index) {
  if (index < ROUTE_COLORS.length) return ROUTE_COLORS[index];
  return `hsl(${Math.round((index * 137.508) % 360)} 78% 68%)`;
}

function svgEl(name, attrs = {}) {
  const el = document.createElementNS(NS, name);
  Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
  return el;
}

function showError(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.hidden = false;
  clearTimeout(showError.timer);
  showError.timer = setTimeout(() => { toast.hidden = true; }, 7000);
}

async function api(url, options) {
  const response = await fetch(url, options);
  let data;
  try { data = await response.json(); } catch { data = {}; }
  if (!response.ok) throw new Error(data.error || `请求失败 (${response.status})`);
  return data;
}

function fillSelect(select, items, valueKey = "id", labelKey = "label") {
  select.innerHTML = "";
  items.forEach((item) => {
    const option = document.createElement("option");
    option.value = item[valueKey];
    option.textContent = item[labelKey];
    select.append(option);
  });
}

/**
 * 数据集下拉：按 `data/` 下的来源子目录分组（controlled / long / oldv1 / mixed /
 * smoke），每项的 title 给出仓库内的相对路径，选中后一眼知道文件在哪。
 */
function fillDatasetSelect(select, items) {
  select.innerHTML = "";
  const groups = new Map();
  items.forEach((item) => {
    const key = item.group || "";
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(item);
  });
  const makeOption = (item) => {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = item.label;
    if (item.relative) option.title = item.relative;
    return option;
  };
  groups.forEach((entries, group) => {
    if (!group) {
      entries.forEach((item) => select.append(makeOption(item)));
      return;
    }
    const holder = document.createElement("optgroup");
    holder.label = `${group}  (${entries.length})`;
    entries.forEach((item) => holder.append(makeOption(item)));
    select.append(holder);
  });
}

function preferred(items, needle) {
  return items.find((item) => item.id.includes(needle))?.id || items[0]?.id || "";
}

async function initialize() {
  try {
    state.catalog = await api("/api/catalog");
    $("#connection").classList.add("online");
    $("#connection span").textContent = `${state.catalog.models.length} 个模型 · ${state.catalog.datasets.length} 个数据集`;
    fillSelect($("#path-model"), state.catalog.models);
    fillDatasetSelect($("#path-dataset"), state.catalog.datasets);
    $("#path-model").value = preferred(state.catalog.models, "v2_rev2_mixed");
    $("#path-dataset").value = preferred(state.catalog.datasets, "controlled_test.pkl");

    const metricDatasets = [...new Set(state.catalog.metrics.map((row) => row.dataset))];
    metricDatasets.forEach((id) => $("#metric-dataset").append(new Option(id.replace(".pkl", "").replaceAll("_", " "), id)));
    Object.entries(METRICS).forEach(([id, label]) => $("#metric-name").append(new Option(label, id)));
    renderModelFilters();
    await updateDatasetInfo();
    renderMetrics();
  } catch (error) {
    $("#connection span").textContent = "连接失败";
    showError(error.message);
  }
}

function renderModelFilters() {
  const root = $("#model-filters");
  root.innerHTML = "";
  state.catalog.models.forEach((model, index) => {
    const label = document.createElement("label");
    label.className = "model-filter";
    const kind = model.kind || "unweighted";
    label.innerHTML = `<input type="checkbox" value="${escapeHtml(model.id)}" ${index < 6 ? "checked" : ""}><span>${escapeHtml(model.label)}</span><em class="kind-badge ${kind}">${KIND_LABEL[kind] || kind}</em>`;
    root.append(label);
  });
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[char]));
}

async function updateDatasetInfo() {
  const id = $("#path-dataset").value;
  if (!id) return;
  try {
    $("#dataset-count").textContent = "读取中";
    const info = await api(`/api/datasets/${encodeURIComponent(id)}`);
    state.sampleCount = Math.max(info.num_queries, 1);
    $("#sample-index").max = Math.max(info.num_queries - 1, 0);
    $("#dataset-count").textContent = `/ ${Math.max(info.num_queries - 1, 0)}`;
  } catch (error) { showError(error.message); }
}

/** 在当前数据集里随机抽一个样本，抽完直接生成（点一次就能看一条新的）。 */
async function pickRandomSample() {
  if ($("#generate").disabled) return;          // 正在推理，不要叠加
  try {
    if (!state.sampleCount) await updateDatasetInfo();
    const count = state.sampleCount || 0;
    if (count <= 0) {
      showError("当前数据集没有可用样本");
      return;
    }
    const current = Number($("#sample-index").value);
    let index = Math.floor(Math.random() * count);
    if (count > 1 && index === current) index = (index + 1) % count;   // 尽量换一条
    $("#sample-index").value = index;
    await generatePath();
  } catch (error) {
    showError(error.message);
  }
}

function setLoading(loading) {
  const button = $("#generate");
  button.disabled = loading;
  $("#random-sample").disabled = loading;
  button.textContent = loading ? "正在运行反向扩散…" : "生成并播放";
  $("#path-status").textContent = loading ? "模型推理中；首次切换模型需要加载 checkpoint" : $("#path-status").textContent;
}

async function generatePath() {
  setLoading(true);
  stopAnimation();
  stopDiffusion();
  try {
    const payload = {
      model: $("#path-model").value,
      dataset: $("#path-dataset").value,
      index: Number($("#sample-index").value),
      decode: $("#decode-mode").value,
      top_k: Number($("#top-k").value),
      beam_width: Number($("#beam-width").value),
      display_paths: Number($("#display-paths").value),
      null_policy: $("#null-policy").value,
      filter_dead_branches: $("#filter-dead").checked,
      deterministic: $("#deterministic").checked,
      seed: 0,
      include_diffusion: true
    };
    state.pathData = await api("/api/path", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
    state.diffusionFrame = 0;
    $("#view-diffusion").disabled = !state.pathData.diffusion;
    renderPathInfo();
    if (state.view === "diffusion" && state.pathData.diffusion) {
      renderDiffusionGraph();
      startDiffusion(true);
    } else {
      renderGraph();
      startAnimation(true);
    }
  } catch (error) {
    $("#path-status").textContent = "推理未完成";
    showError(error.message);
  } finally { setLoading(false); }
}

function statusText(status) {
  return status === "goal" ? "到达" : status === "loop" ? "成环" : status === "broken" ? "中止" : status || "—";
}

const READOUT_TITLE = {
  final_prob_argmax: "Single 解码 = 最终 candidate_prob 的组内 argmax（reverse chain 仍按 posterior 采样）",
  deterministic_rollout: "Single 解码 = 全程 argmax rollout（每一步 posterior 都取 argmax）",
  sampled_z0: "Single 解码 = 采样出来的 z₀（旧行为）"
};

/**
 * single 解码**不再**跟随采样状态：默认用"最终候选概率的组内 argmax"。
 * 扩散链自己携带的状态（默认是采样出来的）只用于可视化/诊断 —— 两者允许不一致
 * （实测 #92：readout 18 跳到达，采样链在最后一个路口抽到 NULL 判 broken）。
 */
function renderDecodeContrast(data) {
  const box = $("#decoder-contrast");
  const contrast = data.contrast;
  if (!contrast) { box.hidden = true; return; }
  const readout = contrast.readout;
  const chainState = contrast.chain_state;
  const differ = readout.status !== chainState.status;
  const line = (tag, entry) =>
    `${tag}：<b>${statusText(entry.status)}</b> · ${entry.hops} 跳 / cost ${Number(entry.path_cost).toFixed(1)}` +
    (entry.reason ? ` · ${escapeHtml(entry.reason)}` : "");
  const chainTag = contrast.chain_stochastic
    ? "扩散链实际采样状态（仅诊断）"
    : "扩散链状态（本模式=贪心 rollout）";
  box.hidden = false;
  box.innerHTML = `<strong>${READOUT_TITLE[contrast.readout_mode] || "Single 解码"}</strong><br>` +
    line("解码使用", readout) + "<br>" + line(chainTag, chainState) +
    (differ ? "<br><span class=\"muted\">两者不同是正常的：解码不再跟随采样状态。</span>" : "");
}

function renderPathInfo() {
  const data = state.pathData;
  const sample = data.sample;
  $("#sample-title").textContent = `#${data.index} · ${sample.start} → ${sample.goal}`;
  const values = [
    `${sample.difficulty} / ${sample.mode}`,
    `${sample.num_nodes} / ${sample.num_decisions}`,
    `${sample.gt_length} 跳`,
    data.summary.coverage ? "至少一条到达" : "未到达"
  ];
  $$("#sample-stats dd").forEach((dd, i) => { dd.textContent = values[i]; });
  renderDecodeContrast(data);
  const legend = $("#path-legend");
  legend.innerHTML = "";
  data.routes.forEach((route, index) => {
    const item = document.createElement("div");
    item.className = "legend-item";
    const label = route.status === "goal" ? "到达" : route.status === "loop" ? "成环" : "中止";
    // 带权图上"跳数"和"真实 cost"不是一回事，两个都显示
    const costText = route.weighted
      ? `${route.cost} 跳 / cost ${Number(route.path_cost).toFixed(1)}`
      : `${route.cost} 跳`;
    item.innerHTML = `<i style="background:${routeColor(index)}"></i><span>#${route.rank} · ${label} · ${costText}</span>`;
    legend.append(item);
  });
  const reached = data.routes.filter((route) => route.status === "goal").length;
  $("#path-status").textContent = `${data.routes.length} 条可视路线 · ${reached} 条到达终点 · checkpoint epoch ${data.checkpoint_epoch ?? "—"}`;
}

function graphCoordinates(data) {
  const margin = 55;
  const width = 1000 - margin * 2;
  const height = 620 - margin * 2;
  return new Map(data.graph.nodes.map((node) => [node.id, {
    x: margin + node.x * width,
    y: margin + (1 - node.y) * height,
    kind: node.kind
  }]));
}

function edgeKey(source, target) {
  return source < target ? `${source}:${target}` : `${target}:${source}`;
}

/**
 * 每条物理边上都有哪些"路线 × 第几段"经过。
 *
 * 额外记录 ``uniqueRoutes``（这条边被多少条**不同**的路线走过）与 ``totalRoutes``：
 * 一条被**全部**显示路线走过的边属于"公共前缀"，它不携带任何区分信息 —— 画成
 * 一束平行线只会让人以为这里就分叉了（实测反馈），所以这种边合并成一条主干线。
 */
function buildEdgeUsage(routes) {
  const usage = new Map();
  const uniqueRoutes = new Map();
  routes.forEach((route, routeIndex) => {
    route.edges.forEach(([source, target], step) => {
      const key = edgeKey(source, target);
      if (!usage.has(key)) {
        usage.set(key, []);
        uniqueRoutes.set(key, new Set());
      }
      usage.get(key).push(`${routeIndex}:${step}`);
      uniqueRoutes.get(key).add(routeIndex);
    });
  });
  return {lanes: usage, uniqueRoutes, totalRoutes: routes.length};
}

const SWEEP_STEP = 3;       // user units between polyline samples of one edge
const MAX_BUS_SPAN = 9;     // shared edges never spread wider than this
const ROUTE_WIDTH = 4.5;    // stroke width of an unshared route edge
const TRUNK_COLOR = "#93a3bd";   // 公共前缀（所有路线都经过）画成一条中性主干

/** Lane assignment for one traversal of one physical edge. */
function edgeLane(source, target, routeIndex, step, coordinates, usage) {
  const key = edgeKey(source, target);
  const lanes = usage.lanes.get(key) || [];
  const laneIndex = Math.max(0, lanes.indexOf(`${routeIndex}:${step}`));
  const laneCount = Math.max(1, lanes.length);
  // 所有显示路线都经过这条边 -> 公共前缀，不铺车道
  const shared =
    usage.totalRoutes > 1 &&
    (usage.uniqueRoutes.get(key)?.size || 0) === usage.totalRoutes;
  if (shared) {
    const canonicalA = coordinates.get(Math.min(source, target));
    const canonicalB = coordinates.get(Math.max(source, target));
    const dx = canonicalB.x - canonicalA.x;
    const dy = canonicalB.y - canonicalA.y;
    const length = Math.hypot(dx, dy) || 1;
    return {offset: 0, strokeWidth: ROUTE_WIDTH, shared: true, nx: -dy / length, ny: dx / length};
  }

  // The normal is derived from the canonical (small id -> large id) edge
  // direction, so a route traversing an edge backwards keeps the same lane.
  const canonicalA = coordinates.get(Math.min(source, target));
  const canonicalB = coordinates.get(Math.max(source, target));
  const dx = canonicalB.x - canonicalA.x;
  const dy = canonicalB.y - canonicalA.y;
  const length = Math.hypot(dx, dy) || 1;

  // Shared edges become a compact, centred multi-colour bus.  The complete
  // bundle never grows beyond roughly nine user units, even with many branches.
  const bundleSpan = laneCount > 1 ? Math.min(MAX_BUS_SPAN, (laneCount - 1) * 1.65) : 0;
  const spacing = laneCount > 1 ? bundleSpan / (laneCount - 1) : 0;
  return {
    offset: (laneIndex - (laneCount - 1) / 2) * spacing,
    strokeWidth: laneCount > 1 ? Math.max(1.1, Math.min(1.75, spacing * 0.72)) : ROUTE_WIDTH,
    shared: false,
    nx: -dy / length,
    ny: dx / length
  };
}

// Both endpoints stay exactly on the node centres; the lane offset is only
// introduced inside the edge, so consecutive traversals always share one point
// and a route can never leave a stub or a hole at a junction.
function edgePathData(source, target, routeIndex, step, coordinates, usage) {
  const a = coordinates.get(source);
  const b = coordinates.get(target);
  const lane = edgeLane(source, target, routeIndex, step, coordinates, usage);
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const length = Math.hypot(dx, dy) || 1;
  const ux = dx / length;
  const uy = dy / length;
  const join = Math.min(18, length * 0.32);
  const c1x = a.x + ux * join + lane.nx * lane.offset;
  const c1y = a.y + uy * join + lane.ny * lane.offset;
  const c2x = b.x - ux * join + lane.nx * lane.offset;
  const c2y = b.y - uy * join + lane.ny * lane.offset;
  const fmt = (value) => value.toFixed(2);
  return {
    d: `M${fmt(a.x)},${fmt(a.y)} C${fmt(c1x)},${fmt(c1y)} ${fmt(c2x)},${fmt(c2y)} ${fmt(b.x)},${fmt(b.y)}`,
    strokeWidth: lane.strokeWidth,
    shared: lane.shared
  };
}

/**
 * Sample a path element that is already part of the rendered document.
 *
 * The reveal is driven by real arc-length samples in user units and never by
 * `stroke-dasharray`.  With `vector-effect: non-scaling-stroke` Chromium
 * measures dash patterns in screen pixels instead of user units, so a dash as
 * long as `getTotalLength()` only covers `1 / scale` of the curve: the tail of
 * every edge fell into the dash gap and vanished right before the next node.
 * Sampling also keeps working for any panel size, zoom level and browser.
 */
function samplePath(element, step = SWEEP_STEP) {
  const total = element.getTotalLength();
  if (!(total > 1e-9)) {
    // A degenerate edge (zero length) still needs a usable head position and a
    // non-zero sample step so the sweep arithmetic never divides by zero.
    const point = element.getPointAtLength(0);
    return {points: [{x: point.x, y: point.y}, {x: point.x, y: point.y}], total: 0, step: 1};
  }
  const count = Math.max(1, Math.ceil(total / step));
  const points = [];
  for (let index = 0; index <= count; index += 1) {
    const point = element.getPointAtLength((index * total) / count);
    points.push({x: point.x, y: point.y});
  }
  return {points, total, step: total / count};
}

/**
 * Reveal one sampled edge up to `traveled` user units and return its moving
 * head.  The emitted prefix is cached, so playing forward only appends the
 * samples that are still missing while a rewind rebuilds from the start.
 */
function sweepSegment(segment, traveled) {
  const {points, total, step} = segment.sample;
  const clamped = Math.max(0, Math.min(total, traveled));
  const whole = Math.min(points.length - 1, Math.floor(clamped / step + 1e-9));
  if (whole < segment.emitted) {
    segment.d = `M${points[0].x.toFixed(2)},${points[0].y.toFixed(2)}`;
    segment.emitted = 0;
  }
  while (segment.emitted < whole) {
    segment.emitted += 1;
    const point = points[segment.emitted];
    segment.d += `L${point.x.toFixed(2)},${point.y.toFixed(2)}`;
  }
  let head = points[whole];
  let d = segment.d;
  const remainder = clamped - whole * step;
  if (whole < points.length - 1 && remainder > 1e-9) {
    const next = points[whole + 1];
    const ratio = remainder / step;
    head = {x: head.x + (next.x - head.x) * ratio, y: head.y + (next.y - head.y) * ratio};
    d += `L${head.x.toFixed(2)},${head.y.toFixed(2)}`;
  }
  if (d !== segment.applied) {
    segment.line.setAttribute("d", d);
    segment.casing.setAttribute("d", d);
    segment.applied = d;
  }
  return head;
}

function pathD(points) {
  return points.map((point, index) => `${index ? "L" : "M"}${point.x.toFixed(2)},${point.y.toFixed(2)}`).join(" ");
}

function renderGraph() {
  const data = state.pathData;
  const svg = $("#graph");
  svg.innerHTML = "";
  const coordinates = graphCoordinates(data);

  const base = svgEl("g", {"aria-hidden": "true"});
  data.graph.edges.forEach((edge) => {
    const a = coordinates.get(edge.source), b = coordinates.get(edge.target);
    base.append(svgEl("line", {x1: a.x, y1: a.y, x2: b.x, y2: b.y, class: "base-edge"}));
  });
  svg.append(base);

  const gtPoints = data.sample.gt_path.map((id) => coordinates.get(id)).filter(Boolean);
  const gt = svgEl("path", {d: pathD(gtPoints), class: "gt-edge", id: "gt-path"});
  gt.hidden = !$("#show-gt").checked;
  svg.append(gt);

  const nodeCircle = (node) => {
    const point = coordinates.get(node.id);
    const radius = node.kind === "ordinary" ? 5 : 8;
    return svgEl("circle", {cx: point.x, cy: point.y, r: radius, class: `node ${node.kind}`});
  };
  const nodeLabel = (node) => {
    const point = coordinates.get(node.id);
    const label = svgEl("text", {x: point.x, y: point.y - 13, class: "node-label"});
    label.textContent = node.kind === "start" ? `S · ${node.id}` : node.kind === "goal" ? `G · ${node.id}` : node.id;
    return label;
  };
  const backgroundNodes = svgEl("g", {"aria-hidden": "true"});
  data.graph.nodes.forEach((node) => {
    backgroundNodes.append(nodeCircle(node));
    if (node.kind !== "ordinary") backgroundNodes.append(nodeLabel(node));
  });
  svg.append(backgroundNodes);

  // Layering is topology-aware: route casings and colours cross above nodes
  // that are not on the route; actual route nodes are redrawn on top. A
  // visual crossing can therefore never erase an edge or look like a join.
  const casingLayer = svgEl("g", {"aria-hidden": "true"});
  const routeLayer = svgEl("g");
  const activeNodeLayer = svgEl("g", {"aria-hidden": "true"});
  const labelLayer = svgEl("g", {"aria-hidden": "true"});
  const headLayer = svgEl("g", {"aria-hidden": "true"});
  const usage = buildEdgeUsage(data.routes);
  const activeNodeIds = new Set(data.routes.flatMap((route) => route.nodes));
  state.routeGraphics = [];
  svg.append(casingLayer);
  svg.append(routeLayer);
  data.routes.forEach((route, index) => {
    const color = routeColor(index);
    const group = svgEl("g");
    const casingGroup = svgEl("g");
    const title = svgEl("title");
    title.textContent = `路线 #${route.rank} · ${route.status} · ${route.cost} 跳${route.reason ? ` · ${route.reason}` : ""}`;
    group.append(title);
    const segments = [];
    route.edges.forEach(([source, target], step) => {
      const geometry = edgePathData(source, target, index, step, coordinates, usage);
      // 公共前缀统一用中性色：8 条路线在这里完全重合，着色反而像"一开始就分叉"
      const line = svgEl("path", {
        d: geometry.d,
        stroke: geometry.shared ? TRUNK_COLOR : color,
        "stroke-width": geometry.strokeWidth,
        class: "route"
      });
      const casing = svgEl("path", {
        d: geometry.d,
        "stroke-width": geometry.strokeWidth + 3.2,
        class: "route-casing"
      });
      casingGroup.append(casing);
      group.append(line);
      segments.push({line, casing});
    });
    casingLayer.append(casingGroup);
    routeLayer.append(group);
    // Measure each edge only now that it belongs to the rendered document, and
    // park it at its start point so nothing is painted before the sweep begins.
    let cumulative = 0;
    segments.forEach((segment) => {
      segment.sample = samplePath(segment.line);
      segment.start = cumulative;
      segment.d = `M${segment.sample.points[0].x.toFixed(2)},${segment.sample.points[0].y.toFixed(2)}`;
      segment.emitted = 0;
      segment.applied = null;
      segment.line.setAttribute("d", segment.d);
      segment.casing.setAttribute("d", segment.d);
      cumulative += segment.sample.total;
    });
    const head = svgEl("circle", {r: 5.5, fill: color, class: "route-head"});
    headLayer.append(head);
    state.routeGraphics.push({segments, head, length: cumulative, startNode: coordinates.get(route.nodes[0])});
  });

  data.graph.nodes.forEach((node) => {
    if (activeNodeIds.has(node.id)) {
      activeNodeLayer.append(nodeCircle(node));
      if (node.kind !== "ordinary") labelLayer.append(nodeLabel(node));
    }
  });
  svg.append(activeNodeLayer);
  svg.append(labelLayer);
  svg.append(headLayer);

  state.routeGraphics.forEach((graphic) => {
    const first = graphic.segments[0];
    const point = first ? first.sample.points[0] : graphic.startNode;
    graphic.head.setAttribute("cx", point.x);
    graphic.head.setAttribute("cy", point.y);
  });
  $("#replay").disabled = false;
  $("#play-pause").disabled = false;
}

const MIN_ARROW_EDGE = 34;  // an edge shorter than this cannot hold a legible marker

/**
 * One direction triangle for one physical edge of a branch.
 *
 * The marker is placed at the point of the edge that is furthest away from
 * every node: the node discs are painted on top of the branches, so a marker
 * pinned to a fixed ratio regularly disappeared underneath a node.
 */
function branchArrow(from, to, nodePoints, mode) {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const length = Math.hypot(dx, dy) || 1;
  const ux = dx / length;
  const uy = dy / length;
  let ratio = 0.5;
  let clearance = -1;
  for (let candidate = 0.3; candidate <= 0.7001; candidate += 0.05) {
    const x = from.x + dx * candidate;
    const y = from.y + dy * candidate;
    let nearest = Infinity;
    for (const point of nodePoints) {
      nearest = Math.min(nearest, Math.hypot(point.x - x, point.y - y));
    }
    if (nearest > clearance) {
      clearance = nearest;
      ratio = candidate;
    }
  }
  const x = from.x + dx * ratio;
  const y = from.y + dy * ratio;
  const nx = -uy;
  const ny = ux;
  const triangle = [
    [x + ux * 6, y + uy * 6],
    [x - ux * 5 + nx * 4.2, y - uy * 5 + ny * 4.2],
    [x - ux * 5 - nx * 4.2, y - uy * 5 - ny * 4.2]
  ].map((point) => `${point[0].toFixed(1)},${point[1].toFixed(1)}`).join(" ");
  return svgEl("polygon", {points: triangle, class: `diffusion-direction ${mode}`});
}

function renderDiffusionGraph() {
  const data = state.pathData;
  const svg = $("#graph");
  svg.innerHTML = "";
  const coordinates = graphCoordinates(data);
  svg._diffusionCoordinates = coordinates;

  const base = svgEl("g", {"aria-hidden": "true"});
  data.graph.edges.forEach((edge) => {
    const a = coordinates.get(edge.source), b = coordinates.get(edge.target);
    base.append(svgEl("line", {x1: a.x, y1: a.y, x2: b.x, y2: b.y, class: "base-edge"}));
  });
  svg.append(base);

  const stateLayer = svgEl("g", {id: "diffusion-state-layer"});
  svg.append(stateLayer);

  const forcedLayer = svgEl("g", {"aria-label": "永久选中的起点被迫段"});
  (data.diffusion.forced_edges || []).forEach(([source, target]) => {
    const a = coordinates.get(source), b = coordinates.get(target);
    forcedLayer.append(svgEl("line", {x1: a.x, y1: a.y, x2: b.x, y2: b.y, class: "forced-edge"}));
  });
  svg.append(forcedLayer);

  const nodes = svgEl("g");
  const decisionShapes = new Map();
  data.graph.nodes.forEach((node) => {
    const point = coordinates.get(node.id);
    const radius = node.kind === "ordinary" ? 5 : 8;
    const circle = svgEl("circle", {cx: point.x, cy: point.y, r: radius, class: `node ${node.kind}`});
    nodes.append(circle);
    if (node.kind === "decision") decisionShapes.set(node.id, circle);
    if (node.kind !== "ordinary") {
      const label = svgEl("text", {x: point.x, y: point.y - 13, class: "node-label"});
      label.textContent = node.kind === "start" ? `S · ${node.id}` : node.kind === "goal" ? `G · ${node.id}` : node.id;
      nodes.append(label);
    }
  });
  svg.append(nodes);
  // The per-frame decision state (lit / dimmed) is painted onto these circles.
  state.diffusionNodeShapes = decisionShapes;

  const frames = data.diffusion.frames;
  $("#diffusion-step").max = Math.max(0, frames.length - 1);
  $("#diffusion-replay").disabled = !frames.length;
  $("#diffusion-play").disabled = !frames.length;
  renderDiffusionFrame(state.diffusionFrame);
}

function renderDiffusionFrame(index) {
  const data = state.pathData?.diffusion;
  const layer = $("#diffusion-state-layer");
  const coordinates = $("#graph")._diffusionCoordinates;
  if (!data || !layer || !coordinates || !data.frames.length) return;
  state.diffusionFrame = Math.max(0, Math.min(index, data.frames.length - 1));
  const frame = data.frames[state.diffusionFrame];
  const mode = state.diffusionMode;
  const selected = frame[mode] || [];
  const candidates = new Map(data.candidates.map((candidate) => [candidate.index, candidate]));

  layer.innerHTML = "";
  const chosen = new Map();
  selected.forEach((candidateIndex) => {
    const candidate = candidates.get(candidateIndex);
    if (candidate && !chosen.has(candidate.owner)) chosen.set(candidate.owner, candidate);
  });

  // A fork that answers NULL is not taking any action at this step, so it stays
  // dark; only the forks that actually committed to a Branch light up.
  if (state.diffusionNodeShapes) {
    state.diffusionNodeShapes.forEach((circle, owner) => {
      const candidate = chosen.get(owner);
      const status = !candidate ? "" : candidate.is_null ? "is-null" : `is-active ${mode}`;
      circle.setAttribute("class", `node decision ${status}`.trim());
    });
  }

  const nodePoints = [...coordinates.values()];
  let branchCount = 0;
  let nullCount = 0;
  chosen.forEach((candidate, owner) => {
    if (candidate.is_null) {
      nullCount += 1;
      return;
    }
    branchCount += 1;
    const ownerPoint = coordinates.get(owner);
    if (ownerPoint) layer.append(svgEl("circle", {cx: ownerPoint.x, cy: ownerPoint.y, r: 11, class: `decision-choice-ring ${mode}`}));
    const branchNodes = candidate.edges.length
      ? [candidate.edges[0][0], ...candidate.edges.map((edge) => edge[1])]
      : [];
    const points = branchNodes.map((node) => coordinates.get(node)).filter(Boolean);
    if (points.length < 2) return;
    layer.append(svgEl("path", {d: pathD(points), class: `diffusion-edge ${mode}`, "stroke-width": 4.1}));

    // Every physical edge of the branch carries its own direction marker, so a
    // long branch reads correctly even when only part of it is on screen.
    let arrows = 0;
    let longest = 0;
    let longestLength = -1;
    for (let step = 0; step < points.length - 1; step += 1) {
      const from = points[step];
      const to = points[step + 1];
      const length = Math.hypot(to.x - from.x, to.y - from.y);
      if (length > longestLength) {
        longestLength = length;
        longest = step;
      }
      if (length < MIN_ARROW_EDGE) continue;
      layer.append(branchArrow(from, to, nodePoints, mode));
      arrows += 1;
    }
    // A branch made only of very short edges still gets one marker.
    if (arrows === 0) layer.append(branchArrow(points[longest], points[longest + 1], nodePoints, mode));
  });

  const caption = svgEl("text", {x: 24, y: 28, class: "diffusion-caption"});
  caption.textContent = mode === "clean"
    ? `t=${frame.t} · 模型预测 ẑ₀（每步 argmax，仅供诊断）→ ${statusText(frame.clean_status)}`
    : `t=${frame.t} → ${frame.t - 1} · 扩散链实际状态 zₜ₋₁（仅诊断，single 解码不跟随它）→ ${statusText(frame.noisy_status)}`;
  layer.append(caption);
  const detail = svgEl("text", {x: 24, y: 47, class: "diffusion-caption muted"});
  detail.textContent = `${chosen.size} 个分叉节点各选 1 项 · ${branchCount} 条 Branch（节点亮起）· ${nullCount} 个 NULL（节点调暗）` +
    ` · 本帧解码：clean→${statusText(frame.clean_status)} / noisy→${statusText(frame.noisy_status)}`;
  layer.append(detail);

  $("#diffusion-step").value = state.diffusionFrame;
  $("#diffusion-step-label").textContent = `t = ${frame.t} → ${frame.t - 1}`;
  const change = mode === "clean" ? frame.clean_changed : frame.noisy_changed;
  const changeText = change == null ? "首帧" : `${change} 个决策改变`;
  const expectedDecisions = state.pathData.sample.num_decisions;
  $("#path-status").textContent = `Reverse step ${state.diffusionFrame + 1}/${data.T} · ${changeText} · ${chosen.size}/${expectedDecisions} 个路口唯一选择 · clean 平均置信度 ${fmt(frame.mean_confidence, 3)}`;
}

function stopDiffusion() {
  if (state.diffusionTimer) clearInterval(state.diffusionTimer);
  state.diffusionTimer = null;
  state.diffusionPlaying = false;
}

function startDiffusion(reset = false) {
  stopDiffusion();
  const frames = state.pathData?.diffusion?.frames || [];
  if (!frames.length) return;
  if (reset || state.diffusionFrame >= frames.length - 1) state.diffusionFrame = 0;
  renderDiffusionFrame(state.diffusionFrame);
  state.diffusionPlaying = true;
  $("#diffusion-play").textContent = "暂停";
  state.diffusionTimer = setInterval(() => {
    if (state.diffusionFrame >= frames.length - 1) {
      stopDiffusion();
      $("#diffusion-play").textContent = "重播";
      return;
    }
    renderDiffusionFrame(state.diffusionFrame + 1);
  }, 260);
}

function toggleDiffusion() {
  if (state.diffusionPlaying) {
    stopDiffusion();
    $("#diffusion-play").textContent = "继续";
  } else {
    startDiffusion(false);
  }
}

function setDiffusionMode(mode) {
  state.diffusionMode = mode;
  ["clean", "noisy"].forEach((name) => {
    const button = $(`#state-${name}`);
    const selected = name === mode;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", selected);
  });
  renderDiffusionFrame(state.diffusionFrame);
}

function setGraphView(view) {
  if (view === "diffusion" && !state.pathData?.diffusion) return;
  state.view = view;
  stopAnimation();
  stopDiffusion();
  ["path", "diffusion"].forEach((name) => {
    const button = $(`#view-${name}`);
    const selected = name === view;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", selected);
  });
  $("#path-playback").hidden = view !== "path";
  $("#gt-switch").hidden = view !== "path";
  $("#path-legend").hidden = view !== "path";
  $("#diffusion-toolbar").hidden = view !== "diffusion";
  if (view === "diffusion") {
    $("#view-hint").textContent = "显示所有分叉节点的选择：选到 Branch 的路口会亮起并用外圈标出发起点，选 NULL 的路口调暗；每条边的三角箭头标出方向。每个路口严格只选一项。";
    renderDiffusionGraph();
    startDiffusion(true);
  } else if (state.pathData) {
    $("#view-hint").textContent = "分叉路线使用独立颜色；合流后仍保持紧凑的多色总线。";
    renderPathInfo();
    renderGraph();
    startAnimation(true);
  }
}

function stopAnimation() {
  if (state.animation) cancelAnimationFrame(state.animation);
  state.animation = null;
}

function drawAnimation(elapsed) {
  const speed = Number($("#speed").value);
  const adjusted = elapsed * speed;
  const progress = Math.min(adjusted / state.duration, 1);
  const maxLength = Math.max(1, ...state.routeGraphics.map((graphic) => graphic.length));
  const traveled = maxLength * progress;
  state.routeGraphics.forEach((graphic) => {
    const own = Math.min(graphic.length, traveled);
    let point = graphic.startNode;
    graphic.segments.forEach((segment) => {
      const local = Math.max(0, Math.min(segment.sample.total, own - segment.start));
      const head = sweepSegment(segment, local);
      if (local > 0) point = head;
    });
    graphic.head.setAttribute("cx", point.x);
    graphic.head.setAttribute("cy", point.y);
  });
  $("#timeline-progress").style.width = `${progress * 100}%`;
  if (progress >= 1) {
    stopAnimation();
    state.paused = false;
    $("#play-pause").textContent = "重播";
  }
  return progress;
}

function animationFrame(now) {
  if (state.paused) return;
  const elapsed = state.elapsedBeforePause + (now - state.startedAt);
  if (drawAnimation(elapsed) < 1) state.animation = requestAnimationFrame(animationFrame);
}

function startAnimation(reset = false) {
  stopAnimation();
  if (reset) {
    state.elapsedBeforePause = 0;
    drawAnimation(0);
  }
  state.paused = false;
  state.startedAt = performance.now();
  $("#play-pause").textContent = "暂停";
  state.animation = requestAnimationFrame(animationFrame);
}

function toggleAnimation() {
  if (!state.routeGraphics.length) return;
  if (!state.animation && !state.paused) { startAnimation(true); return; }
  if (state.paused) { startAnimation(false); return; }
  const now = performance.now();
  state.elapsedBeforePause += now - state.startedAt;
  state.paused = true;
  stopAnimation();
  $("#play-pause").textContent = "继续";
}

function selectedModels() {
  return new Set($$("#model-filters input:checked").map((input) => input.value));
}

function visibleMetrics() {
  const models = selectedModels();
  const dataset = $("#metric-dataset").value;
  const decode = $("#metric-decode").value;
  return state.catalog.metrics.filter((row) =>
    models.has(row.model) &&
    (dataset === "all" || row.dataset === dataset) &&
    (decode === "all" || (decode === "single" ? row.decoding === "single" : row.decoding.startsWith("multi")))
  );
}

function fmt(value, digits = 3) {
  return value == null ? "—" : Number(value).toFixed(digits);
}

function renderMetrics() {
  if (!state.catalog) return;
  const rows = visibleMetrics();
  $("#metric-combinations").textContent = rows.length;
  $("#metric-model-count").textContent = new Set(rows.map((row) => row.model)).size;
  $("#metric-dataset-count").textContent = new Set(rows.map((row) => row.dataset)).size;
  const metric = $("#metric-name").value || "goal_hit_rate";
  $("#chart-title").textContent = METRICS[metric];
  renderMetricChart(rows, metric);
  const body = $("#metrics-body");
  const kindOf = (modelId) => (state.catalog.models.find((m) => m.id === modelId) || {}).kind || "unweighted";
  body.innerHTML = rows.map((row) => `<tr>
    <td>${escapeHtml(row.model)}</td>
    <td><em class="kind-badge ${kindOf(row.model)}">${KIND_LABEL[kindOf(row.model)] || "—"}</em></td>
    <td>${escapeHtml(row.dataset.replace(".pkl", ""))}</td><td class="muted">${escapeHtml(row.decoding)}</td>
    <td class="numeric">${fmt(row.goal_hit_rate)}</td><td class="numeric">${fmt(row.optimal_path_rate)}</td>
    <td class="numeric">${fmt(row.success_cost_ratio)}</td><td class="numeric">${fmt(row.loop_rate)}</td>
    <td class="numeric">${fmt(row.broken_rate)}</td><td class="numeric">${fmt(row.coverage_rate)}</td>
    <td class="numeric">${fmt(row.weighted_optimal_coverage_rate)}</td>
    <td class="numeric">${row.mean_goal_paths == null ? "—" : Number(row.mean_goal_paths).toFixed(2)}</td>
  </tr>`).join("") || `<tr><td colspan="12" class="muted">当前筛选条件没有已有评测结果</td></tr>`;
}

function modelColor(model) {
  const models = state.catalog.models.map((item) => item.id);
  return ROUTE_COLORS[Math.max(0, models.indexOf(model)) % ROUTE_COLORS.length];
}

function renderMetricChart(rows, metric) {
  const root = $("#metric-chart");
  root.innerHTML = "";
  const usable = rows.filter((row) => row[metric] != null);
  if (!usable.length) {
    root.innerHTML = `<p class="muted">该指标暂无可展示的评测记录。</p>`;
    return;
  }
  const datasets = [...new Set(usable.map((row) => row.dataset))];
  const width = Math.max(760, root.clientWidth || 1000);
  const panelHeight = 185;
  const height = panelHeight * datasets.length + 20;
  const svg = svgEl("svg", {viewBox: `0 0 ${width} ${height}`, height});
  const left = 58, right = 20, top = 34, bottom = 48;
  const isRatio = metric === "success_cost_ratio";
  const maxValue = isRatio ? Math.max(1.05, ...usable.map((row) => row[metric])) : 1;

  datasets.forEach((dataset, panelIndex) => {
    const panelRows = usable.filter((row) => row.dataset === dataset);
    const yBase = panelIndex * panelHeight;
    const plotTop = yBase + top, plotBottom = yBase + panelHeight - bottom;
    const plotWidth = width - left - right;
    [0, .25, .5, .75, 1].forEach((fraction) => {
      const y = plotBottom - fraction * (plotBottom - plotTop);
      svg.append(svgEl("line", {x1: left, x2: width - right, y1: y, y2: y, class: "chart-grid"}));
      const tick = svgEl("text", {x: left - 9, y: y + 4, "text-anchor": "end", class: "chart-axis"});
      tick.textContent = (fraction * maxValue).toFixed(isRatio ? 2 : 1);
      svg.append(tick);
    });
    const title = svgEl("text", {x: left, y: yBase + 19, class: "chart-label"});
    title.textContent = dataset.replace(".pkl", "").replaceAll("_", " ");
    svg.append(title);
    const gap = 10;
    const barWidth = Math.max(16, Math.min(58, (plotWidth - gap * (panelRows.length - 1)) / panelRows.length));
    const groupWidth = panelRows.length * barWidth + (panelRows.length - 1) * gap;
    const startX = left + Math.max(0, (plotWidth - groupWidth) / 2);
    panelRows.forEach((row, index) => {
      const value = row[metric];
      const h = (value / maxValue) * (plotBottom - plotTop);
      const x = startX + index * (barWidth + gap);
      const rect = svgEl("rect", {x, y: plotBottom - h, width: barWidth, height: h, rx: 3, fill: modelColor(row.model), class: "bar"});
      const tooltip = svgEl("title");
      tooltip.textContent = `${row.model} · ${row.decoding}\n${METRICS[metric]}: ${fmt(value, 4)}`;
      rect.append(tooltip);
      svg.append(rect);
      const valueLabel = svgEl("text", {x: x + barWidth / 2, y: Math.max(plotTop + 10, plotBottom - h - 7), class: "bar-value"});
      valueLabel.textContent = fmt(value);
      svg.append(valueLabel);
      const label = svgEl("text", {x: x + barWidth / 2, y: plotBottom + 15, class: "chart-axis", "text-anchor": "middle"});
      const shortName = row.model.split("/").at(-1).replace("v2_", "");
      label.textContent = shortName.length > 13 ? `${shortName.slice(0, 12)}…` : shortName;
      svg.append(label);
      const decodeLabel = svgEl("text", {x: x + barWidth / 2, y: plotBottom + 30, class: "chart-axis", "text-anchor": "middle"});
      decodeLabel.textContent = row.decoding === "single" ? "单路径" : row.decoding === "multi" ? "多分支" : row.decoding.replace("multi", "").trim();
      svg.append(decodeLabel);
    });
  });
  root.append(svg);
}

/* ------------------------------------------------------------------
 * 实验报告面板：读取 /api/reports/<id>，Markdown 用一个小渲染器画出来，
 * JSON 汇总按原样格式化展示（数据本身在 outputs/*.json，面板只做只读呈现）。
 * ------------------------------------------------------------------ */
function inlineMd(text) {
  return escapeHtml(text)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
}

function renderMarkdown(markdown) {
  const lines = String(markdown || "").split(/\r?\n/);
  const out = [];
  const isTableRow = (line) => /^\s*\|.*\|\s*$/.test(line);
  const isList = (line) => /^\s*([-*]|\d+\.)\s+/.test(line);
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i += 1; continue; }
    if (line.startsWith("```")) {
      const buffer = []; i += 1;
      while (i < lines.length && !lines[i].startsWith("```")) { buffer.push(lines[i]); i += 1; }
      i += 1;
      out.push(`<pre class="md-code">${escapeHtml(buffer.join("\n"))}</pre>`);
      continue;
    }
    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    if (heading) {
      const level = Math.min(6, heading[1].length + 1);
      out.push(`<h${level}>${inlineMd(heading[2])}</h${level}>`);
      i += 1; continue;
    }
    if (/^---+\s*$/.test(line.trim())) { out.push("<hr>"); i += 1; continue; }
    if (isTableRow(line)) {
      const rows = [];
      while (i < lines.length && isTableRow(lines[i])) { rows.push(lines[i]); i += 1; }
      const cells = (row) => row.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((cell) => cell.trim());
      const head = cells(rows[0]);
      const body = rows.slice(1).filter((row) => !/^\s*\|?[\s:|-]+\|?\s*$/.test(row)).map(cells);
      out.push(`<table class="md-table"><thead><tr>${head.map((cell) => `<th>${inlineMd(cell)}</th>`).join("")}</tr></thead>` +
        `<tbody>${body.map((row) => `<tr>${row.map((cell) => `<td>${inlineMd(cell)}</td>`).join("")}</tr>`).join("")}</tbody></table>`);
      continue;
    }
    if (isList(line)) {
      const items = [];
      while (i < lines.length && isList(lines[i])) { items.push(lines[i].replace(/^\s*([-*]|\d+\.)\s+/, "")); i += 1; }
      out.push(`<ul>${items.map((item) => `<li>${inlineMd(item)}</li>`).join("")}</ul>`);
      continue;
    }
    const paragraph = [];
    while (i < lines.length && lines[i].trim() && !lines[i].startsWith("```") && !isTableRow(lines[i]) && !isList(lines[i])
           && !/^(#{1,4})\s/.test(lines[i]) && !/^---+\s*$/.test(lines[i].trim())) {
      paragraph.push(lines[i]); i += 1;
    }
    out.push(`<p>${inlineMd(paragraph.join(" "))}</p>`);
  }
  return out.join("\n");
}

async function loadReport(reportId) {
  $$("#report-items button").forEach((button) => button.classList.toggle("active", button.dataset.report === reportId));
  $("#report-body").innerHTML = `<p class="muted">加载中…</p>`;
  try {
    const data = await api(`/api/reports/${encodeURIComponent(reportId)}`);
    $("#report-title").textContent = data.label;
    $("#report-path").textContent = data.path;
    $("#report-kind").textContent = data.kind === "markdown" ? "Markdown 报告" : "JSON 汇总";
    $("#report-body").innerHTML = data.kind === "markdown"
      ? renderMarkdown(data.markdown)
      : `<pre class="md-code">${escapeHtml(JSON.stringify(data.json, null, 1))}</pre>`;
  } catch (error) {
    $("#report-body").innerHTML = `<p class="muted">读取失败：${escapeHtml(error.message)}</p>`;
  }
}

function renderReports() {
  const reports = (state.catalog && state.catalog.reports) || [];
  const root = $("#report-items");
  if (!root.dataset.built) {
    root.innerHTML = reports.map((item) => `<li><button type="button" class="report-item" data-report="${escapeHtml(item.id)}" ${item.exists ? "" : "disabled"}>` +
      `<strong>${escapeHtml(item.label)}</strong><small>${escapeHtml(item.path)}` +
      `${item.size_kb == null ? " · 缺失" : ` · ${item.size_kb} KB`}</small></button></li>`).join("")
      || `<li class="muted">还没有生成任何报告产物。</li>`;
    root.dataset.built = "1";
    root.querySelectorAll("button[data-report]").forEach((button) =>
      button.addEventListener("click", () => loadReport(button.dataset.report)));
    const first = reports.find((item) => item.exists);
    if (first) loadReport(first.id);
  }
}

function bindEvents() {
  $$(".tab").forEach((tab) => tab.addEventListener("click", () => {
    $$(".tab").forEach((item) => { item.classList.toggle("active", item === tab); item.setAttribute("aria-selected", item === tab); });
    $$(".tab-panel").forEach((panel) => { panel.hidden = panel.id !== tab.getAttribute("aria-controls"); });
    if (tab.id === "metrics-tab") requestAnimationFrame(renderMetrics);
    if (tab.id === "reports-tab") requestAnimationFrame(renderReports);
  }));
  $("#decode-mode").addEventListener("change", () => { $("#multi-controls").hidden = $("#decode-mode").value !== "multi"; });
  $("#path-dataset").addEventListener("change", updateDatasetInfo);
  $("#generate").addEventListener("click", generatePath);
  $("#random-sample").addEventListener("click", pickRandomSample);
  $("#replay").addEventListener("click", () => startAnimation(true));
  $("#play-pause").addEventListener("click", toggleAnimation);
  $("#speed").addEventListener("change", () => { if (state.routeGraphics.length) startAnimation(true); });
  $("#show-gt").addEventListener("change", () => { const gt = $("#gt-path"); if (gt) gt.hidden = !$("#show-gt").checked; });
  $("#view-path").addEventListener("click", () => setGraphView("path"));
  $("#view-diffusion").addEventListener("click", () => setGraphView("diffusion"));
  $("#state-clean").addEventListener("click", () => setDiffusionMode("clean"));
  $("#state-noisy").addEventListener("click", () => setDiffusionMode("noisy"));
  $("#diffusion-replay").addEventListener("click", () => startDiffusion(true));
  $("#diffusion-play").addEventListener("click", toggleDiffusion);
  $("#diffusion-step").addEventListener("input", (event) => {
    stopDiffusion();
    $("#diffusion-play").textContent = "继续";
    renderDiffusionFrame(Number(event.target.value));
  });
  ["#metric-dataset", "#metric-name", "#metric-decode"].forEach((selector) => $(selector).addEventListener("change", renderMetrics));
  $("#model-filters").addEventListener("change", renderMetrics);
  window.addEventListener("resize", () => { if (!$("#metrics-panel").hidden) renderMetrics(); });
}

bindEvents();
initialize();
