"use strict";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const NS = "http://www.w3.org/2000/svg";
const ROUTE_COLORS = ["#b7ff4a", "#73d7ff", "#ff8c68", "#c59cff", "#ffd166", "#5ee6b8", "#ff6b9a", "#a6b4ff"];
const KIND_LABEL = { didi: "滴滴·带权", weighted: "带权", ablated: "带权·无cost", unweighted: "无权" };

const state = {
  catalog: null,
  pathData: null,
  animation: null,
  startedAt: 0,
  elapsedBeforePause: 0,
  duration: 6000,
  paused: false,
  routeGraphics: [],
  //: 按"完成步数"分组的路线下标（strict 束搜索的时序），供"按搜索顺序"播放用。
  searchGroups: [],
  //: 上一次画到的进度，改视野/玩家切换后用来原位重画，而不是从零重播。
  lastProgress: 0,
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

function toast(message, kind) {
  const box = $("#toast");
  box.textContent = message;
  box.classList.toggle("notice", kind === "notice");
  box.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { box.hidden = true; }, 7000);
}

function showError(message) {
  toast(message, "error");
}

/** 中性提示（例如"已把数据集切到这个模型训练用的那一份"），不该长得像报错。 */
function showNotice(message) {
  toast(message, "notice");
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
 * 模型下拉：标签里直接写清类别（无权 / 带权 / 滴滴·带权 / 带权·无cost）与数据目录。
 *
 * 三个 run 的网络结构完全一样，只有 ``data.weighted`` / ``use_edge_cost`` / 数据来源
 * 不同 —— 光看 run 名字根本分不出把哪个 checkpoint 加载进来了，选错模型会得到一份
 * "看起来正常但口径不对"的路径。
 */
function fillModelSelect(select, models) {
  select.innerHTML = "";
  models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.id;
    const kind = KIND_LABEL[model.kind] || model.kind;
    option.textContent = `${model.id} · ${kind}`;
    const bits = [kind];
    if (model.data_dir) bits.push(`data_dir=${model.data_dir}`);
    if (model.flow_steps) bits.push(`flow_steps=${model.flow_steps}`);
    if (model.live_config) bits.push(model.live_config);
    if (model.source) bits.push(`source=${model.source}`);
    option.title = bits.join(" · ");
    select.append(option);
  });
}

/**
 * 数据集下拉：按 `data/` 下的来源子目录分组（``unweighted`` / ``weighted`` /
 * ``didi/graph/chengdu``），每项的 title 给出仓库内相对路径与体积。
 *
 * 分组标签必须写相对路径而不是"直接父目录名"：滴滴的 split 落在
 * ``data/didi/graph/chengdu/``，只显示 ``chengdu`` 看不出它和合成图有什么本质区别，
 * 而这两类数据的 GT 语义完全不同（真实车辆历史路径 vs 最短路径）。
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
    const bits = [];
    if (item.relative) bits.push(item.relative);
    if (item.size_mb != null) bits.push(`${item.size_mb} MB`);
    if (bits.length) option.title = bits.join("  ·  ");
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

//: 被解码器**接管**的控件：strict 下 NULL 永远不合法、必死 branch 恒被剔除，
//: 所以这两个永远置灰。"每次分叉 / 路径表上限"是**可调的搜索预算**，不置灰。
const RULER_LOCKED_CONTROLS = ["#null-policy", "#filter-dead"];

/** 当前模型有没有多分支标尺。 */
function currentModelRuler() {
  return modelById($("#path-model").value)?.ruler || null;
}

/**
 * 该 run 的评测标尺给出的搜索预算（top_k / beam_width）与来源说明。
 *
 * 没有多分支标尺的 run（两个 controlled 的 config 没写 ``evaluation.decode``）落到
 * evaluate.py 的兜底默认值 —— 但**仍然是 strict**：面板只有一套解码语义，
 * 不会因为 config 没写就偷偷换一把尺子。
 */
function rulerBudget() {
  const ruler = currentModelRuler();
  if (ruler && ruler.multi) {
    return {
      top_k: Number(ruler.top_k),
      beam_width: Number(ruler.beam_width),
      source: `${ruler.declared ? "config" : "默认"}${ruler.source ? " · " + ruler.source : ""}`,
      multi: true
    };
  }
  return {top_k: 2, beam_width: 64, source: "", multi: false};
}

/**
 * 把「每次分叉 / 路径表上限」铺成该 run 的评测标尺值。
 *
 * 只在**换模型**和初始化时调用 —— 放进入 refreshRulerControls 会把用户手调的数字
 * 每次刷新都抹掉，那就等于没得调。
 */
function applyRulerDefaults() {
  const budget = rulerBudget();
  $("#top-k").value = String(budget.top_k);
  $("#beam-width").value = String(budget.beam_width);
  refreshRulerControls();
}

/**
 * 刷新口径提示。
 *
 * 面板只有一套解码语义（strict 三池），所以这里不再有"切口径"这回事，只报告
 * **实际会用的**分叉/路径表上限，以及它是否还等于该 run 的评测标尺。
 * 面板/报告/评测口径不一致是踩过的坑：调整过就必须写在脸上，不能让人对着图猜。
 */
function refreshRulerControls() {
  const note = $("#ruler-note");
  RULER_LOCKED_CONTROLS.forEach((selector) => { $(selector).disabled = true; });
  const budget = rulerBudget();
  const topK = Number($("#top-k").value);
  const beam = Number($("#beam-width").value);
  let text = `strict ${topK}/${beam}`;
  if (topK !== budget.top_k || beam !== budget.beam_width) {
    text += `（手动覆盖；该 run 标尺 ${budget.top_k}/${budget.beam_width}）`;
  } else if (budget.multi) {
    text += `（${budget.source}）`;
  } else {
    text += "（该 run 没有多分支标尺，用默认值）";
  }
  note.textContent = text;
  note.title = budget.multi
    ? `固定 strict 三池解码：NULL / loop / dead-end 在 top-k 之前 mask、失败路径淘汰。`
      + `分叉/路径表上限默认取 ${budget.source} 的 ${budget.top_k}/${budget.beam_width}，`
      + "手动改过就不再等于该 run 的评测标尺。"
    : "固定 strict 三池解码。该 run 的 config 没写 evaluation.decode，按默认 strict 2/64 跑。";
}

function modelById(id) {
  return state.catalog.models.find((model) => model.id === id) || null;
}

/**
 * 该模型**真正训过**的那份数据里最合适的 split。
 *
 * 匹配靠 ``data_dir``（来自**当前** configs/<run>.yaml，不是训练时的快照 —— 见
 * dashboard/server.py 的 ``_live_run_settings``）：先看 ``test_1000``（GDP 风格固定
 * 子集，取一条样本就能和主表对上），再退 ``test``。
 *
 * 为什么需要它：模型下拉和数据集下拉是独立的，选 ``didi_chengdu`` 却留着
 * ``unweighted_test.pkl`` 会拿一份合成图去喂真实路网模型 —— 图和特征维度都对不上，
 * 只会得到一句"推理未完成"。
 */
function datasetForModel(model) {
  if (!model || !model.data_dir) return "";
  const dir = model.data_dir.replaceAll("\\", "/").replace(/\/+$/, "");
  const inDir = state.catalog.datasets.filter((item) =>
    (item.relative || "").replaceAll("\\", "/").startsWith(dir + "/"));
  if (!inDir.length) return "";
  const byPreference = (needle) => inDir.find((item) => item.id.includes(needle))?.id || "";
  return byPreference("test_1000") || byPreference("_test.pkl") || byPreference("test") || inDir[0].id;
}

/** 切换模型时把数据集跟过去（只在明显不匹配时动手，并且让用户看得见）。 */
function syncDatasetToModel(announce = true) {
  const model = modelById($("#path-model").value);
  const wanted = datasetForModel(model);
  if (!wanted || wanted === $("#path-dataset").value) return;
  const current = state.catalog.datasets.find((item) => item.id === $("#path-dataset").value);
  const dir = (model?.data_dir || "").replaceAll("\\", "/");
  if (dir && (current?.relative || "").replaceAll("\\", "/").startsWith(dir + "/")) return;
  $("#path-dataset").value = wanted;
  if (announce) showNotice(`已把数据集切到 ${wanted}（${model.id} 训练用的就是这一份）`);
  updateDatasetInfo();
}

async function initialize() {
  try {
    state.catalog = await api("/api/catalog");
    $("#connection").classList.add("online");
    $("#connection span").textContent = `${state.catalog.models.length} 个模型 · ${state.catalog.datasets.length} 个数据集`;
    fillModelSelect($("#path-model"), state.catalog.models);
    fillDatasetSelect($("#path-dataset"), state.catalog.datasets);
    // 默认落在无权合成模型上：它最小、CPU 上也能秒出，先让人看到面板是活的。
    // 想看滴滴就切到 didi_chengdu，数据集会自动跟过去。
    $("#path-model").value = preferred(state.catalog.models, "controlled_unweighted");
    $("#path-dataset").value =
      datasetForModel(modelById($("#path-model").value)) ||
      preferred(state.catalog.datasets, "unweighted_test.pkl");
    applyRulerDefaults();
    await updateDatasetInfo();
  } catch (error) {
    $("#connection span").textContent = "连接失败";
    showError(error.message);
  }
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[char]));
}

/**
 * 数字格式化。**这是文件作用域的函数，不要顺手删掉**：2026-09-16 删指标页时它被
 * 一起删了，而 `renderDiffusionFrame()` 的最后一行还在用它 —— 于是每一帧都抛
 * ReferenceError，异常从 `renderDiffusionGraph()` 里冒出来，把紧随其后的
 * `startDiffusion()` 一起吞掉，最终表现成"扩散过程不能自动播放、只能拖时间轴"。
 * `tests/test_dashboard.py` 里有一条作用域检查专门守这类"调用了但已不存在"。
 */
function fmt(value, digits = 3) {
  return value == null ? "—" : Number(value).toFixed(digits);
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
      // 面板固定 strict 三池口径（历史口径已移除）；top_k / beam_width 可手动覆盖，
      // 后端会照用并按 MULTI_LIMITS 夹取，note 里会标注是否偏离该 run 的评测标尺。
      ruler: "ruler",
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
  const isReal = sample.source === "didi_chengdu";
  $("#sample-title").textContent = isReal
    ? `#${data.index} · 滴滴 ${sample.date || ""}`
    : `#${data.index} · ${sample.start} → ${sample.goal}`;
  // 真实数据没有 difficulty / mode（那是合成图生成器的标签），换成司机绕行比 ——
  // "这条 GT 比最短路多走了多少"，它才是真实数据里真正影响难度的量。
  const detour = sample.gt_cost_ratio;
  const values = [
    isReal
      ? `真实车辆路径${detour ? ` · 绕行 ×${Number(detour).toFixed(2)}` : ""}`
      : `${sample.difficulty} / ${sample.mode}`,
    `${sample.num_nodes} / ${sample.num_decisions}`,
    isReal && sample.gt_cost != null
      ? `${sample.gt_length} 跳 / ${(Number(sample.gt_cost) / 1000).toFixed(2)} km`
      : `${sample.gt_length} 跳`,
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
    // rank 是**概率序**（success 池最后按 log_prob 重排过），不是到达顺序；
    // 搜索时序必须单独标出来，否则"#1"会被读成"最先到达"。
    const timing = route.depth != null
      ? ` · 第 ${route.found_index ?? "?"} 个完成（搜索第 ${route.depth} 步）`
      : "";
    item.innerHTML = `<i style="background:${routeColor(index)}"></i><span>#${route.rank} · ${label} · ${costText}${timing}</span>`;
    legend.append(item);
  });
  const reached = data.routes.filter((route) => route.status === "goal").length;
  const bits = [
    `${data.routes.length} 条可视路线`,
    `${reached} 条到达终点`,
    `checkpoint epoch ${data.checkpoint_epoch ?? "—"}`
  ];
  if (isReal) bits.push(`corridor rho=${sample.rho ?? "—"}`);
  if (sample.order_id) bits.push(`order ${String(sample.order_id).slice(0, 8)}`);
  // 这张图到底是按哪把尺子解出来的 —— 面板/报告/评测口径不一致是踩过的坑
  if (data.ruler && data.ruler.note) bits.push(`口径 ${data.ruler.note}`);
  $("#path-status").textContent = bits.join(" · ");
}

/**
 * 画布几何。**必须和后端 dashboard/server.py 的 PANEL_WIDTH / PANEL_HEIGHT /
 * PANEL_MARGIN 完全一致**：地理模式下后端就是按这个长宽比做 letterbox 的，两边一旦
 * 漂移，真实路网就会被拉伸（成都经度方向已经被 cos(30.7°) 缩短了 19%，再拉一次就
 * 完全不像地图了）。
 */
const GRAPH_MARGIN = 55;
const GRAPH_WIDTH = 1000;
const GRAPH_HEIGHT = 620;
const INNER_WIDTH = GRAPH_WIDTH - GRAPH_MARGIN * 2;
const INNER_HEIGHT = GRAPH_HEIGHT - GRAPH_MARGIN * 2;

function panelX(x) { return GRAPH_MARGIN + x * INNER_WIDTH; }
function panelY(y) { return GRAPH_MARGIN + (1 - y) * INNER_HEIGHT; }

function graphCoordinates(data) {
  return new Map(data.graph.nodes.map((node) => [node.id, {
    x: panelX(node.x),
    y: panelY(node.y),
    kind: node.kind
  }]));
}

function graphIsGeo() {
  return state.pathData?.graph?.geo === true;
}

/**
 * 节点半径。
 *
 * 真实 corridor 有 900~2900 个节点、绝大多数是 decision：画大了会直接糊成一片白点，
 * 把路线和真实路网都盖住。所以：普通节点不画、decision 画小圈、只有 S/G 保持醒目；
 * 合成图也同步缩小（3.5 / 5.5）。
 */
function nodeRadius(kind, active = false) {
  if (!graphIsGeo()) return kind === "ordinary" ? 3.5 : 5.5;
  if (kind === "start" || kind === "goal") return 5;
  if (kind === "decision") return active ? 2.2 : 1.6;
  return active ? 1.6 : 0;
}

/** 地理模式下 200 个 decision 标签会盖满整张图；只留 S/G。 */
function shouldLabelNode(kind) {
  if (graphIsGeo()) return kind === "start" || kind === "goal";
  return kind !== "ordinary";
}

function nodeLabelText(kind, id) {
  return kind === "start" ? `S · ${id}` : kind === "goal" ? `G · ${id}` : String(id);
}

/** 真实路网底图：不走 graphCoordinates —— 街道两端坐标由后端算好，不带节点 id。 */
function renderStreetBasemap(layer, graph) {
  (graph.street_edges || []).forEach((edge) => {
    layer.append(svgEl("line", {
      x1: panelX(edge[0]).toFixed(2), y1: panelY(edge[1]).toFixed(2),
      x2: panelX(edge[2]).toFixed(2), y2: panelY(edge[3]).toFixed(2),
      class: "street-edge"
    }));
  });
}

/**
 * 比例尺。后端给的是"归一化 x 方向的整幅宽度 = 多少 km"（等距圆柱投影，已经按
 * cos(lat0) 校正，x/y 同尺度），换算成用户单位后挑一个好读的整数距离。
 *
 * 没有它的话，"真实经纬度底图"就只是一堆灰线 —— 看不出这段路是 300 米还是 3 公里。
 */
function renderScaleBar(svg, geo) {
  const kmPerUnit = geo.km_per_x_unit / INNER_WIDTH;
  if (!(kmPerUnit > 0)) return;
  const NICE_KM = [0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 100];
  const target = 150;                       // 想要的比例尺长度（用户单位）
  let km = NICE_KM[NICE_KM.length - 1];
  for (const candidate of NICE_KM) {
    if (candidate / kmPerUnit <= target * 1.35) { km = candidate; break; }
  }
  const length = km / kmPerUnit;
  if (!(length > 8 && length < INNER_WIDTH)) return;
  const x = GRAPH_MARGIN + 8;
  const y = GRAPH_HEIGHT - GRAPH_MARGIN + 18;
  const group = svgEl("g", {class: "scale-bar", "aria-hidden": "true"});
  group.append(svgEl("line", {x1: x, y1: y, x2: x + length, y2: y}));
  group.append(svgEl("line", {x1: x, y1: y - 5, x2: x, y2: y + 5}));
  group.append(svgEl("line", {x1: x + length, y1: y - 5, x2: x + length, y2: y + 5}));
  const label = svgEl("text", {x: x + length / 2, y: y - 9, "text-anchor": "middle"});
  label.textContent = km >= 1 ? `${km} km` : `${Math.round(km * 1000)} m`;
  group.append(label);
  svg.append(group);
}

/** 图下方的底图来源说明（哪份 OSMnx 文件、覆盖率、画面多少公里）。 */
function renderGraphCaption(data) {
  const caption = $("#graph-caption");
  const graph = data.graph;
  const geoHint = $("#geo-hint");
  if (!graph.geo) {
    caption.hidden = true;
    if (geoHint) geoHint.hidden = true;
    return;
  }
  const crop = graph.crop_km || [];
  const bits = [
    "真实经纬度底图（OSMnx）",
    escapeHtml(graph.source || ""),
    `覆盖 ${(100 * (graph.coverage || 0)).toFixed(1)}%`,
    crop.length === 2 ? `画面 ${crop[0].toFixed(1)} × ${crop[1].toFixed(1)} km` : "",
    `${(graph.street_edges || []).length} 条街道${graph.street_truncated ? "（已截断）" : ""}`
  ].filter(Boolean);
  caption.innerHTML = bits.join("  ·  ");
  caption.hidden = false;
  if (geoHint) geoHint.hidden = false;
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
  // 注意：这里刻意**不用** fmt 这个名字 —— 文件里还有一个同名的全局格式化函数，
  // 局部遮蔽会让"这一行到底调的哪个"全靠上下文猜
  const coord = (value) => value.toFixed(2);
  return {
    d: `M${coord(a.x)},${coord(a.y)} C${coord(c1x)},${coord(c1y)} ${coord(c2x)},${coord(c2y)} ${coord(b.x)},${coord(b.y)}`,
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
  const geo = data.graph.geo === true;
  const coordinates = graphCoordinates(data);

  const base = svgEl("g", {"aria-hidden": "true"});
  if (geo) renderStreetBasemap(base, data.graph);
  data.graph.edges.forEach((edge) => {
    const a = coordinates.get(edge.source), b = coordinates.get(edge.target);
    base.append(svgEl("line", {
      x1: a.x, y1: a.y, x2: b.x, y2: b.y,
      class: geo ? "base-edge corridor" : "base-edge"
    }));
  });
  svg.append(base);

  const gtPoints = data.sample.gt_path.map((id) => coordinates.get(id)).filter(Boolean);
  const gt = svgEl("path", {d: pathD(gtPoints), class: "gt-edge", id: "gt-path"});
  gt.hidden = !$("#show-gt").checked;
  svg.append(gt);

  const nodeCircle = (node, active = false) => {
    const point = coordinates.get(node.id);
    return svgEl("circle", {
      cx: point.x, cy: point.y, r: nodeRadius(node.kind, active), class: `node ${node.kind}`
    });
  };
  const nodeLabel = (node) => {
    const point = coordinates.get(node.id);
    const label = svgEl("text", {x: point.x, y: point.y - 13, class: "node-label"});
    label.textContent = nodeLabelText(node.kind, node.id);
    return label;
  };
  // 背景节点圆点：真实 corridor 有 900~2900 个节点，全画成小圆会糊成一片白点，
  // 把路线和路网底图都盖住。默认只画**显示出来的路线经过的节点**（+ S/G）。
  const nodeMode = $("#node-mode") ? $("#node-mode").value : "all";
  const backgroundNodes = svgEl("g", {"aria-hidden": "true"});
  data.graph.nodes.forEach((node) => {
    if (nodeMode === "all" && nodeRadius(node.kind) > 0) backgroundNodes.append(nodeCircle(node));
    if (shouldLabelNode(node.kind)) backgroundNodes.append(nodeLabel(node));
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
  // 搜索时序分组：strict 束搜索按深度逐层展开，route.depth 就是"第几轮被找到"，
  // **同一个 depth 的多条路线是同一轮同时毕业的**。没有 depth（单路径解码 /
  // 历史口径的残骸）就留空，前端自动退回"按长度"播放。
  const byDepth = new Map();
  data.routes.forEach((route, index) => {
    if (route.depth == null) return;
    if (!byDepth.has(route.depth)) byDepth.set(route.depth, []);
    byDepth.get(route.depth).push(index);
  });
  state.searchGroups = [...byDepth.keys()]
    .sort((left, right) => left - right)
    .map((depth) => ({depth, members: byDepth.get(depth)}));
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
      if (nodeMode !== "none") activeNodeLayer.append(nodeCircle(node, true));
      if (shouldLabelNode(node.kind)) labelLayer.append(nodeLabel(node));
    }
  });
  svg.append(activeNodeLayer);
  svg.append(labelLayer);
  svg.append(headLayer);
  if (geo) renderScaleBar(svg, data.graph);

  state.routeGraphics.forEach((graphic) => {
    const first = graphic.segments[0];
    const point = first ? first.sample.points[0] : graphic.startNode;
    graphic.head.setAttribute("cx", point.x);
    graphic.head.setAttribute("cy", point.y);
  });
  renderGraphCaption(data);
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
  if (graphIsGeo()) renderStreetBasemap(base, data.graph);
  data.graph.edges.forEach((edge) => {
    const a = coordinates.get(edge.source), b = coordinates.get(edge.target);
    base.append(svgEl("line", {
      x1: a.x, y1: a.y, x2: b.x, y2: b.y,
      class: graphIsGeo() ? "base-edge corridor" : "base-edge"
    }));
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
    const radius = nodeRadius(node.kind);
    // 地理模式下普通节点的半径是 0：它们只是 corridor 的中间点，画出来只会挡住路网
    if (radius <= 0) return;
    const circle = svgEl("circle", {cx: point.x, cy: point.y, r: radius, class: `node ${node.kind}`});
    nodes.append(circle);
    if (node.kind === "decision") decisionShapes.set(node.id, circle);
    if (shouldLabelNode(node.kind)) {
      const label = svgEl("text", {x: point.x, y: point.y - 13, class: "node-label"});
      label.textContent = nodeLabelText(node.kind, node.id);
      nodes.append(label);
    }
  });
  svg.append(nodes);
  // The per-frame decision state (lit / dimmed) is painted onto these circles.
  state.diffusionNodeShapes = decisionShapes;

  renderGraphCaption(data);
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

//: 扩散播放 1× 时每帧的毫秒数。2×（默认）就是 130ms，T=50 走完约 6.5 秒。
//: 原来写死 260ms 且没有调速入口，T=50 要 13 秒，实测"太慢"。
const DIFFUSION_BASE_MS = 260;

function diffusionDelay() {
  const speed = Number($("#diffusion-speed")?.value || 2) || 1;
  return Math.max(30, Math.round(DIFFUSION_BASE_MS / speed));
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
  }, diffusionDelay());
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

//: "按搜索顺序"播放时，每一组在自己的时间片里长出来的比例（剩下的是停留展示）
const SEARCH_RAMP = 0.55;

/** 把一条路线画到 `own` 这么长的弧距；没开始画的路线连"笔尖"一起藏起来。 */
function paintRoute(graphic, own) {
  let point = graphic.startNode;
  graphic.segments.forEach((segment) => {
    const local = Math.max(0, Math.min(segment.sample.total, own - segment.start));
    const head = sweepSegment(segment, local);
    if (local > 0) point = head;
  });
  graphic.head.setAttribute("cx", point.x);
  graphic.head.setAttribute("cy", point.y);
  // 未开始的路线如果把笔尖留在起点，8 条路线会在起点堆出一排圆点
  graphic.head.style.display = own > 0 ? "" : "none";
}

/**
 * 两种播放口径 —— **画的是同一批结果，区别只在"什么时候出现"**：
 *
 * * `length`（默认）：所有路线**按几何弧长同步**推进。这是"最终结果"的画法，
 *   看不出搜索过程。
 * * `search`：按 strict 束搜索的**完成步数**分组、一组一组出现。同一个 depth 的
 *   多条路线是同一轮里同时毕业的，所以只有这个模式才看得出
 *   "同时只有 beam 条在走、到一条毕业一条"。
 *   没有 depth（单路径解码）时自动退回 length。
 *
 * ⚠️ `rank`（图例里的 #1..#N）是**概率序**，不是到达顺序；到达顺序看 `depth`。
 */
function drawAnimation(elapsed) {
  const speed = Number($("#speed").value);
  const adjusted = elapsed * speed;
  const progress = Math.min(adjusted / state.duration, 1);
  const groups = state.searchGroups;
  const searchMode = $("#anim-mode").value === "search" && groups.length > 0;
  if (searchMode) {
    const slot = progress * groups.length;
    state.routeGraphics.forEach((graphic) => paintRoute(graphic, 0));
    groups.forEach((group, groupIndex) => {
      const local = Math.max(0, Math.min(1, (slot - groupIndex) / SEARCH_RAMP));
      group.members.forEach((index) => {
        const graphic = state.routeGraphics[index];
        if (graphic) paintRoute(graphic, graphic.length * local);
      });
    });
    const shown = Math.min(groups.length, Math.max(1, Math.floor(slot) + 1));
    const active = groups[shown - 1];
    $("#path-status").textContent =
      `按搜索顺序播放：第 ${shown}/${groups.length} 组 · 搜索第 ${active.depth} 步完成` +
      `（该组 ${active.members.length} 条）`;
  } else {
    const maxLength = Math.max(1, ...state.routeGraphics.map((graphic) => graphic.length));
    const traveled = maxLength * progress;
    state.routeGraphics.forEach((graphic) => {
      paintRoute(graphic, Math.min(graphic.length, traveled));
    });
  }
  state.lastProgress = progress;
  $("#timeline-progress").style.width = `${progress * 100}%`;
  if (progress >= 1) {
    stopAnimation();
    state.paused = false;
    $("#play-pause").textContent = "重播";
    // 播放时状态行被"第几组"占用了，播完恢复成常规摘要
    if (searchMode && state.pathData) renderPathInfo();
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
    if (tab.id === "reports-tab") requestAnimationFrame(renderReports);
  }));
  $("#decode-mode").addEventListener("change", () => { $("#multi-controls").hidden = $("#decode-mode").value !== "multi"; });
  $("#path-dataset").addEventListener("change", updateDatasetInfo);
  $("#path-model").addEventListener("change", () => {
    syncDatasetToModel();
    applyRulerDefaults();
  });
  $("#top-k").addEventListener("change", refreshRulerControls);
  $("#beam-width").addEventListener("change", refreshRulerControls);
  $("#ruler-reset").addEventListener("click", applyRulerDefaults);
  $("#generate").addEventListener("click", generatePath);
  $("#random-sample").addEventListener("click", pickRandomSample);
  $("#replay").addEventListener("click", () => startAnimation(true));
  $("#play-pause").addEventListener("click", toggleAnimation);
  $("#speed").addEventListener("change", () => { if (state.routeGraphics.length) startAnimation(true); });
  $("#anim-mode").addEventListener("change", () => { if (state.routeGraphics.length) startAnimation(true); });
  // 换视野 / 换节点显示都不重播：先记下当前进度，重画完再原位填回去
  const redrawKeepingProgress = () => {
    if (!state.pathData || state.view !== "path") return;
    const keep = state.lastProgress > 0 ? state.lastProgress : 1;
    renderGraph();
    drawAnimation(state.duration * keep);
  };
  $("#node-mode").addEventListener("change", redrawKeepingProgress);
  $("#show-gt").addEventListener("change", () => { const gt = $("#gt-path"); if (gt) gt.hidden = !$("#show-gt").checked; });
  $("#view-path").addEventListener("click", () => setGraphView("path"));
  $("#view-diffusion").addEventListener("click", () => setGraphView("diffusion"));
  $("#state-clean").addEventListener("click", () => setDiffusionMode("clean"));
  $("#state-noisy").addEventListener("click", () => setDiffusionMode("noisy"));
  $("#diffusion-replay").addEventListener("click", () => startDiffusion(true));
  $("#diffusion-play").addEventListener("click", toggleDiffusion);
  $("#diffusion-speed").addEventListener("change", () => {
    // 正在播就按新速度重开定时器；startDiffusion(false) 不会把帧号重置回 0，
    // 所以画面不会跳，只是接下来每帧的间隔变了。
    if (state.diffusionPlaying) startDiffusion(false);
  });
  $("#diffusion-step").addEventListener("input", (event) => {
    stopDiffusion();
    $("#diffusion-play").textContent = "继续";
    renderDiffusionFrame(Number(event.target.value));
  });
}

bindEvents();
initialize();
