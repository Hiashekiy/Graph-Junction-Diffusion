/**
 * 在 Node 里用 DOM 桩真正执行 dashboard/static/app.js，走一遍用户的真实操作路径。
 *
 * 为什么必须真的跑一遍：`node --check` 只查语法，引用一个不存在的全局函数是完全合法
 * 的语法，只有执行时才抛 ReferenceError。2026-09-16 删指标页时把 `fmt()` 一起删了，
 * 而 `renderDiffusionFrame()` 最后一行还在用它 —— 于是进入"扩散过程"时
 * `renderDiffusionGraph()` 抛异常，把紧随其后的 `startDiffusion()` 一起吞掉，
 * 表现成"不能自动播放、只能拖时间轴"。
 *
 * 用法：node dashboard_smoke.js <payloads.json>
 *   payloads.json = { "payloads": [ { "name": "...", "payload": {...} }, ... ] }
 * 输出：一行 JSON，{ ok, error, scenarios: [{ name, checks: [...] }] }
 *
 * 每个 payload 都在**全新的 vm context** 里跑，互不污染。
 */
"use strict";

const fs = require("fs");
const path = require("path");
const vm = require("vm");

const payloadsPath = process.argv[2];
const appPath = process.argv[3] || path.join(__dirname, "..", "..", "dashboard", "static", "app.js");
const scenarios = JSON.parse(fs.readFileSync(payloadsPath, "utf8")).payloads;
const source = fs.readFileSync(appPath, "utf8");

// --------------------------------------------------------------- DOM 桩 --
function makeNode(tag) {
  const node = {
    tagName: tag,
    attributes: {},
    children: [],
    hidden: false,
    disabled: false,
    textContent: "",
    innerHTML: "",
    value: "",
    max: "",
    min: "",
    className: "",
    id: "",
    dataset: {},
    style: {},
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    append(...items) { node.children.push(...items); },
    appendChild(item) { node.children.push(item); return item; },
    setAttribute(key, value) { node.attributes[key] = value; },
    getAttribute(key) { return node.attributes[key]; },
    listeners: {},
    addEventListener(type, fn) {
      (node.listeners[type] = node.listeners[type] || []).push(fn);
    },
    removeEventListener() {},
    querySelector() { return makeNode("stub"); },
    querySelectorAll() { return []; },
    getTotalLength() { return 100; },
    getPointAtLength(length) { return { x: length, y: length }; },
    focus() {},
    remove() {}
  };
  return node;
}

function runScenario(scenario) {
  const checks = [];
  const check = (name, condition, detail) => {
    checks.push({
      name: `${scenario.name} · ${name}`,
      ok: Boolean(condition),
      detail: detail === undefined ? "" : String(detail)
    });
  };

  const elements = new Map();
  const elementFor = (selector) => {
    if (!elements.has(selector)) elements.set(selector, makeNode(selector));
    return elements.get(selector);
  };

  const timers = [];
  const sandbox = {
    console, Math, JSON, Object, Array, Number, String, Boolean, Map, Set, Date,
    Promise, Error, RegExp, isNaN, parseInt, parseFloat, Symbol, WeakMap, Intl,
    // 下面这些都是浏览器全局，Node 里没有
    Option: function Option(text, value) { return { text, value }; },
    performance: { now: () => Date.now() },
    setTimeout: (fn, ms) => { timers.push({ fn, ms, kind: "timeout" }); return timers.length; },
    clearTimeout: () => {},
    setInterval: (fn, ms) => { timers.push({ fn, ms, kind: "interval" }); return timers.length; },
    clearInterval: () => {},
    requestAnimationFrame: () => 1,
    cancelAnimationFrame: () => {},
    encodeURIComponent, decodeURIComponent,
    fetch: () => Promise.reject(new Error("offline harness")),
    document: {
      querySelector: (selector) => elementFor(selector),
      querySelectorAll: () => [],
      createElement: (tag) => makeNode(tag),
      createElementNS: (ns, tag) => makeNode(tag),
      addEventListener() {}
    },
    window: { addEventListener() {}, location: { href: "" } },
    navigator: { userAgent: "node" }
  };
  // 让测试能像浏览器那样派发事件（桩元素默认不触发任何监听器）
  sandbox.__fire = (selector, type) => {
    const node = elements.get(selector);
    const handlers = (node && node.listeners && node.listeners[type]) || [];
    handlers.forEach((handler) => handler({ target: node }));
  };
  sandbox.globalThis = sandbox;
  sandbox.self = sandbox;

  const context = vm.createContext(sandbox);
  let error = null;
  let stage = "load";
  const payload = scenario.payload;

  try {
    vm.runInContext(source, context, { filename: appPath });
    check("app.js 加载", true);

    sandbox.__payload = payload;
    sandbox.__catalog = scenario.catalog;
    vm.runInContext("state.pathData = __payload;", context);
    if (scenario.catalog) {
      vm.runInContext(
        "state.catalog = __catalog; document.querySelector('#path-model').value = __catalog.models[0].id;",
        context
      );
    }

    // ---- 1) 路径视图 -----------------------------------------------------
    stage = "renderGraph";
    vm.runInContext("renderGraph();", context);
    check("renderGraph() 不抛异常", true);
    if (payload.graph.geo) {
      check("地理模式：payload 里带了街道线", payload.graph.street_edges.length > 0);
      check("地理模式：底图说明行已显示", elementFor("#graph-caption").hidden === false);
    }

    // ---- 2) 用户报的那条路径：点「扩散过程」 ------------------------------
    stage = "setGraphView";
    vm.runInContext("setGraphView('diffusion');", context);
    check("setGraphView('diffusion') 不抛异常", true);
    check("state.view === 'diffusion'",
          vm.runInContext("state.view", context) === "diffusion");
    // 关键断言：自动播放必须真的启动（这就是被 ReferenceError 吞掉的那一步）
    check("自动播放已启动 state.diffusionPlaying === true",
          vm.runInContext("state.diffusionPlaying", context) === true);
    check("播放按钮显示「暂停」",
          elementFor("#diffusion-play").textContent === "暂停",
          elementFor("#diffusion-play").textContent);
    const intervals = () => timers.filter((item) => item.kind === "interval");
    check("注册了 interval 定时器", intervals().length > 0);
    // 默认 2x -> 130ms；改到 4x 必须让间隔变短（否则"速度"是个摆设）
    const defaultDelay = intervals().slice(-1)[0].ms;
    check("默认间隔是 1x 的一半（130ms）", defaultDelay === 130, defaultDelay);
    elementFor("#diffusion-speed").value = "4";
    const beforeSpeedChange = vm.runInContext("state.diffusionFrame", context);
    vm.runInContext("__fire('#diffusion-speed', 'change');", context);
    const fastDelay = intervals().slice(-1)[0].ms;
    check("调速后间隔变短", fastDelay < defaultDelay, `${defaultDelay} -> ${fastDelay}`);
    check("调速不会跳回第一帧",
          vm.runInContext("state.diffusionFrame", context) === beforeSpeedChange);
    // 后面的推进用新定时器（旧的那个已经被 clearInterval 掉了，桩里不区分，
    // 但两者回调是同一个函数，推进效果一致）
    elementFor("#diffusion-speed").value = "2";

    // ---- 3) 推进计时器：帧必须前进 ---------------------------------------
    stage = "tick";
    const frames = payload.diffusion.frames.length;
    const interval = timers.filter((item) => item.kind === "interval").pop();
    const before = vm.runInContext("state.diffusionFrame", context);
    for (let index = 0; index < Math.min(3, frames - 1); index += 1) interval.fn();
    const after = vm.runInContext("state.diffusionFrame", context);
    check("帧号会前进", after > before, `${before} -> ${after}`);
    check("滑块跟着走", String(elementFor("#diffusion-step").value) === String(after),
          elementFor("#diffusion-step").value);
    check("t 标签有更新", /t = \d+/.test(elementFor("#diffusion-step-label").textContent),
          elementFor("#diffusion-step-label").textContent);

    // ---- 4) 播到末尾：停下来并显示「重播」 --------------------------------
    stage = "playToEnd";
    for (let index = 0; index < frames + 5; index += 1) interval.fn();
    check("播到末尾后停止", vm.runInContext("state.diffusionPlaying", context) === false);
    check("末尾按钮显示「重播」",
          elementFor("#diffusion-play").textContent === "重播",
          elementFor("#diffusion-play").textContent);
    check("停在最后一帧",
          vm.runInContext("state.diffusionFrame", context) === frames - 1,
          vm.runInContext("state.diffusionFrame", context));

    // ---- 5) 解码口径：只剩 strict 标尺，分叉 / 路径表上限可调 ----------------
    // 历史（存活路径表）口径已移除；"每次分叉 / 路径表上限"默认铺成该 run 的评测标尺值，
    // 但**不置灰** —— 调过之后提示必须写明"手动覆盖"，否则又会变成对着一个
    // 不是评测口径的数字下结论。被解码器接管的两个控件（NULL 策略 / 必死预筛选）仍置灰。
    if (scenario.catalog) {
      stage = "rulerControls";
      vm.runInContext("applyRulerDefaults();", context);
      const ruler = scenario.catalog.models[0].ruler;
      if (ruler.multi) {
        check("分叉 / 路径表上限铺成标尺值",
              elementFor("#top-k").value === String(ruler.top_k)
              && elementFor("#beam-width").value === String(ruler.beam_width),
              `${elementFor("#top-k").value}/${elementFor("#beam-width").value}`);
        check("分叉 / 路径表上限可编辑",
              elementFor("#top-k").disabled === false
              && elementFor("#beam-width").disabled === false);
        check("被解码器接管的两个控件仍置灰",
              elementFor("#null-policy").disabled === true
              && elementFor("#filter-dead").disabled === true);
        check("提示写清了标尺来源",
              elementFor("#ruler-note").textContent.includes(String(ruler.beam_width)),
              elementFor("#ruler-note").textContent);
        const before = elementFor("#ruler-note").textContent;
        elementFor("#beam-width").value = String(Number(ruler.beam_width) + 5);
        vm.runInContext("__fire('#beam-width', 'change');", context);
        check("手改后提示写明「手动覆盖」",
              elementFor("#ruler-note").textContent.includes("手动覆盖"),
              elementFor("#ruler-note").textContent);
        check("手改确实改了提示（不是死值）",
              elementFor("#ruler-note").textContent !== before,
              elementFor("#ruler-note").textContent);
        vm.runInContext("applyRulerDefaults();", context);
        check("「↺ 标尺值」把预算恢复成标尺值",
              elementFor("#beam-width").value === String(ruler.beam_width)
              && !elementFor("#ruler-note").textContent.includes("手动覆盖"),
              elementFor("#ruler-note").textContent);
      } else {
        check("该 run 没有多分支标尺时用默认 strict 值",
              elementFor("#top-k").value === "2"
              && elementFor("#beam-width").value === "64",
              `${elementFor("#top-k").value}/${elementFor("#beam-width").value}`);
        check("并说明原因",
              elementFor("#ruler-note").textContent.includes("没有多分支标尺"),
              elementFor("#ruler-note").textContent);
      }
    }

    // ---- 6) 按搜索顺序播放 -------------------------------------------------
    // rank 是**概率序**、不是到达顺序；"谁先到达"看 depth（strict 束搜索按深度逐层
    // 展开，所以 depth 就是"搜索第几步完成"）。这个模式必须真的按 depth 分组
    // 一组一组出现，而不是所有路线一起长完。
    stage = "searchPlayback";
    vm.runInContext(`
      const template = state.pathData.routes[0];
      state.pathData.routes = [
        {...template, depth: 10, found_index: 1},
        {...template, depth: 13, found_index: 2},
        {...template, depth: 16, found_index: 3},
      ];
      setGraphView('path');
      document.querySelector('#anim-mode').value = 'search';
      // 桩元素的 value 默认是空串 -> Number('') = 0 -> progress 永远 0，必须先钉住速度
      document.querySelector('#speed').value = '1';
    `, context);
    // 图例项是 **children**（桩元素的 innerHTML 不会被解析），所以读 children 的 innerHTML
    const legendHtml = vm.runInContext(
      "document.querySelector('#path-legend').children.map((item) => item.innerHTML).join(' | ')",
      context);
    check("图例标注了完成顺序（rank 是概率序，不是到达序）",
          legendHtml.includes("个完成（搜索第"), legendHtml.slice(0, 90));
    vm.runInContext("drawAnimation(0);", context);
    check("还没开始的路线连笔尖都不画",
          vm.runInContext("state.routeGraphics.every((g) => g.head.style.display === 'none')", context));
    vm.runInContext("drawAnimation(state.duration * 0.5);", context);
    const revealed = vm.runInContext(
      "state.routeGraphics.filter((g) => g.head.style.display !== 'none').length", context);
    check("只画到当前组，后面的组还没出现", revealed === 2, `${revealed}/3`);
    check("状态行写明第几组",
          elementFor("#path-status").textContent.includes("按搜索顺序播放"),
          elementFor("#path-status").textContent);
    vm.runInContext("drawAnimation(state.duration);", context);
    check("播完恢复常规摘要",
          !elementFor("#path-status").textContent.includes("按搜索顺序播放"),
          elementFor("#path-status").textContent);
    vm.runInContext("document.querySelector('#anim-mode').value = 'length';", context);

    // ---- 7) 节点显示 -----------------------------------------
    // 真实 corridor 上千个节点全画成小圆会糊成一片白点，把路线和路网都盖住。
    stage = "nodeDisplay";
    const countNodeCircles = () => vm.runInContext(`
      (() => {
        let total = 0;
        const walk = (node) => {
          const cls = node.attributes && node.attributes.class ? String(node.attributes.class) : "";
          if (node.tagName === "circle" && cls.split(" ")[0] === "node") total += 1;
          (node.children || []).forEach(walk);
        };
        walk(document.querySelector("#graph"));
        return total;
      })()
    `, context);
    // 桩元素的 innerHTML = "" 不会清空 children（浏览器会），所以每次渲染前手动清一遍，
    // 否则三次渲染的圆点是**累加**的，数出来的数字没有意义。
    const renderWithNodeMode = (mode) => vm.runInContext(
      `document.querySelector('#graph').children.length = 0;
       document.querySelector('#node-mode').value = '${mode}';
       renderGraph();`, context);
    renderWithNodeMode("all");
    const allNodeCircles = countNodeCircles();
    renderWithNodeMode("route");
    const routeNodeCircles = countNodeCircles();
    renderWithNodeMode("none");
    const noNodeCircles = countNodeCircles();
    check("「仅路线」比「全部」少画节点（且不是全不画）",
          routeNodeCircles > 0 && routeNodeCircles < allNodeCircles,
          `route=${routeNodeCircles} all=${allNodeCircles}`);
    check("「全关」一个节点圆点都不画", noNodeCircles === 0, String(noNodeCircles));
    vm.runInContext("document.querySelector('#node-mode').value = 'route'; renderGraph();", context);

    // ---- 8) 切回路径视图 -------------------------------------------------
    stage = "backToPath";
    vm.runInContext("setGraphView('path');", context);
    check("setGraphView('path') 不抛异常", true);
    // 状态行必须印出实际口径 —— 面板/报告/评测口径不一致是踩过的坑，
    // 不能让人对着图猜用的是哪把尺子
    check("状态行印出了实际解码口径",
          elementFor("#path-status").textContent.includes("口径"),
          elementFor("#path-status").textContent);
  } catch (caught) {
    error = `${scenario.name}: ${stage}: ${caught && caught.message ? caught.message : caught}`;
  }

  return { name: scenario.name, error, checks };
}

const results = scenarios.map(runScenario);
const ok = results.every(
  (item) => !item.error && item.checks.every((entry) => entry.ok)
);
process.stdout.write(JSON.stringify({
  ok,
  error: results.map((item) => item.error).filter(Boolean).join(" | ") || null,
  scenarios: results
}));
