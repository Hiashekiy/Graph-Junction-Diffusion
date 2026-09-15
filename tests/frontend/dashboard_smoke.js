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
    vm.runInContext("state.pathData = __payload;", context);

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

    // ---- 5) 切回路径视图 -------------------------------------------------
    stage = "backToPath";
    vm.runInContext("setGraphView('path');", context);
    check("setGraphView('path') 不抛异常", true);
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
