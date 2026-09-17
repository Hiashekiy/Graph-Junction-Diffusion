"""前端静态检查 + 真跑一遍（Node + DOM 桩）。

守的是**只有运行时才会暴露**的两类问题：

1. **调用了不存在的函数**。引用一个未定义的全局函数是完全合法的 JS 语法，
   `node --check` 查不出来。2026-09-16 删指标页时把 `fmt()` 一起删了，而
   `renderDiffusionFrame()` 最后一行还在用它 —— 每帧抛 ReferenceError，异常从
   `renderDiffusionGraph()` 里冒出来，把紧随其后的 `startDiffusion()` 一起吞掉，
   最终表现成"扩散过程不能自动播放、只能拖时间轴"。
2. **`$("#id")` 指向已经被删掉的 DOM 节点**。删掉一整个面板后，JS 里遗留的
   `$("#metrics-panel")` 会静默拿到 `null`，等到某次交互才炸。

第 3 个测试用 DOM 桩真的执行 `app.js`，走一遍"点扩散过程"的用户路径。
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STATIC = PROJECT_ROOT / "dashboard" / "static"


# ---------------------------------------------------------------------------
# 1) 作用域感知的未定义调用检查
# ---------------------------------------------------------------------------
LOCAL_DECL = re.compile(r"\b(?:const|let|var|function|class)\s+([A-Za-z_$][\w$]*)")
#: 文件作用域的声明必须按**行首无缩进**判断：用 ``\b`` 会把函数体里的
#: ``const fmt = ...`` 也算成全局，于是"在别的函数里调用 fmt()"正好被漏掉。
TOP_DECL = re.compile(
    r"^(?:async\s+)?(?:const|let|var|function|class)\s+([A-Za-z_$][\w$]*)", re.M
)
TOP_FUNC = re.compile(r"^(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(", re.M)
CATCH = re.compile(r"\bcatch\s*\(\s*([A-Za-z_$][\w$]*)")
CALL = re.compile(r"(?<![\w$.])([A-Za-z_$][\w$]*)\s*\(")

JS_GLOBALS = {
    "document", "window", "console", "Math", "JSON", "Object", "Array", "Number",
    "String", "Boolean", "Map", "Set", "Date", "Promise", "Error", "RegExp",
    "setTimeout", "clearTimeout", "setInterval", "clearInterval", "fetch",
    "requestAnimationFrame", "cancelAnimationFrame", "encodeURIComponent",
    "decodeURIComponent", "parseInt", "parseFloat", "isNaN", "isFinite", "Option",
    "structuredClone", "Intl", "WeakMap", "Function", "Symbol", "BigInt",
    "globalThis", "queueMicrotask", "getComputedStyle", "matchMedia",
    "if", "for", "while", "switch", "catch", "return", "typeof", "new", "function",
    "class", "else", "do", "try", "finally", "in", "of", "delete", "void", "throw",
    "case", "default", "break", "continue", "instanceof", "yield", "await", "async",
    "super", "this",
}

#: 正则字面量只可能出现在这些字符之后（或行首 / return 之类的关键字之后）。
#: 少了这一步，`escapeHtml` 里的 `/[&<>'"]/g` 会被当成字符串开头，把后面几百行
#: 全挖空 —— 上一版检查器就是这么把 bug 一起漏掉的。
REGEX_PREFIX = set("(,=:[!&|?{};+-*%~^<>")
REGEX_KEYWORDS = ("return", "typeof", "case", "in", "of", "do", "else", "void",
                  "delete", "instanceof", "new", "yield", "await")


def _regex_allowed(source: str, index: int) -> bool:
    cursor = index - 1
    while cursor >= 0 and source[cursor] in " \t\r\n":
        cursor -= 1
    if cursor < 0:
        return True
    previous = source[cursor]
    if previous in REGEX_PREFIX:
        return True
    if previous in ")]" or previous.isalnum() or previous in "_$":
        word_end = cursor + 1
        word_start = word_end
        while word_start > 0 and (
            source[word_start - 1].isalnum() or source[word_start - 1] in "_$"
        ):
            word_start -= 1
        return source[word_start:word_end] in REGEX_KEYWORDS
    return True


def code_mask(source: str) -> str:
    """等长掩码：代码原样，注释与字符串**字面量**替换成空格。

    模板字面量里的 ``${...}`` 还原成代码（只挖掉纯文本部分）—— 真实调用经常写在
    模板串里，整段挖掉就什么都查不到了。
    """
    out = list(source)
    size = len(source)
    index = 0

    def blank(start: int, end: int) -> None:
        for position in range(start, min(end, size)):
            if out[position] != "\n":
                out[position] = " "

    while index < size:
        char = source[index]
        nxt = source[index + 1] if index + 1 < size else ""
        if char == "/" and nxt == "/":
            end = source.find("\n", index)
            end = size if end < 0 else end
            blank(index, end)
            index = end
        elif char == "/" and nxt == "*":
            end = source.find("*/", index + 2)
            end = size if end < 0 else end + 2
            blank(index, end)
            index = end
        elif char == "/" and _regex_allowed(source, index):
            cursor = index + 1
            in_class = False
            while cursor < size:
                current = source[cursor]
                if current == "\\":
                    cursor += 2
                    continue
                if current == "[":
                    in_class = True
                elif current == "]":
                    in_class = False
                elif current == "/" and not in_class:
                    break
                elif current == "\n":
                    break
                cursor += 1
            blank(index + 1, cursor)
            index = cursor + 1
        elif char in "'\"":
            cursor = index + 1
            while cursor < size:
                if source[cursor] == "\\":
                    cursor += 2
                    continue
                if source[cursor] == char:
                    break
                cursor += 1
            blank(index, cursor + 1)
            index = cursor + 1
        elif char == "`":
            cursor = index + 1
            depth = 0
            while cursor < size:
                if source[cursor] == "\\":
                    cursor += 2
                    continue
                if depth == 0 and source[cursor] == "`":
                    break
                if source[cursor] == "$" and cursor + 1 < size and source[cursor + 1] == "{":
                    depth += 1
                    cursor += 2
                    continue
                if depth and source[cursor] == "}":
                    depth -= 1
                    cursor += 1
                    continue
                if depth == 0:
                    out[cursor] = "\n" if out[cursor] == "\n" else " "
                cursor += 1
            index = cursor + 1
        else:
            index += 1
    return "".join(out)


def undefined_calls(path: Path):
    source = path.read_text(encoding="utf-8")
    code = code_mask(source)
    top_level = set(TOP_DECL.findall(code))
    problems = {}
    for match in TOP_FUNC.finditer(code):
        name = match.group(1)
        brace = code.find("{", match.end())
        depth, cursor = 0, brace
        while cursor < len(code):
            if code[cursor] == "{":
                depth += 1
            elif code[cursor] == "}":
                depth -= 1
                if depth == 0:
                    break
            cursor += 1
        body = code[brace:cursor]
        locals_ = set(LOCAL_DECL.findall(body)) | set(CATCH.findall(body))
        header = code[match.start():brace]
        locals_ |= set(re.findall(r"([A-Za-z_$][\w$]*)", header[header.find("(") + 1:]))
        for call in CALL.finditer(body):
            called = call.group(1)
            if called in locals_ or called in top_level or called in JS_GLOBALS:
                continue
            line = source.count("\n", 0, brace + call.start()) + 1
            problems.setdefault(called, []).append((line, name))
    return problems


def test_app_js_has_no_undefined_calls() -> None:
    problems = undefined_calls(STATIC / "app.js")
    assert not problems, "\n".join(
        f"{called}() 在 {name} 内被调用但本作用域与文件顶层都没有定义（第 {line} 行）"
        for called, hits in sorted(problems.items())
        for line, name in hits
    )


def test_video_export_controls_and_capture_pipeline_exist() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    script = (STATIC / "app.js").read_text(encoding="utf-8")

    assert 'id="export-video"' in html
    for token in ("exportCurrentVideo", "captureStream", "MediaRecorder", "paintVideoFrame"):
        assert token in script


# ---------------------------------------------------------------------------
# 2) DOM id 交叉检查
# ---------------------------------------------------------------------------
def test_every_dom_selector_used_by_app_js_exists_in_index_html() -> None:
    """`$("#id")` / `$$("#id …")` 指向的节点必须还在 index.html 里。

    删掉一整个面板后，JS 里遗留的 `$("#metrics-panel")` 会静默拿到 null，
    等到下一次交互才炸 —— 这类问题必须在测试里就挡住。
    """
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    known = set(re.findall(r'id="([^"]+)"', html))
    # app.js 自己用 svgEl(..., {id: "..."}) 建出来的节点（例如扩散状态层）
    known |= set(re.findall(r"""id:\s*["']([A-Za-z][\w-]*)["']""", script))

    # 模板拼出来的选择器只有一处：["path","diffusion"].forEach(name => $(`#view-${name}`))
    used = {"view-path", "view-diffusion"}
    for pattern in (r"""["']#([A-Za-z][\w-]*)["']""", r"""["']#([A-Za-z][\w-]*)[\s.#\[]"""):
        for match in re.finditer(pattern, script):
            name = match.group(1)
            # "#b7ff4a" 这类是配色字面量，不是选择器
            if re.fullmatch(r"[0-9a-fA-F]{3}|[0-9a-fA-F]{6}", name):
                continue
            used.add(name)

    missing = sorted(used - known)
    assert not missing, f"app.js 引用了 index.html 与自身都没创建的 id：{missing}"


def test_index_html_does_not_reference_removed_panels() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    for banned in ("metrics-panel", "metrics-tab", "model-filters", "metric-chart"):
        assert banned not in html, banned


# ---------------------------------------------------------------------------
# 3) 真的执行一遍："点扩散过程"必须能自动播放
# ---------------------------------------------------------------------------
def frontend_payload(sample):
    """用 server.py 自己的序列化函数造一份响应（不需要加载任何模型）。"""
    from dashboard.server import _diffusion_payload, _graph_payload, _route_payload
    from src.data.collate import collate_samples
    from src.evaluation.path_decoder import decode_flat

    batch = collate_samples([sample], device="cpu")
    steps = 4
    log_prob = [torch.full((sample.num_candidates,), -1.0) for _ in range(steps)]
    z_path = [batch.target_candidate.clone() for _ in range(steps + 1)]
    trace = SimpleNamespace(log_prob=log_prob, z_path=z_path)
    decoded = decode_flat(
        sample, batch.target_candidate, decision_offset=0, candidate_offset=0
    )
    return {
        "model": "unweighted_synthetic",
        "dataset": "manual.pkl",
        "index": 0,
        "num_queries": 1,
        "checkpoint_epoch": 1,
        "sample": {
            "start": int(sample.start),
            "goal": int(sample.goal),
            "num_nodes": int(sample.num_nodes),
            "num_decisions": int(sample.num_decisions),
            "gt_length": int(sample.gt_length),
            "difficulty": "n/a",
            "mode": "n/a",
            "gt_path": [int(node) for node in sample.gt_path],
            "source": "",
        },
        "graph": _graph_payload(sample),
        "routes": [_route_payload(decoded, 1, sample.graph)],
        "summary": {
            "coverage": decoded.status == "goal",
            "num_finished": 1,
            "num_goal_paths": int(decoded.status == "goal"),
            "pruned": 0,
            "num_filtered_dead_branches": 0,
        },
        "diffusion": _diffusion_payload(sample, trace, batch.candidate_owner),
        "contrast": None,
        "ruler": {"mode": "ruler", "strict": True, "top_k": 2, "beam_width": 3,
                  "note": "评测标尺 strict 2/3（configs/didi_chengdu.yaml）"},
    }


#: 两种模型口径各一份：有 strict 多分支标尺的滴滴，和只有 single 兜底的合成模型
MULTI_RULER_CATALOG = {
    "models": [{
        "id": "didi_chengdu", "label": "didi_chengdu", "kind": "didi",
        "ruler": {"decode": "multi", "strict": True, "top_k": 2, "beam_width": 3,
                  "declared": True, "multi": True, "source": "configs/didi_chengdu.yaml",
                  "label": "strict 2/3"},
    }],
    "datasets": [], "reports": [],
}

SINGLE_RULER_CATALOG = {
    "models": [{
        "id": "controlled_unweighted", "label": "controlled_unweighted",
        "kind": "unweighted",
        "ruler": {"decode": "single", "strict": False, "top_k": 2, "beam_width": 64,
                  "declared": False, "multi": False, "source": "configs/controlled_unweighted.yaml",
                  "label": "single（该 run 没有多分支标尺）"},
    }],
    "datasets": [], "reports": [],
}


def geo_variant(payload: dict) -> dict:
    """把合成 payload 改造成"地理模式"：街道层 / 比例尺 / 说明行的代码路径都要跑到。

    坐标直接复用弹簧布局的归一化值 —— 前端只是把它们线性映射到画布，数值从哪来无所谓；
    这里要验的是 geo 分支不抛异常、自动播放照样启动。
    """
    import copy

    payload = copy.deepcopy(payload)
    graph = payload["graph"]
    nodes = graph["nodes"]
    graph["geo"] = True
    graph["source"] = "data/didi/raw/chengdu/ChengDu.pkl"
    graph["lat0"] = 30.69
    graph["coverage"] = 0.962
    graph["crop_km"] = [6.2, 3.56]
    graph["km_per_x_unit"] = 6.204
    graph["street_truncated"] = False
    # 用真实节点坐标造几条街道，避免越界坐标掩盖问题
    graph["street_edges"] = [
        [nodes[0]["x"], nodes[0]["y"], nodes[1]["x"], nodes[1]["y"]],
        [nodes[1]["x"], nodes[1]["y"], nodes[2]["x"], nodes[2]["y"]],
    ]
    return payload


@pytest.mark.skipif(shutil.which("node") is None, reason="need node to execute app.js")
def test_diffusion_view_autoplays_in_a_real_js_runtime(manual_sample, tmp_path: Path) -> None:
    """回归：用户报的「扩散过程不能自动播放、只能拖时间轴」。

    根因是 `renderDiffusionFrame()` 调了一个已经被删掉的 `fmt()`，每帧抛
    ReferenceError；而 `setGraphView('diffusion')` 的顺序是
    `renderDiffusionGraph()` -> `startDiffusion(true)`，异常从前者冒出来就把自动播放
    整个吞掉；点「播放」也一样（它先 render 再置 playing 标志），所以按钮完全没反应。

    合成图与地理图各跑一遍。
    """
    base = frontend_payload(manual_sample)
    assert len(base["diffusion"]["frames"]) > 2, "帧数太少，测不出前进"
    scenarios = [
        {"name": "合成图", "payload": base, "catalog": MULTI_RULER_CATALOG},
        {"name": "地理模式", "payload": geo_variant(base), "catalog": MULTI_RULER_CATALOG},
        {"name": "单路径标尺的 run", "payload": base, "catalog": SINGLE_RULER_CATALOG},
    ]

    payload_path = tmp_path / "payloads.json"
    payload_path.write_text(json.dumps({"payloads": scenarios}), encoding="utf-8")
    harness = Path(__file__).resolve().parent / "frontend" / "dashboard_smoke.js"

    completed = subprocess.run(
        ["node", str(harness), str(payload_path)],
        capture_output=True, text=True, timeout=120,
        # node 按 UTF-8 写 stdout；Windows 上 text=True 默认跟本地代码页（GBK），
        # 中文检查名会直接把读取线程搞崩
        encoding="utf-8", errors="replace",
    )
    assert completed.returncode == 0, (completed.stderr or completed.stdout)[-2000:]
    lines = [line for line in (completed.stdout or "").strip().splitlines() if line.strip()]
    assert lines, "harness 没有输出"
    result = json.loads(lines[-1])

    failed = [
        item
        for scenario in result["scenarios"]
        for item in scenario["checks"]
        if not item["ok"]
    ]
    assert result["ok"], "\n".join(
        [f"error={result['error']}"]
        + [f"  x {item['name']} {item['detail']}" for item in failed]
    )
    # 三个场景都要真的跑过（防止 harness 悄悄少跑一个）
    assert {item["name"] for item in result["scenarios"]} == {
        "合成图", "地理模式", "单路径标尺的 run",
    }

