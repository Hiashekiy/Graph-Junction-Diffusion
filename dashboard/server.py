"""Zero-dependency web server for the interactive experiment dashboard.

The browser UI is static HTML/CSS/JavaScript.  This module exposes a small JSON
API that discovers local runs/datasets, reads existing evaluation artifacts and
runs one-query inference through the project's real decoding pipeline.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import mimetypes
import sys
import threading
import traceback
import webbrowser
from dataclasses import asdict, dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
from urllib.parse import unquote, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = Path(__file__).resolve().parent / "static"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


#: 面板"评测标尺"模式的默认值。**必须与 ``scripts/evaluate.py`` 的
#: ``_DECODE_RULER_DEFAULTS`` 逐键一致** —— 面板存在的意义就是把评测口径复现出来，
#: 两份常量一旦漂移，面板就变成第四把尺子。
#: ``tests/test_dashboard_frontend.py::test_panel_ruler_defaults_match_evaluate_script``
#: 会静态比对这两份，改了一边而忘了另一边会直接测试失败。
RULER_DEFAULTS: Dict[str, Any] = {
    "decode": "single",
    "top_k": 2,
    "beam_width": 64,
    "null_policy": "stop",
    "filter_dead_branches": False,
    "strict_decode": False,
}

#: 面板手动控件在服务端的夹取范围（别让请求里的离谱数字把 beam search 拖死）。
MULTI_LIMITS: Dict[str, Tuple[int, int]] = {
    "top_k": (1, 6),
    "beam_width": (1, 256),
    "display_paths": (1, 24),
}

#: ``POST /api/path`` 里 ``ruler`` 的合法取值。
#:
#: 2026-09-16：**只剩 ruler**。"历史（存活路径表）"口径已从面板移除 —— 它把
#: goal / NULL / loop / dead-end 放进同一个池子按累计 log 概率排序，于是"最早被打断的
#: 残骸"因为负数加得少而当选（DiDi test_1000 实测 `multi.best` 平均 6.89 跳，真正走到
#: 终点的那条平均 18.5 跳）。同一份 best.pt：历史 2/3 是 GoalHit 0.336 / DTW 1.15 km，
#: 标尺 strict 2/3 是 GoalHit 0.996 / DTW 0.27 km。两把尺子同时摆在面板里，
#: 只会让人对着图猜"到底哪个数才算数"——所以直接删掉一把。
RULER_MODES = ("ruler",)


@dataclass(frozen=True)
class DecodeRuler:
    """一个 run 的**评测标尺**：验证 / 选 best.pt / 最终测试三处共用的那套解码口径。

    来源是**当前** ``configs/<run>.yaml`` 的 ``evaluation.*``（不是训练快照，理由同
    :func:`_live_run_settings`）。没写这些键的旧 config 落到 :data:`RULER_DEFAULTS`，
    与 ``scripts/evaluate.py`` 的 CLI 默认逐位一致。
    """

    decode: str = "single"
    strict: bool = False
    top_k: int = 2
    beam_width: int = 64
    null_policy: str = "stop"
    filter_dead_branches: bool = False
    #: config 里是否**真的写了** ``evaluation.decode``。没写说明这个 run 的标尺是
    #: 兜底默认值（``single``），面板不该声称"对齐了评测"。
    declared: bool = False
    source: str = ""

    @property
    def is_multi(self) -> bool:
        return self.decode == "multi"

    @property
    def label(self) -> str:
        if not self.is_multi:
            return "single（该 run 没有多分支标尺）"
        head = "strict" if self.strict else "历史"
        return f"{head} {self.top_k}/{self.beam_width}"


@dataclass(frozen=True)
class RunInfo:
    id: str
    label: str
    path: Path
    checkpoint: Path
    # Weighted 扩展之后，模型分三类，面板需要把这件事显式标出来：
    #   weighted=True  + use_edge_cost=True   -> 带权模型（看得到 edge cost）
    #   weighted=True  + use_edge_cost=False  -> cost 消融对照
    #   weighted=False                        -> 无权模型
    weighted: bool = False
    use_edge_cost: bool = False
    flow_steps: int = 0
    data_dir: str = ""
    train_dataset: str = ""
    #: ``data.source``：``"didi_chengdu"`` 表示真实成都路网 + 真实车辆历史路径当 GT。
    #: 它决定了面板能不能画街道底图，也决定模型徽章上显示"滴滴"。
    source: str = ""
    #: **当前** ``configs/<run>.yaml`` 里记录的坐标文件（OSMnx 图，节点带经纬度）。
    #: 空串 = 该 run 没有地理信息（合成图），面板退回弹簧布局。
    coords_file: str = ""
    #: 解析到的 live config（相对仓库根的路径），给面板显示来源用。
    live_config: str = ""
    #: 该 run 的评测标尺（验证 / 选 best / 最终测试共用），面板默认按它解码
    ruler: DecodeRuler = field(default_factory=DecodeRuler)

    @property
    def is_geo(self) -> bool:
        return bool(self.coords_file)

    @property
    def kind(self) -> str:
        if self.source == "didi_chengdu":
            return "didi"
        if not self.weighted:
            return "unweighted"
        return "weighted" if self.use_edge_cost else "ablated"

    @property
    def kind_label(self) -> str:
        return {
            "didi": "滴滴·带权",
            "weighted": "带权",
            "ablated": "带权·无cost",
            "unweighted": "无权",
        }[self.kind]


@dataclass(frozen=True)
class DatasetInfo:
    id: str
    label: str
    path: Path
    size_mb: float
    group: str = ""          # 相对 data/ 的所属子目录（unweighted / didi/graph/chengdu …）
    relative: str = ""       # 相对仓库根的可读路径，给面板当提示用


def _relative_id(path: Path, base: Path) -> str:
    return path.resolve().relative_to(base.resolve()).as_posix()


#: ``data/`` 下**不是**数据集的 pkl。数据集必须是某个 split 的落盘文件；
#: ``graph_global.pkl`` 是整城路网、``_*.pkl`` 是 prepare 阶段的中间缓存，
#: 两者用 ``GraphQueryDataset.load`` 打开都会直接抛异常。
_NON_DATASET_NAMES = frozenset({"graph_global.pkl"})

#: 原始数据目录名：``data/didi/raw/chengdu`` 里是 dicts.pkl / ChengDu.pkl 之类的
#: **输入**而不是数据集，整个子树都要跳过。
_RAW_DIR_NAMES = frozenset({"raw"})


def _read_live_config(root: Path, name: str) -> Dict[str, Any]:
    """读 ``configs/<name>.yaml`` 并抽出面板关心的字段；读不到返回空 dict。"""
    path = root / "configs" / f"{name}.yaml"
    if not path.is_file():
        return {}
    try:
        import yaml
    except ImportError:  # pragma: no cover - 仓库必装 PyYAML
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    if not isinstance(payload, Mapping):
        return {}
    data_cfg = payload.get("data") if isinstance(payload.get("data"), Mapping) else {}
    paths_cfg = payload.get("paths") if isinstance(payload.get("paths"), Mapping) else {}
    eval_cfg = (
        payload.get("evaluation") if isinstance(payload.get("evaluation"), Mapping) else {}
    )
    source = data_cfg.get("source")
    coords = data_cfg.get("coords_file")
    ruler = {}
    for key, fallback in RULER_DEFAULTS.items():
        value = eval_cfg.get(key, None)
        ruler[key] = fallback if value is None else value
    ruler["declared"] = "decode" in eval_cfg
    return {
        "live_config": f"configs/{name}.yaml",
        "run_name": str(paths_cfg.get("run_name", "") or ""),
        "data_dir": str(paths_cfg.get("data_dir", "") or ""),
        # source 在合成数据上是个 dict（generator 的 source 子配置），只有字符串才是
        # 真正的"数据来源"标记
        "source": source if isinstance(source, str) else "",
        "coords_file": str(coords) if coords else "",
        "ruler": ruler,
    }


def _ruler_from_settings(live: Mapping[str, Any], run_name: str) -> DecodeRuler:
    """把 live config 的 ``evaluation.*`` 变成一个 :class:`DecodeRuler`。"""
    raw = live.get("ruler")
    raw = raw if isinstance(raw, Mapping) else {}

    def pick(key: str) -> Any:
        value = raw.get(key, None)
        return RULER_DEFAULTS[key] if value is None else value

    return DecodeRuler(
        decode=str(pick("decode")).lower(),
        strict=bool(pick("strict_decode")),
        top_k=int(pick("top_k")),
        beam_width=int(pick("beam_width")),
        null_policy=str(pick("null_policy")),
        filter_dead_branches=bool(pick("filter_dead_branches")),
        declared=bool(raw.get("declared", False)),
        source=str(live.get("live_config", "") or ""),
    )


def _live_run_settings(root: Path, names: Iterable[str]) -> Dict[str, Any]:
    """按候选名字找 live config；都不中就在 ``configs/`` 里按 ``paths.run_name`` 反查。

    为什么不能只用 run 目录里的 ``run_config.json``：那是**训练当时**的快照，仓库
    重构后可能指向早就删掉的路径。实测 ``outputs/runs/didi_chengdu/run_config.json``
    里还写着

        data.root        = data/DiDiChengduXian/didi_datasets/datasets/didi_chengdu
        data.coords_file = data/DiDiChengduXian/data/data/cd/ChengDu.pkl
        paths.data_dir   = data/didi_chengdu_gjd
        paths.run_name   = didi_chengdu_flow1_weighted_new

    这四个名字/路径在 2026-09-16 的清理里全失效了（真实位置是 ``data/didi/raw/
    chengdu/ChengDu.pkl`` 与 ``data/didi/graph/chengdu``）。所以职责拆开：

    * **模型结构 / 扩散参数** → 必须用 ``run_config.json``（权重兼容，见 ``_bundle``）
    * **数据在哪、坐标在哪** → 一律用 live config

    名字候选顺序是"**目录名优先**，再退到快照里的 run_name" —— 合并/改名过的 run
    目录（``didi_chengdu``）留着的是旧快照，只有目录名才是当前配置名。最后再扫一遍
    ``configs/*.yaml`` 里 ``paths.run_name`` 的声明，覆盖"目录被改名"的反向情况。
    """
    tried: List[str] = []
    for name in names:
        name = str(name or "").strip()
        if not name or name in tried:
            continue
        tried.append(name)
        found = _read_live_config(root, name)
        if found:
            return found
    config_dir = root / "configs"
    if config_dir.is_dir():
        for path in sorted(config_dir.glob("*.yaml")):
            found = _read_live_config(root, path.stem)
            if found and found.get("run_name") in tried:
                return found
    return {}


def _existing_relative(root: Path, value: str) -> str:
    """把配置里的相对路径规范化；文件不存在就返回空串（面板据此关闭地理模式）。"""
    value = str(value or "").strip()
    if not value:
        return ""
    path = Path(value.replace("\\", "/"))
    if not path.is_absolute():
        path = root / path
    return _relative_id(path, root) if path.is_file() else ""


def discover_runs(root: Path = PROJECT_ROOT) -> Dict[str, RunInfo]:
    runs_root = root / "outputs" / "runs"
    found: Dict[str, RunInfo] = {}
    if not runs_root.exists():
        return found
    for config_path in sorted(runs_root.rglob("run_config.json")):
        run_dir = config_path.parent
        checkpoint = run_dir / "best.pt"
        if not checkpoint.exists():
            continue
        run_id = _relative_id(run_dir, runs_root)
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            config = {}
        if not isinstance(config, Mapping):
            config = {}
        data_cfg = config.get("data") if isinstance(config.get("data"), Mapping) else {}
        model_cfg = config.get("model") if isinstance(config.get("model"), Mapping) else {}
        paths_cfg = config.get("paths") if isinstance(config.get("paths"), Mapping) else {}
        snapshot_data_dir = str(paths_cfg.get("data_dir", "") or "")
        snapshot_source = data_cfg.get("source")
        snapshot_source = snapshot_source if isinstance(snapshot_source, str) else ""

        # 目录名优先：合并/改名过的 run 目录里那份快照可能还写着旧 run_name
        # （didi_chengdu 的快照写的是 didi_chengdu_flow1_weighted_new），照着它去找
        # configs/*.yaml 会一个都找不到，然后静默退回一份过期快照。
        live = _live_run_settings(
            root, (run_dir.name, str(paths_cfg.get("run_name") or ""))
        )
        found[run_id] = RunInfo(
            run_id,
            run_id.replace("/", " / "),
            run_dir,
            checkpoint,
            weighted=bool(data_cfg.get("weighted", False)),
            use_edge_cost=bool(model_cfg.get("use_edge_cost", False)),
            flow_steps=int(model_cfg.get("flow_steps", 0) or 0),
            data_dir=str(live.get("data_dir") or snapshot_data_dir),
            train_dataset=str(data_cfg.get("train_dataset", "")),
            source=str(live.get("source") or snapshot_source),
            coords_file=_existing_relative(root, str(live.get("coords_file", ""))),
            live_config=str(live.get("live_config", "")),
            ruler=_ruler_from_settings(live, run_id),
        )
    return found


def discover_datasets(root: Path = PROJECT_ROOT) -> Dict[str, DatasetInfo]:
    """Discover every **dataset** below ``data/``.

    Datasets live in per-family sub-directories, so the scan is recursive and the
    listing comes back grouped.  ``group`` is the path **relative to ``data/``**
    (``unweighted`` / ``weighted`` / ``didi/graph/chengdu``), which is what the
    picker shows as an ``optgroup`` label -- it has to be the relative path and
    not the immediate parent, otherwise the DiDi splits show up under a bare
    ``chengdu`` with no hint that they are the real-road ones.

    The id stays the bare file name: historical ``eval*.json`` / ``mp_*.json``
    artifacts record their dataset by file name, and the id is what the browser
    sends back on every request.  Two same-named files in different groups fall
    back to the relative id so one can never shadow the other.

    Excluded on purpose:

    * anything under a ``raw/`` directory -- ``data/didi/raw/chengdu`` holds
      ``dicts.pkl`` / ``ChengDu.pkl``, which are model **inputs**, not datasets;
    * ``_``-prefixed files (``_didi_candidates.pkl`` is the prepare-stage cache);
    * ``graph_global.pkl`` (the whole-city junction graph behind every corridor).

    All three would blow up in ``GraphQueryDataset.load`` if the picker offered
    them, and the failure only shows up as "推理未完成" after the click.
    """
    data_root = root / "data"
    found: Dict[str, DatasetInfo] = {}
    if not data_root.exists():
        return found
    for path in sorted(data_root.rglob("*.pkl")):
        parts = path.relative_to(data_root).parts
        if any(part in _RAW_DIR_NAMES for part in parts[:-1]):
            continue
        if path.name.startswith("_") or path.name in _NON_DATASET_NAMES:
            continue
        dataset_id = path.name
        if dataset_id in found:  # never let one group shadow another
            dataset_id = _relative_id(path, data_root)
        label = path.stem.replace("_", " ")
        group = Path(*parts[:-1]).as_posix() if len(parts) > 1 else ""
        try:
            relative = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:  # 数据集在仓库外（测试用的 tmp_path）
            relative = path.as_posix()
        found[dataset_id] = DatasetInfo(
            dataset_id,
            label,
            path,
            round(path.stat().st_size / (1024 * 1024), 2),
            group,
            relative,
        )
    return found


def _finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


def resolve_multi_decode(
    request: Mapping[str, Any], run: Optional[RunInfo]
) -> Dict[str, Any]:
    """把「面板请求 + run 的评测标尺」解析成 ``decode_multi_path`` 的实参。

    面板现在**只有一套解码语义**：strict 三池
    （``src.evaluation.strict_beam_decoder``）—— NULL / loop / dead-end 在 top-k
    之前就被 mask、失败路径直接淘汰，最终候选集只有完整走到 Goal 的路径。
    这一点**不可配置**（历史口径已移除，见 :data:`RULER_MODES`）。

    可配置的只有**搜索预算**：

    * ``top_k``（每次分叉）与 ``beam_width``（路径表上限）：默认取该 run 的评测标尺，
      请求里给了就用请求里的（仍按 :data:`MULTI_LIMITS` 夹取）。一旦改了，
      面板就**不再等于**该 run 的评测标尺 —— ``note`` 会写明"手动覆盖"和标尺原值，
      避免对着一个不是评测口径的数字下结论。
    * ``null_policy`` / ``filter_dead_branches``：strict 下由解码器接管，恒取标尺值。

    run 没有多分支标尺时（``controlled_unweighted`` / ``controlled_weighted`` 的
    config 没写 ``evaluation.decode``）用 :data:`RULER_DEFAULTS` 的 top_k/beam_width
    兜底，并在 ``note`` 里说明 —— 而不是悄悄按另一套参数跑完还声称对齐了评测。

    这是纯函数（不碰 dataset / model），所以可以直接单测。
    """
    mode = str(request.get("ruler", "ruler")).lower()
    if mode not in RULER_MODES:
        raise ValueError(
            f"ruler 必须是 {' | '.join(RULER_MODES)}，收到 {mode!r}"
            "（历史口径已移除：它与评测标尺在 DiDi test_1000 上 GoalHit 差 66 个百分点）"
        )

    def clamp(key: str, value: Any) -> int:
        low, high = MULTI_LIMITS[key]
        return max(low, min(int(value), high))

    display_paths = clamp("display_paths", request.get("display_paths", 8))

    ruler = run.ruler if run is not None else None
    if ruler is not None and ruler.is_multi:
        base_top_k = clamp("top_k", ruler.top_k)
        base_beam = clamp("beam_width", ruler.beam_width)
        null_policy = ruler.null_policy
        filter_dead_branches = bool(ruler.filter_dead_branches)
        declared = bool(ruler.declared)
        source = ruler.source
    else:
        # 该 run 没有多分支标尺：用 evaluate.py 的兜底默认，但**仍然走 strict**
        # —— 面板只有一套解码语义，不因为 config 没写就偷偷换一把尺子。
        base_top_k = clamp("top_k", RULER_DEFAULTS["top_k"])
        base_beam = clamp("beam_width", RULER_DEFAULTS["beam_width"])
        null_policy = str(RULER_DEFAULTS["null_policy"])
        filter_dead_branches = bool(RULER_DEFAULTS["filter_dead_branches"])
        declared = False
        source = ""

    top_k = clamp("top_k", request.get("top_k", base_top_k))
    beam_width = clamp("beam_width", request.get("beam_width", base_beam))

    note = f"strict {top_k}/{beam_width}"
    if (top_k, beam_width) != (base_top_k, base_beam):
        note += f"（手动覆盖；该 run 标尺 {base_top_k}/{base_beam}）"
    elif ruler is not None and ruler.is_multi:
        note += f"（{'config' if declared else '默认'}{' · ' + source if source else ''}）"
    else:
        note += "（该 run 没有多分支标尺，用默认值）"

    return {
        "mode": "ruler",
        "strict": True,
        "top_k": top_k,
        "beam_width": beam_width,
        # strict 下 decoder 会接管这两个：NULL 永远不合法、必死 branch 恒被剔除
        "null_policy": null_policy,
        "filter_dead_branches": filter_dead_branches,
        "display_paths": display_paths,
        "note": note,
    }


#: 面板要展示的报告 / 汇总产物。(id, 标题, 仓库内相对路径)
#:
#: 只列**当前仓库里真的存在**的产物。``docs/REPORT_multipath_and_weighted.md``
#: 在 2026-09-16 的清理里被删掉了（方法上已被 README §24 的真实数据章节取代），
#: 这里同步移除 —— 留着一个恒 ``exists: false`` 的条目只会让报告页多一个死链接。
REPORT_ARTIFACTS: Tuple[Tuple[str, str, str], ...] = (
    ("architecture", "V2 网络结构设计报告", "docs/ARCHITECTURE_V2.md"),
    ("guide", "V2 代码实施指南", "docs/Graph-Junction-Diffusion_V2_代码实施指南.md"),
    ("all_models", "全模型统一评测汇总（8 配置 × 2 测试集）", "outputs/reports/all_models_multipath_summary.json"),
    ("weighted_experiment", "加权模型 vs cost 消融（单路径口径）", "outputs/reports/weighted_experiment.json"),
    ("multipath_weighted", "加权模型 beam=64：过滤 on/off 对照", "outputs/reports/multipath_weighted_summary.json"),
    ("multipath_filteron", "只开过滤的 beam=64 结果", "outputs/reports/multipath_weighted_filteron_summary.json"),
    ("beam_compare", "beam=64 vs beam=3 对照（CPU）", "outputs/reports/multipath_weighted_beam_compare.json"),
    ("benchmark", "推理耗时基准", "outputs/reports/benchmark_inference.json"),
    ("paired_goal_hit", "配对显著性：Goal hit", "outputs/reports/weighted_paired_goal_hit.json"),
    ("paired_optimal", "配对显著性：Optimal path", "outputs/reports/weighted_paired_optimal.json"),
    ("regression", "零破坏回归：旧 checkpoint 逐位复现", "outputs/runs/controlled_unweighted/regression_after_weighted_extension.json"),
)


def discover_reports(root: Path = PROJECT_ROOT) -> List[Dict[str, Any]]:
    """报告 / 汇总产物的清单（只读，路径写死，不做目录遍历）。"""
    reports: List[Dict[str, Any]] = []
    for report_id, label, relative in REPORT_ARTIFACTS:
        path = root / relative
        exists = path.is_file()
        reports.append(
            {
                "id": report_id,
                "label": label,
                "path": relative,
                "kind": "markdown" if path.suffix.lower() == ".md" else "json",
                "exists": exists,
                "size_kb": round(path.stat().st_size / 1024, 1) if exists else None,
            }
        )
    return reports


#: SVG 画布尺寸与留白，必须和前端 ``graphCoordinates()`` 里的常量一致：
#:
#:     x = margin + x_norm * (PANEL_WIDTH  - 2 * margin)
#:     y = margin + (1 - y_norm) * (PANEL_HEIGHT - 2 * margin)
#:
#: 地理模式下后端要按**同一个长宽比**做 letterbox，否则真实路网会被拉伸：经度方向
#: 1 度只有纬度方向 cos(lat0) 倍长（成都约 0.81），如果 x/y 各自归一化到 [0,1]，
#: 城市会被横向压扁约 19%，看上去就不像地图了。
PANEL_WIDTH = 1000.0
PANEL_HEIGHT = 620.0
PANEL_MARGIN = 55.0
PANEL_ASPECT = (PANEL_WIDTH - 2 * PANEL_MARGIN) / (PANEL_HEIGHT - 2 * PANEL_MARGIN)

#: 等距圆柱投影下 1 度 ≈ 110.57 km（纬度方向；经度方向已乘 cos(lat0) 校正过）。
KM_PER_DEGREE = 110.57

#: 裁剪框在"刚好装下 corridor"之后再整体放大多少（四周各留这么多比例的街道背景）。
#: 顺序很关键：**先补面板长宽比、再整体缩放**，不能先加 margin 再补比例 —— 后者在
#: 宽屏（面板 1.75:1）上会把近乎方形的 corridor 压成中间一小块（实测只剩 40% 宽）；
#: 整体缩放不改变形状，corridor 能占到约 47% × 77%。
CROP_MARGIN = 0.15

#: 一次最多回传多少条街道边。整城 4403 条全发也就 ~110KB，但 25% 外扩的裁剪框在
#: 长 OD 上可能覆盖大半座城市，所以留一个上限兜底。
MAX_STREET_EDGES = 6000


@dataclass(frozen=True)
class GeoLayout:
    """一张真实城市图的地理布局（等距圆柱投影，单位 = "投影度"）。

    ``positions`` 按**全局** OSM node id 索引；样本内部的编号是 corridor 独立
    relabel 过的（0..N-1），必须经 ``sample.meta['local_to_global']`` 换回来才能查表。
    """

    positions: Dict[Any, Tuple[float, float]]
    street_edges: List[Tuple[Any, Any]]
    lat0: float
    coverage: float
    source: str


def _load_pickled_graph(path: Path) -> Any:
    import pickle

    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, Mapping) and "graph" in payload:
        return payload["graph"]
    return payload


def load_geo_layout(
    root: Path, run: RunInfo, cache: Dict[str, Optional[GeoLayout]]
) -> Optional[GeoLayout]:
    """按 run 拉一次真实经纬度底图；失败/不适用返回 ``None``（并缓存这个结论）。

    只有滴滴（``data.source: didi_chengdu``）这类真实城市图才有坐标：合成图是随机
    生成的，没有任何地理位置可言，面板退回弹簧布局。
    """
    if not (run.coords_file and run.data_dir):
        return None
    if run.id in cache:
        return cache[run.id]
    # 先写 None：坐标文件缺失/损坏时不要每次请求都重试一遍几百 MB 的反序列化
    cache[run.id] = None
    coords_path = root / run.coords_file
    graph_path = root / run.data_dir / "graph_global.pkl"
    if not coords_path.is_file() or not graph_path.is_file():
        return None
    try:
        from src.data import didi_dataset as didi

        # load_node_coordinates 内部会处理 OSMnx 存档里 shapely 1.x/2.x 的兼容问题，
        # 之后再去反序列化 graph_global.pkl 才是安全的
        coordinates = didi.load_node_coordinates(coords_path)
        graph = _load_pickled_graph(graph_path)
        graph, stats = didi.attach_coordinates(graph, coordinates, fill_missing=True)
        latitudes = [float(graph.nodes[node]["y"]) for node in graph.nodes()]
        if not latitudes:
            return None
        lat0 = sum(latitudes) / len(latitudes)
        scale = math.cos(math.radians(lat0))
        positions = {
            node: (
                float(graph.nodes[node]["x"]) * scale,
                float(graph.nodes[node]["y"]),
            )
            for node in graph.nodes()
        }
        street_edges = [(int(u), int(v)) for u, v in graph.edges()]
    except Exception:  # 真实数据缺失只该让面板退回弹簧布局，不该 500
        traceback.print_exc()
        return None
    layout = GeoLayout(
        positions=positions,
        street_edges=street_edges,
        lat0=lat0,
        coverage=float(stats.get("coverage", 0.0)),
        source=run.coords_file,
    )
    cache[run.id] = layout
    return layout


def _local_to_global_positions(
    sample: Any, layout: GeoLayout
) -> Optional[Dict[int, Tuple[float, float]]]:
    """corridor 的**样本内编号** -> 投影坐标；有一个节点查不到就返回 ``None``。"""
    graph = sample.graph
    local_ids = [int(node) for node in graph.nodes()]
    mapping = sample.meta.get("local_to_global")
    if not mapping or len(mapping) < len(local_ids):
        return None
    positions: Dict[int, Tuple[float, float]] = {}
    for local in local_ids:
        global_id = mapping[local]
        point = layout.positions.get(global_id)
        if point is None:
            try:
                point = layout.positions.get(int(global_id))
            except (TypeError, ValueError):
                point = None
        if point is None:
            return None
        positions[local] = (float(point[0]), float(point[1]))
    return positions


def _letterbox(
    box: Tuple[float, float, float, float], aspect: float
) -> Tuple[float, float, float, float]:
    """把裁剪框按 ``aspect`` 补成同比例，保证归一化后形状不被拉伸。"""
    x0, x1, y0, y1 = box
    width = max(x1 - x0, 1e-9)
    height = max(y1 - y0, 1e-9)
    if width / height > aspect:
        needed = width / aspect
        center = (y0 + y1) / 2.0
        y0, y1 = center - needed / 2.0, center + needed / 2.0
    else:
        needed = height * aspect
        center = (x0 + x1) / 2.0
        x0, x1 = center - needed / 2.0, center + needed / 2.0
    return x0, x1, y0, y1


def geo_payload(
    sample: Any,
    layout: GeoLayout,
    max_street_edges: int = MAX_STREET_EDGES,
) -> Optional[Dict[str, Any]]:
    """真实经纬度底图 + corridor 的归一化坐标。

    返回 ``None`` 表示这个样本用不了地理模式（坐标查不到），调用方退回弹簧布局。

    做法和 ``tools/visualize_didi_paths.py`` 一致：整城路网当浅灰底图，corridor 叠在
    上面；裁剪到 corridor 外扩 ``CROP_MARGIN``，再 letterbox 到面板长宽比。街道只回传
    **已经算好的两端坐标**（不传 id）—— 底图是纯装饰层，混进两套编号空间只会出错。
    """
    positions = _local_to_global_positions(sample, layout)
    if not positions:
        return None
    corridor_points = list(positions.values())
    x0 = min(p[0] for p in corridor_points)
    x1 = max(p[0] for p in corridor_points)
    y0 = min(p[1] for p in corridor_points)
    y1 = max(p[1] for p in corridor_points)
    # 1) 先按面板长宽比补成同比例（形状不变，只是把短边撑开）
    bx0, bx1, by0, by1 = _letterbox((x0, x1, y0, y1), PANEL_ASPECT)
    # 2) 再绕中心整体放大，四周留出街道背景。缩放不改形状，所以第 1 步的结论仍然成立。
    grow = 1.0 + 2.0 * CROP_MARGIN
    cx, cy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
    bx0, bx1 = cx - (cx - bx0) * grow, cx + (bx1 - cx) * grow
    by0, by1 = cy - (cy - by0) * grow, cy + (by1 - cy) * grow
    span_x = bx1 - bx0
    span_y = by1 - by0

    def normalize(point: Tuple[float, float]) -> Tuple[float, float]:
        return ((point[0] - bx0) / span_x, (point[1] - by0) / span_y)

    # 街道：只保留两端都落在裁剪框里的边，并给一点容差，免得边界上出现断头路
    tol_x = span_x * 0.05
    tol_y = span_y * 0.05
    street: List[List[float]] = []
    truncated = False
    for source, target in layout.street_edges:
        a = layout.positions.get(source)
        b = layout.positions.get(target)
        if a is None or b is None:
            continue
        if not (
            bx0 - tol_x <= a[0] <= bx1 + tol_x
            and by0 - tol_y <= a[1] <= by1 + tol_y
            and bx0 - tol_x <= b[0] <= bx1 + tol_x
            and by0 - tol_y <= b[1] <= by1 + tol_y
        ):
            continue
        if len(street) >= max_street_edges:
            truncated = True
            break
        na, nb = normalize(a), normalize(b)
        street.append([na[0], na[1], nb[0], nb[1]])

    normalized = {int(node): normalize(point) for node, point in positions.items()}
    return {
        "geo": True,
        "node_positions": normalized,
        "street_edges": street,
        "street_truncated": truncated,
        "source": layout.source,
        "lat0": layout.lat0,
        "coverage": layout.coverage,
        # 前端据此画比例尺：归一化 x 方向的整幅宽度相当于多少 km
        "km_per_x_unit": span_x * KM_PER_DEGREE,
        "crop_km": [span_x * KM_PER_DEGREE, span_y * KM_PER_DEGREE],
    }


def _graph_payload(
    sample: Any, geo: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    import networkx as nx

    graph = sample.graph
    geo_meta: Dict[str, Any] = {"geo": False}
    if geo is not None:
        positions = geo["node_positions"]
        geo_meta = {key: value for key, value in geo.items() if key != "node_positions"}
    else:
        if graph.number_of_nodes() == 1:
            positions = {next(iter(graph.nodes)): (0.5, 0.5)}
        else:
            raw = nx.spring_layout(graph, seed=17, iterations=120)
            xs = [float(point[0]) for point in raw.values()]
            ys = [float(point[1]) for point in raw.values()]
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            dx, dy = max(x1 - x0, 1e-9), max(y1 - y0, 1e-9)
            positions = {
                node: ((float(point[0]) - x0) / dx, (float(point[1]) - y0) / dy)
                for node, point in raw.items()
            }
    decisions = {int(node) for node in sample.segments.decision_nodes}
    nodes = []
    for node in sorted(graph.nodes):
        kind = "ordinary"
        if int(node) == int(sample.start):
            kind = "start"
        elif int(node) == int(sample.goal):
            kind = "goal"
        elif int(node) in decisions:
            kind = "decision"
        x, y = positions[node]
        nodes.append({"id": int(node), "x": x, "y": y, "kind": kind})
    edges = [{"source": int(u), "target": int(v)} for u, v in graph.edges]
    return {"nodes": nodes, "edges": edges, **geo_meta}


def _has_edge_weights(graph: Any) -> bool:
    """图上的边是否带 weight（无权图的边没有这个属性）。"""
    try:
        return any("weight" in data for _, _, data in graph.edges(data=True))
    except (AttributeError, TypeError, ValueError):
        return False


def _weights_cost(graph: Any, edges: Iterable[Any], fallback: float) -> float:
    """按边权求和；图对象不支持按边取属性时退回 fallback（= 跳数）。

    测试与工具里会用轻量 stub 当图（只有 ``has_edge``），所以这里必须容错，
    不能让面板因为一个 stub 就 500。
    """
    try:
        return float(
            sum(float(graph.edges[u, v].get("weight", 1.0)) for u, v in edges)
        )
    except (AttributeError, TypeError, KeyError, ValueError):
        return float(fallback)


def annotate_found_order(routes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """给路线补上`found_index`（第几个完成），并返回完成步数的范围。

    为什么需要：strict 束搜索是**按深度逐层展开**的，所以 `depth` 就等于
    "搜索第几步完成"，而同一个 depth 的多条路线是**同一轮里同时毕业**的 ——
    "发现顺序"= depth 升序（同 depth 内保持概率序，`sorted` 本来就稳定）。
    面板图例里的 `rank` 是 success 池最后按 log_prob 重排出来的**概率序**，
    两者必须分开，否则 `#1` 会被读成"最先到达"。

    `depth` 全为 None（单路径解码）时不做任何标注，返回空范围。
    """
    depths = [route["depth"] for route in routes if route["depth"] is not None]
    if not depths:
        # 一个 depth 都没有（单路径解码）-> 不编造顺序，前端退回"按长度"播放
        return {"first_found_depth": None, "last_found_depth": None}
    order = sorted(
        range(len(routes)),
        key=lambda index: (routes[index]["depth"], index),
    )
    for found, position in enumerate(order, start=1):
        routes[position]["found_index"] = found
    ordered = [routes[index]["depth"] for index in order]
    return {
        "first_found_depth": int(min(ordered)),
        "last_found_depth": int(max(ordered)),
    }


def _route_payload(path: Any, rank: int, graph: Any) -> Dict[str, Any]:
    if hasattr(path, "nodes"):
        nodes = path.nodes
        status = path.status
        reason = path.reason
        log_prob = path.log_prob
        cost = path.cost
    else:
        nodes = path.path
        status = path.status
        reason = path.reason
        log_prob = None
        cost = max(len(nodes) - 1, 0)
    node_ids = [int(node) for node in nodes]
    edges = [[source, target] for source, target in zip(node_ids[:-1], node_ids[1:])]
    invalid_edges = [edge for edge in edges if not graph.has_edge(*edge)]
    if invalid_edges:
        raise ValueError(f"解码路径包含不属于原图的边：{invalid_edges[:3]}")
    # 真实 cost = sum of edge weights；无权图（没有 weight 属性）自然退化成跳数。
    # 多分支解码器已经在 PathCandidate.path_cost 里算好了，单路径在这里补算。
    path_cost = getattr(path, "path_cost", None)
    if path_cost is None:
        path_cost = _weights_cost(graph, edges, fallback=float(cost))
    # 这条路线**在第几轮扩展里被找到**。strict 束搜索是**按深度逐层展开**的
    # （第 k 轮把深度 k 的 alive 全部展开成 k+1），所以 PathCandidate.num_branches
    # 就是"搜索第几步完成" —— 面板要回答"谁先到达"靠的就是它。
    # rank 是**概率序**（success 池最后按 log_prob 重排过），两者不是一回事。
    # decode_flat 的单路径没有这个量 -> None（前端就退回按长度播放）。
    depth = getattr(path, "num_branches", None)
    return _finite(
        {
            "rank": rank,
            "nodes": node_ids,
            "edges": edges,
            "status": status,
            "reason": reason,
            "cost": int(cost),
            "path_cost": float(path_cost),
            "weighted": _has_edge_weights(graph),
            "log_prob": log_prob,
            "depth": None if depth is None else int(depth),
        }
    )


def _decode_nodes(sample: Any, state: Any) -> Dict[str, Any]:
    """把「链最终携带的状态」解出来，作为诊断口径（不是 single 的答案）。"""
    from src.evaluation.path_decoder import decode_flat

    decoded = decode_flat(sample, state, decision_offset=0, candidate_offset=0)
    return {
        "status": decoded.status,
        "reason": decoded.reason,
        **_path_stats(sample, decoded.path),
    }


def _path_stats(sample: Any, nodes: Iterable[Any]) -> Dict[str, Any]:
    """路径的跳数与真实 cost（加权图按边权）。"""
    node_ids = [int(node) for node in nodes]
    edges = [[u, v] for u, v in zip(node_ids[:-1], node_ids[1:])]
    hops = max(len(node_ids) - 1, 0)
    return {"hops": hops, "path_cost": _weights_cost(sample.graph, edges, fallback=float(hops))}


def _single_decode_contrast(
    sample: Any,
    frames: List[Dict[str, Any]],
    decoded: Any,
    chain_state: Any,
    readout: str,
    stochastic: bool,
) -> Optional[Dict[str, Any]]:
    """把「single 解码实际用的状态」与「扩散链自己携带的状态」摆在一起。

    现在两者**默认就不是同一个东西**（方案要求）：

    * single 解码用 readout：最终 candidate_prob 的**组内 argmax**（``final_prob_argmax`），
      或者勾选确定性开关后的全程 argmax rollout（``deterministic_rollout`）；
    * 扩散链自己携带的状态（``chain_state`）默认是**采样**出来的，只用于可视化/诊断。

    实测（加权测试集 #92，seed 0）：readout 18 跳到达（node 8 -> 9），而采样链在最后一个
    路口抽到 NULL → 16 跳 broken。面板把两者都摊开，避免再被误读成解码 bug。
    """
    if not frames:
        return None
    last = frames[-1]
    clean_path = [int(node) for node in last.get("clean_path", [])]
    chain_decoded = _decode_nodes(sample, chain_state)
    return {
        "readout_mode": readout,
        "chain_stochastic": bool(stochastic),
        "chain_state": chain_decoded,
        "readout": {
            "status": decoded.status,
            "reason": decoded.reason,
            **_path_stats(sample, decoded.path),
        },
        "clean_prediction": {
            "status": last.get("clean_status"),
            "reason": last.get("clean_reason"),
            **_path_stats(sample, clean_path),
        },
        "clean_path": clean_path,
    }


def _diffusion_payload(sample: Any, trace: Any, candidate_owner: Any) -> Dict[str, Any]:
    """Serialize the real reverse-chain trace in a compact browser-friendly form.

    Each categorical state contains one choice for every decision variable,
    including decisions that are unreachable from the sample start.  Decode
    both states here so the UI can distinguish the actual reachable path from
    that full decision field.
    """
    import torch

    from src.evaluation.path_decoder import decode_flat
    from src.training.losses import grouped_argmax

    candidates = sample.field.candidates
    candidate_visuals: List[Dict[str, Any]] = []
    for index, owner in enumerate(candidates.candidate_owner):
        branch = candidates.candidate_branch[index]
        edges: List[List[int]] = []
        end: Optional[int] = None
        if branch is not None:
            edges = [
                [int(source), int(target)]
                for source, target in zip(branch.nodes[:-1], branch.nodes[1:])
            ]
            end = int(branch.end)
        candidate_visuals.append(
            {
                "index": index,
                "owner": int(sample.segments.decision_nodes[int(owner)]),
                "end": end,
                "is_null": bool(candidates.candidate_is_null[index]),
                "edges": edges,
            }
        )

    frames: List[Dict[str, Any]] = []
    previous_clean = None
    expected_owners = [int(node) for node in sample.segments.decision_nodes]

    def selected_indices(state: Any) -> List[int]:
        indices = [int(value) for value in state.detach().cpu().tolist()]
        owners = [candidate_visuals[index]["owner"] for index in indices]
        if owners != expected_owners:
            raise RuntimeError(
                "扩散状态必须为每个 decision node 且仅选择一个 candidate；"
                f"期望 owners={expected_owners}，实际 owners={owners}"
            )
        return indices

    for offset, log_prob in enumerate(trace.log_prob):
        clean = grouped_argmax(log_prob, candidate_owner, sample.num_decisions)
        noisy = trace.z_path[offset + 1]
        input_noisy = trace.z_path[offset]
        clean_decoded = decode_flat(sample, clean)
        noisy_decoded = decode_flat(sample, noisy)
        confidence = torch.exp(log_prob[clean]).mean().item() if clean.numel() else 0.0
        frames.append(
            {
                "t": len(trace.log_prob) - offset,
                "clean": selected_indices(clean),
                "noisy": selected_indices(noisy),
                "clean_path": [int(node) for node in clean_decoded.path],
                "clean_status": clean_decoded.status,
                "clean_reason": clean_decoded.reason,
                "noisy_path": [int(node) for node in noisy_decoded.path],
                "noisy_status": noisy_decoded.status,
                "noisy_reason": noisy_decoded.reason,
                "mean_confidence": float(confidence),
                "noisy_changed": int((noisy != input_noisy).sum().item()),
                "clean_changed": (
                    None
                    if previous_clean is None
                    else int((clean != previous_clean).sum().item())
                ),
            }
        )
        previous_clean = clean

    forced_nodes = [int(node) for node in sample.segments.source_forced_nodes]
    return {
        "T": len(frames),
        "candidates": candidate_visuals,
        "forced_edges": [
            [source, target]
            for source, target in zip(forced_nodes[:-1], forced_nodes[1:])
        ],
        "frames": frames,
    }


class DashboardService:
    def __init__(self, root: Path = PROJECT_ROOT, device: str = "auto") -> None:
        self.root = root
        self.device_request = device
        self.runs = discover_runs(root)
        self.datasets = discover_datasets(root)
        self.reports = discover_reports(root)
        self._dataset_cache: Dict[str, Any] = {}
        self._model_key: Optional[str] = None
        self._model_bundle: Optional[Tuple[Any, Any, Any, Any]] = None
        #: run -> GeoLayout（None 表示这个 run 用不了地理底图，结论也缓存）
        self._geo_cache: Dict[str, Optional[GeoLayout]] = {}
        self._lock = threading.RLock()

    def catalog(self) -> Dict[str, Any]:
        return {
            "models": [
                {
                    "id": item.id,
                    "label": item.label,
                    "kind": item.kind,
                    "kind_label": item.kind_label,
                    "weighted": item.weighted,
                    "use_edge_cost": item.use_edge_cost,
                    "flow_steps": item.flow_steps,
                    "data_dir": item.data_dir,
                    # source / 地理底图：面板据此显示"滴滴"徽章、开街道图、并把数据集
                    # 下拉自动切到这个模型真正训过的那一份
                    "source": item.source,
                    "geo": item.is_geo,
                    "coords_file": item.coords_file,
                    "live_config": item.live_config,
                    # 该 run 的评测标尺：面板默认按它解码，并在标尺不是多分支时
                    # 禁用"评测标尺"选项（而不是悄悄退回另一套口径）
                    "ruler": {
                        "decode": item.ruler.decode,
                        "strict": item.ruler.strict,
                        "top_k": item.ruler.top_k,
                        "beam_width": item.ruler.beam_width,
                        "null_policy": item.ruler.null_policy,
                        "filter_dead_branches": item.ruler.filter_dead_branches,
                        "declared": item.ruler.declared,
                        "multi": item.ruler.is_multi,
                        "label": item.ruler.label,
                    },
                }
                for item in self.runs.values()
            ],
            "datasets": [
                {
                    "id": item.id,
                    "label": item.label,
                    "size_mb": item.size_mb,
                    "group": item.group,
                    "relative": item.relative,
                }
                for item in self.datasets.values()
            ],
            "reports": self.reports,
        }

    def report(self, report_id: str) -> Dict[str, Any]:
        """取一份报告内容：Markdown 返回原文，JSON 返回解析后的对象。"""
        entry = next((item for item in self.reports if item["id"] == report_id), None)
        if entry is None:
            raise ValueError(f"未知报告：{report_id}")
        path = self.root / entry["path"]
        if not path.is_file():
            raise ValueError(f"报告文件不存在：{entry['path']}")
        text = path.read_text(encoding="utf-8")
        payload: Dict[str, Any] = dict(entry)
        if entry["kind"] == "json":
            payload["json"] = json.loads(text)
        else:
            payload["markdown"] = text
        return payload

    def geo_layout(self, run_id: str) -> Optional[GeoLayout]:
        """该 run 的真实地理底图；合成模型返回 ``None``。"""
        run = self.runs.get(run_id)
        if run is None:
            return None
        with self._lock:
            return load_geo_layout(self.root, run, self._geo_cache)

    def _dataset(self, dataset_id: str) -> Any:
        info = self.datasets.get(dataset_id)
        if info is None:
            raise ValueError(f"未知数据集：{dataset_id}")
        if dataset_id not in self._dataset_cache:
            from src.data.dataset import GraphQueryDataset

            # Retain at most two datasets to avoid keeping every large pkl in RAM.
            if len(self._dataset_cache) >= 2:
                self._dataset_cache.pop(next(iter(self._dataset_cache)))
            self._dataset_cache[dataset_id] = GraphQueryDataset.load(info.path)
        return self._dataset_cache[dataset_id]

    def dataset_info(self, dataset_id: str) -> Dict[str, Any]:
        with self._lock:
            dataset = self._dataset(dataset_id)
            return {"id": dataset_id, "num_queries": len(dataset), "name": dataset.name}

    def _bundle(self, run_id: str) -> Tuple[Any, Any, Any, Any]:
        if run_id not in self.runs:
            raise ValueError(f"未知模型：{run_id}")
        if self._model_key == run_id and self._model_bundle is not None:
            return self._model_bundle

        import torch
        from src.training.checkpoint import load_checkpoint
        from src.training.setup import build_diffusion, build_model, get_device
        from src.utils.config import load_config

        if self._model_bundle is not None:
            self._model_bundle = None
            self._model_key = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        info = self.runs[run_id]
        config = load_config(info.path / "run_config.json")
        requested = self.device_request
        if requested == "auto":
            requested = str(config.get("training.device", "auto"))
        device = get_device(requested)
        model = build_model(config, device)
        checkpoint = load_checkpoint(info.checkpoint, model=model, map_location=device)
        model = model.to(device)
        model.eval()
        diffusion = build_diffusion(config)
        self._model_key = run_id
        self._model_bundle = (model, diffusion, device, checkpoint)
        return self._model_bundle

    def infer(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        import torch
        from src.data.collate import collate_samples
        from src.diffusion.sampler import sample_reverse_chain
        from src.evaluation.multi_path_decoder import decode_multi_path
        from src.evaluation.path_decoder import decode_flat
        from src.utils.seed import make_generator, set_seed

        run_id = str(request.get("model", ""))
        dataset_id = str(request.get("dataset", ""))
        index = int(request.get("index", 0))
        seed = int(request.get("seed", 0))
        decode = str(request.get("decode", "single"))
        if decode not in {"single", "multi"}:
            raise ValueError("decode 必须是 single 或 multi")

        with self._lock, torch.no_grad():
            dataset = self._dataset(dataset_id)
            if index < 0 or index >= len(dataset):
                raise ValueError(f"样本下标越界：{index}（数据集共 {len(dataset)} 条）")
            sample = dataset[index]
            model, diffusion, device, checkpoint = self._bundle(run_id)
            set_seed(seed)
            batch = collate_samples([sample], device=device)
            include_diffusion = bool(request.get("include_diffusion", False))
            chain = sample_reverse_chain(
                diffusion,
                model,
                batch,
                generator=make_generator(seed, device="cpu"),
                stochastic=not bool(request.get("deterministic", False)),
                record=include_diffusion,
            )
            candidate_prob = chain["candidate_prob"][: sample.num_candidates]
            deterministic = bool(request.get("deterministic", False))
            if decode == "single":
                # 新默认 single readout：最终 candidate_prob 的组内 argmax（链本身仍然采样）。
                # 勾了确定性开关时，链本身就是"每步 posterior argmax"的 rollout，解码它自己的
                # 最终状态即可 —— 那是另一条路径（deterministic_rollout），不要和默认混为一谈。
                if deterministic:
                    state = chain["z0"]
                    readout = "deterministic_rollout"
                else:
                    from src.evaluation.readout import single_path_state

                    state = single_path_state(chain, batch, "single")
                    readout = "final_prob_argmax"
                decoded = decode_flat(sample, state, decision_offset=0, candidate_offset=0)
                routes = [_route_payload(decoded, 1, sample.graph)]
                summary = {
                    "coverage": decoded.status == "goal",
                    "num_finished": 1,
                    "num_goal_paths": int(decoded.status == "goal"),
                    "pruned": 0,
                    "num_filtered_dead_branches": 0,
                }
            else:
                options = resolve_multi_decode(request, self.runs.get(run_id))
                decoded = decode_multi_path(
                    sample,
                    candidate_prob,
                    top_k=options["top_k"],
                    beam_width=options["beam_width"],
                    null_policy=options["null_policy"],
                    filter_dead_branches=options["filter_dead_branches"],
                    strict=options["strict"],
                )
                routes = [
                    _route_payload(path, rank + 1, sample.graph)
                    for rank, path in enumerate(
                        decoded.finished[: options["display_paths"]]
                    )
                ]
                summary = decoded.summary()
                # 搜索时序：面板要回答"谁先到达"，光有概率序的 rank 不够。
                summary.update(annotate_found_order(routes))

            response = {
                    "model": run_id,
                    "dataset": dataset_id,
                    "index": index,
                    "num_queries": len(dataset),
                    "checkpoint_epoch": checkpoint.get("epoch"),
                    "sample": {
                        "start": int(sample.start),
                        "goal": int(sample.goal),
                        "num_nodes": int(sample.num_nodes),
                        "num_decisions": int(sample.num_decisions),
                        "gt_length": int(sample.gt_length),
                        "difficulty": sample.meta.get("difficulty", "n/a"),
                        "mode": sample.meta.get("mode", "n/a"),
                        "gt_path": [int(node) for node in sample.gt_path],
                        # 真实数据（滴滴）没有 difficulty / mode —— 那是合成图生成器
                        # 的标签。带上真实语义的字段，否则面板左侧只能显示
                        # "n/a / n/a"，看不出这条样本到底是什么。
                        "source": str(sample.meta.get("source", "") or ""),
                        "date": sample.meta.get("date"),
                        "order_id": sample.meta.get("order_id"),
                        "gt_cost": sample.meta.get("gt_cost"),
                        "dijkstra_cost": sample.meta.get("dijkstra_cost"),
                        # GT 不是最短路：这个比值就是"司机绕了多远"（实测中位 1.12）
                        "gt_cost_ratio": sample.meta.get("gt_cost_ratio"),
                        "rho": sample.meta.get("rho"),
                        "u_turns": sample.meta.get("u_turns"),
                        "split": sample.meta.get("split"),
                    },
                    "graph": _graph_payload(
                        sample,
                        geo_payload(sample, layout)
                        if (layout := self.geo_layout(run_id)) is not None
                        else None,
                    ),
                    "routes": routes,
                    "summary": summary,
                    # 这张图到底是按哪把尺子解出来的。面板直接把它印在状态行上 ——
                    # 面板/报告/评测三处口径不一致是踩过的坑，这里不留给用户猜。
                    "ruler": (
                        options
                        if decode == "multi"
                        else {
                            "mode": "single",
                            "note": "单路径 readout（最终 candidate_prob 组内 argmax）",
                        }
                    ),
                }
            if include_diffusion:
                response["diffusion"] = _diffusion_payload(
                    sample, chain["trace"], batch.candidate_owner
                )
                if decode == "single":
                    # single readout vs 链实际携带的（采样）状态：面板直接对照
                    response["contrast"] = _single_decode_contrast(
                        sample,
                        response["diffusion"]["frames"],
                        decoded,
                        chain["z0"],
                        readout,
                        stochastic=not deterministic,
                    )
            return _finite(response)


class DashboardHandler(BaseHTTPRequestHandler):
    service: DashboardService

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[dashboard] " + fmt % args + "\n")

    def _json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(_finite(payload), ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, message: str, status: int = HTTPStatus.BAD_REQUEST) -> None:
        self._json({"error": message}, status)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/catalog":
            self._json(self.service.catalog())
            return
        if parsed.path.startswith("/api/reports/"):
            try:
                report_id = unquote(parsed.path.removeprefix("/api/reports/"))
                self._json(self.service.report(report_id))
            except Exception as exc:  # user-facing validation boundary
                self._error(str(exc))
            return
        if parsed.path.startswith("/api/datasets/"):
            try:
                dataset_id = unquote(parsed.path.removeprefix("/api/datasets/"))
                self._json(self.service.dataset_info(dataset_id))
            except Exception as exc:  # user-facing validation boundary
                self._error(str(exc))
            return
        self._static(parsed.path)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/path":
            self._error("Not found", HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 64 * 1024:
                raise ValueError("请求体为空或过大")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError("请求必须是 JSON object")
            self._json(self.service.infer(payload))
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self._error(str(exc))
        except Exception as exc:  # keep the server alive and return a useful message
            traceback.print_exc()
            self._error(f"推理失败：{exc}", HTTPStatus.INTERNAL_SERVER_ERROR)

    def _static(self, request_path: str) -> None:
        relative = "index.html" if request_path in {"", "/"} else unquote(request_path.lstrip("/"))
        candidate = (STATIC_ROOT / relative).resolve()
        try:
            candidate.relative_to(STATIC_ROOT.resolve())
        except ValueError:
            self._error("Not found", HTTPStatus.NOT_FOUND)
            return
        if not candidate.is_file():
            self._error("Not found", HTTPStatus.NOT_FOUND)
            return
        body = candidate.read_bytes()
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type + ("; charset=utf-8" if content_type.startswith("text/") else ""))
        self.send_header("Content-Length", str(len(body)))
        # Always re-read the assets: this is a local workbench and a browser
        # that keeps serving yesterday's app.js makes the panel look broken.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Graph-Junction-Diffusion interactive dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda")
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    service = DashboardService(PROJECT_ROOT, device=args.device)
    handler = type("BoundDashboardHandler", (DashboardHandler,), {"service": service})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Graph-Junction-Diffusion dashboard: {url}")
    print(f"发现 {len(service.runs)} 个模型、{len(service.datasets)} 个数据集")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
