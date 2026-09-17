"""把 ``outputs/eval/`` 里的协议结果汇总成论文用的报告。

数据来源是**评测 JSON 文件本身**（文件名即 ``<dataset>__beam<N>__<model>.json``），
不依赖 ``protocol_index.json`` —— 索引只是运行日志，文件坏了索引不一定知道。

产出::

    outputs/eval/PROTOCOL_REPORT.md      主表 / 搜索预算表 / shuffled 表 / 验收判定
    outputs/eval/protocol_index.json     从实际文件重建的索引（覆盖运行器写的）

用法::

    python scripts/report_eval_protocol.py
    python scripts/report_eval_protocol.py --out outputs/eval/PROTOCOL_REPORT.md
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EVAL_DIR = PROJECT_ROOT / "outputs" / "eval"


def _load_protocol():
    """直接读 run_eval_protocol.py 里的矩阵，避免把"期望跑哪些格"抄成第二份。"""
    spec = importlib.util.spec_from_file_location(
        "eval_protocol_matrix", PROJECT_ROOT / "scripts" / "run_eval_protocol.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PROTO = _load_protocol()
#: 协议规定的格子集合：(短名, beam, 模型 slug)
EXPECTED_CELLS = [
    (_PROTO.DATASETS[dataset_key][0], beam, slug)
    for dataset_key, beam, _baselines in _PROTO.MATRIX
    for slug in ("base", "finetuned")
]

#: 短名 -> (数据集代号, 中文名, GT 是否为 dijkstra 占位符)
DATASETS = {
    "chengdu_test1000": ("D1", "成都 normal", False),
    "chengdu_long_test1000": ("D2", "成都 long", False),
    "xian_test1000": ("D3", "西安", False),
    "chengdu_shuffled1000": ("D4", "成都 shuffled", True),
    "chengdu_long_shuffled1000": ("D5", "大图 shuffled", True),
    "xian_shuffled1000": ("D6", "西安 shuffled", True),
}
MODELS = {"base": "M0 base", "finetuned": "M1 finetuned"}

#: 论文表格里要报的指标（顺序即列顺序）
MAIN_METRICS = [
    ("goal_hit_rate", "GoalHit", 4),
    ("broken_rate", "Broken", 4),
    ("path_similarity_score", "PathSim", 4),
    # 模型的 metrics 里没有 normalized_lcs（只有 _success 版）；两者差别只是分母用
    # 全样本还是只用成功样本。这里用 _success，语义更干净："到了的里面像不像"。
    ("normalized_lcs_success", "nLCS", 4),
    ("edge_f1", "EdgeF1", 4),
    ("pred_over_gt_cost_ratio", "pred/GT", 3),
    ("dtw_km", "DTW(km)", 3),
]

#: 方案 §23 的验收阈值（M0 的基线值）
ACCEPT = [
    ("D2", 3, "goal_hit_rate", ">", 0.782, "大图 GoalHit 明显高于基线"),
    ("D2", 3, "broken_rate", "<", 0.218, "大图 Broken 明显低于基线"),
    ("D2", 3, "path_similarity_score", ">", 0.116, "大图 PathSim 明显高于基线"),
    ("D2", 3, "pred_over_gt_cost_ratio", "<", 1.350, "大图 pred/GT 明显低于基线"),
    ("D1", 3, "goal_hit_rate", ">=", 0.990, "成都 normal 无灾难性遗忘"),
    ("D3", 3, "goal_hit_rate", ">=", 0.990, "西安 无灾难性遗忘"),
]


def parse_name(path: Path) -> Optional[Tuple[str, int, str]]:
    stem = path.stem
    if stem == "protocol_index":
        return None
    parts = stem.split("__")
    if len(parts) != 3:
        return None
    dataset, beam_part, model = parts
    if dataset not in DATASETS or model not in MODELS or not beam_part.startswith("beam"):
        return None
    return dataset, int(beam_part[4:]), model


def load_cells() -> Dict[Tuple[str, int, str], Dict[str, Any]]:
    cells: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    for path in sorted(EVAL_DIR.glob("*.json")):
        key = parse_name(path)
        if key is None:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:  # noqa: BLE001
            print(f"[report] SKIP unreadable {path.name}: {error}")
            continue
        if "metrics" not in payload:
            print(f"[report] SKIP no metrics: {path.name}")
            continue
        cells[key] = {"payload": payload, "path": path}
    return cells


def fmt(value: Any, digits: int) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if number != number:  # NaN
        return "n/a"
    return f"{number:.{digits}f}"


def delta_str(a: Any, b: Any, digits: int, higher_better: bool) -> str:
    try:
        d = float(b) - float(a)
    except (TypeError, ValueError):
        return "—"
    if d != d:
        return "—"
    mark = "+" if d > 0 else ""
    arrow = ""
    if abs(d) > 1e-9:
        good = (d > 0) if higher_better else (d < 0)
        arrow = " ▲" if good else " ▼"
    return f"{mark}{d:.{digits}f}{arrow}"


def main() -> int:
    parser = argparse.ArgumentParser(description="build the paper evaluation report")
    parser.add_argument("--out", default=str(EVAL_DIR / "PROTOCOL_REPORT.md"))
    args = parser.parse_args()

    cells = load_cells()
    if not cells:
        raise SystemExit(f"no usable eval json under {EVAL_DIR}")

    datasets = sorted({k[0] for k in cells}, key=lambda d: DATASETS[d][0])
    beams = sorted({k[1] for k in cells})
    # 期望的格子来自协议矩阵（8 组 x 2 模型 = 16），**不是** 数据集 x beam 的全交叉
    expected = len(EXPECTED_CELLS)
    missing = [cell for cell in EXPECTED_CELLS if cell not in cells]

    lines: List[str] = []
    add = lines.append

    add("# 论文评测报告（Evaluation Report）")
    add("")
    add("本文件由 `scripts/report_eval_protocol.py` 从 `outputs/eval/*.json` 自动生成，**不要手改**。")
    add("协议定义见 `docs/EVAL_PROTOCOL.md`。")
    add("")
    add(f"- 评测格子：**{len(cells)}**（期望 {expected}）")
    if missing:
        add(f"- ⚠️ 缺失 {len(missing)} 格：" + ", ".join(
            f"{d}/beam{b}/{m}" for d, b, m in missing))
    else:
        add("- 矩阵完整，无缺失")
    add("")

    # ---------------- 主表 ----------------
    add("## 1. 主表（口径 R3 = strict multi, top_k=2, beam=3, deterministic）")
    add("")
    add("| 数据集 | 模型 | " + " | ".join(name for _, name, _ in MAIN_METRICS) + " |")
    add("|---|---|" + "---|" * len(MAIN_METRICS))
    for d in datasets:
        if (d, 3, "base") not in cells and (d, 3, "finetuned") not in cells:
            continue
        code, cn, placeholder = DATASETS[d]
        for m in ("base", "finetuned"):
            cell = cells.get((d, 3, m))
            if cell is None:
                continue
            metrics = cell["payload"]["metrics"]
            values = []
            for key, _name, digits in MAIN_METRICS:
                if placeholder and key in ("path_similarity_score", "normalized_lcs_success"):
                    values.append("—")
                else:
                    values.append(fmt(metrics.get(key), digits))
            add(f"| {code} {cn} | {MODELS[m]} | " + " | ".join(values) + " |")
    add("")
    add("> D4–D6 的 GT 是 `dijkstra_placeholder`（合成最短路），PathSim / nLCS 无意义，记 `—`。")
    add("")

    # ---------------- 变化表 ----------------
    add("## 2. 微调前后逐格对比")
    add("")
    add("| 数据集 | beam | 指标 | M0 base | M1 finetuned | Δ |")
    add("|---|---|---|---|---|---|")
    for d in datasets:
        code, cn, placeholder = DATASETS[d]
        for b in beams:
            a = cells.get((d, b, "base"))
            c = cells.get((d, b, "finetuned"))
            if a is None or c is None:
                continue
            ma, mc = a["payload"]["metrics"], c["payload"]["metrics"]
            for key, name, digits in MAIN_METRICS:
                if placeholder and key in ("path_similarity_score", "normalized_lcs_success"):
                    continue
                if ma.get(key) is None or mc.get(key) is None:
                    continue
                higher_better = key not in ("broken_rate", "pred_over_gt_cost_ratio", "dtw_km")
                add(f"| {code} {cn} | {b} | {name} | {fmt(ma[key], digits)} | "
                    f"{fmt(mc[key], digits)} | {delta_str(ma[key], mc[key], digits, higher_better)} |")
    add("")

    # ---------------- 搜索预算（C3）----------------
    long_cells = [(b, cells.get(("chengdu_long_test1000", b, m)))
                  for b in beams for m in MODELS]
    if any(c is not None for _b, c in long_cells):
        add("## 3. 搜索预算曲线（C3：提升来自模型还是 decoder 搜索）")
        add("")
        add("| beam | M0 base GoalHit | M1 finetuned GoalHit | Δ |")
        add("|---|---|---|---|")
        for b in beams:
            a = cells.get(("chengdu_long_test1000", b, "base"))
            c = cells.get(("chengdu_long_test1000", b, "finetuned"))
            if a is None or c is None:
                continue
            ga = a["payload"]["metrics"]["goal_hit_rate"]
            gc = c["payload"]["metrics"]["goal_hit_rate"]
            add(f"| {b} | {fmt(ga, 4)} | {fmt(gc, 4)} | {delta_str(ga, gc, 4, True)} |")
        # 差距收敛：小 beam 与最大 beam 的差
        if (("chengdu_long_test1000", min(beams), "base") in cells
                and ("chengdu_long_test1000", max(beams), "base") in cells):
            lo, hi = min(beams), max(beams)
            gaps = {}
            for m in MODELS:
                a = cells.get(("chengdu_long_test1000", lo, m))
                c = cells.get(("chengdu_long_test1000", hi, m))
                if a and c:
                    gaps[m] = (c["payload"]["metrics"]["goal_hit_rate"]
                               - a["payload"]["metrics"]["goal_hit_rate"])
            if len(gaps) == 2:
                add("")
                add(f"**beam{hi} − beam{lo} 的 GoalHit 差距**："
                    f"M0 = {gaps['base']:.4f}，M1 = {gaps['finetuned']:.4f}")
                shrink = gaps["base"] - gaps["finetuned"]
                add("")
                if shrink > 0.05:
                    add(f"→ 差距收窄 **{shrink:.4f}**。小 beam 已接近上限，"
                        "说明提升来自模型的概率场，而不是 decoder 搜得更暴力。**C3 成立。**")
                else:
                    add(f"→ 差距只收窄 {shrink:.4f}，C3 证据偏弱，需谨慎表述。")
                add("")

    # ---------------- 传统基线 ----------------
    add("## 4. 与传统基线的配对比较")
    add("")
    found_baseline = False
    for d in datasets:
        for b in beams:
            for m in MODELS:
                cell = cells.get((d, b, m))
                if cell is None:
                    continue
                payload = cell["payload"]
                paired = payload.get("baselines_real") or payload.get("baselines")
                if not paired:
                    continue
                found_baseline = True
                code, cn, _ = DATASETS[d]
                add(f"### {code} {cn} · beam{b} · {MODELS[m]}")
                add("")
                if isinstance(paired, dict) and paired:
                    model_metrics = payload.get("metrics", {})
                    add("| 方法 | GoalHit | PathSim | pred/GT | EdgeF1 |")
                    add("|---|---|---|---|---|")
                    add(f"| **模型 {MODELS[m]}** | "
                        f"{fmt(model_metrics.get('goal_hit_rate'), 4)} | "
                        f"{fmt(model_metrics.get('path_similarity_score'), 4)} | "
                        f"{fmt(model_metrics.get('pred_over_gt_cost_ratio'), 3)} | "
                        f"{fmt(model_metrics.get('edge_f1'), 4)} |")
                    for name, row in paired.items():
                        add(f"| {name} | {fmt(row.get('goal_hit_rate'), 4)} | "
                            f"{fmt(row.get('path_similarity_score'), 4)} | "
                            f"{fmt(row.get('pred_over_gt_cost_ratio'), 3)} | "
                            f"{fmt(row.get('edge_f1'), 4)} |")
                add("")
    if found_baseline:
        add("> 上表只是**同一批 query 上的均值**。配对显著性（ΔnLCS + 95% CI，McNemar / bootstrap）")
        add("> 用 `tools/paired_compare.py` 单独出，本报告不重复实现统计检验：")
        add(">")
        add("> ```bash")
        add("> python tools/paired_compare.py \\")
        add(">   --a outputs/eval/chengdu_test1000__beam3__base.json \\")
        add(">   --b outputs/eval/chengdu_test1000__beam3__finetuned.json \\")
        add(">   --data data/didi/graph/chengdu/test_1000.pkl --metric goal_hit")
        add("> ```")
        add("")
    else:
        add("（本次评测未启用 `--baselines`）")
        add("")

    # ---------------- 验收 ----------------
    add("## 5. 验收判定（方案 §23）")
    add("")
    add("| # | 判据 | 阈值 | 实测 (M1) | 结论 |")
    add("|---|---|---|---|---|")
    all_pass = True
    for i, (code, beam, key, op, threshold, text) in enumerate(ACCEPT, start=1):
        dataset = next((d for d, (c, _cn, _p) in DATASETS.items() if c == code), None)
        cell = cells.get((dataset, beam, "finetuned")) if dataset else None
        if cell is None:
            add(f"| {i} | {text} | {op} {threshold} | 缺数据 | ⏳ |")
            all_pass = False
            continue
        value = cell["payload"]["metrics"].get(key)
        ok = value is not None and (
            value > threshold if op == ">" else
            value < threshold if op == "<" else
            value >= threshold if op == ">=" else value <= threshold
        )
        all_pass = all_pass and bool(ok)
        add(f"| {i} | {text} | {op} {threshold} | {fmt(value, 4)} | "
            f"{'✅ 通过' if ok else '❌ 未过'} |")
    add("")
    add(f"**总体：{'全部通过 ✅' if all_pass else '存在未通过项 ❌'}**")
    add("")

    report = "\n".join(lines) + "\n"
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.write_text(report, encoding="utf-8")

    # 从实际文件重建索引（覆盖运行器写的局部索引）
    index = {
        "protocol": "docs/EVAL_PROTOCOL.md",
        "generated_by": "scripts/report_eval_protocol.py",
        "cells": len(cells),
        "expected": expected,
        "missing": [f"{d}/beam{b}/{m}" for d, b, m in missing],
        "runs": [
            {
                "dataset": DATASETS[d][0], "dataset_key": d, "beam": b,
                "model": MODELS[m], "model_slug": m,
                "out": str(cells[(d, b, m)]["path"].relative_to(PROJECT_ROOT)),
                "goal_hit_rate": cells[(d, b, m)]["payload"]["metrics"].get("goal_hit_rate"),
                "broken_rate": cells[(d, b, m)]["payload"]["metrics"].get("broken_rate"),
                "path_similarity_score": cells[(d, b, m)]["payload"]["metrics"].get("path_similarity_score"),
                "placeholder_gt": DATASETS[d][2],
            }
            for (d, b, m) in sorted(cells, key=lambda k: (DATASETS[k[0]][0], k[1], k[2]))
        ],
    }
    (EVAL_DIR / "protocol_index.json").write_text(
        json.dumps(index, indent=1, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[report] wrote {out_path}")
    print(f"[report] cells={len(cells)}/{expected}  missing={len(missing)}")
    print(f"[report] index rebuilt -> {EVAL_DIR / 'protocol_index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
