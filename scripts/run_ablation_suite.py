"""四个结构消融实验的**一次性顺序执行器**（对应 docs/ABLATION_GUIDE.md）。

做三件事，全部可断点续跑：

    1. 依次训练 5 个 run（ab_full + 4 个消融），共享 configs/ablation_chengdu.yaml，
       差异全部通过 --set 注入，每个 run 的 run_config.json 都能自证是哪个变体；
    2. 每个 run 依次在 3 个测试集上评测（成都 normal / 成都 long / 西安），
       统一 strict multi top_k=2 beam=3 deterministic；
    3. 汇总成一张对比表写 outputs/ablation/ABLATION_REPORT.md。

顺序是**串行**的：一张卡上并行跑会互相抢显存，而且耗时不可比。

``train.py`` 的输出是**实时透传**到终端的（不是攒到结束才打），所以训练期间能直接
看到 loss / 每轮验证指标。

断点续跑的判定**不能只看 best.pt**：``best.pt`` 在第 1 个 epoch 之后就会被写出来，
只看它会把"只训了 1 轮"的 run 当成已完成而静默跳过 —— 那样最后的消融表是错的，
而且没有任何报错。所以这里按 ``history.json`` 里**已完成的 epoch 数**判断：

    已完成 >= 目标 epoch        -> 跳过
    0 < 已完成 < 目标 epoch     -> 从 last.pt 续训到目标 epoch
    其余                        -> 从头训练

用法::

    python scripts/run_ablation_suite.py                 # 全程：训练 + 评测 + 报告
    python scripts/run_ablation_suite.py --stage train   # 只训练
    python scripts/run_ablation_suite.py --stage eval    # 只评测
    python scripts/run_ablation_suite.py --stage report  # 只出报告
    python scripts/run_ablation_suite.py --only ab_full ab_direct
    python scripts/run_ablation_suite.py --force         # 已完成的也重跑
    python scripts/run_ablation_suite.py --dry-run       # 只打印将执行什么
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config  # noqa: E402

CONFIG = "configs/ablation_chengdu.yaml"
TRAIN_DATA = "data/didi/graph/chengdu/train.pkl"
VAL_DATA = "data/didi/graph/chengdu/val.pkl"
OUT_DIR = PROJECT_ROOT / "outputs" / "ablation"
RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"

#: (run 名, 该变体相对 Full 的 --set 覆盖, 中文说明)
VARIANTS: List[Tuple[str, Dict[str, str], str]] = [
    ("ab_full", {}, "完整模型（唯一基准）"),
    ("ab_reset_h", {"model.persistent_state": "false"},
     "w/o Persistent State：每个 reverse step 重新 init_nodes"),
    ("ab_no_edge_state", {"model.use_edge_state_conditioning": "false"},
     "w/o Edge-State Conditioning：切断 z_t -> edge state -> GraphFlow"),
    ("ab_first_node", {"model.branch_readout": "first_node"},
     "First-Node Readout：候选仍是整条 branch，但只用第一个节点打分"),
    ("ab_direct", {"model.generation_mode": "direct"},
     "Direct Prediction：去掉扩散链，一次前向直接给 p(z_0)（非纯单变量消融）"),
]

#: (短名, pkl, 该数据集要配的 config, 中文名, GT 是否为占位符)
EVAL_SETS: List[Tuple[str, str, str, str, bool]] = [
    ("chengdu_test1000", "data/didi/graph/chengdu/test_1000.pkl",
     "configs/didi_chengdu.yaml", "成都 normal", False),
    ("chengdu_long_test1000", "data/didi/graph/chengdu_long/test_1000.pkl",
     "configs/didi_chengdu.yaml", "成都 long", False),
    ("xian_test1000", "data/didi/graph/xian/test_1000.pkl",
     "configs/didi_xian.yaml", "西安", False),
]

METRICS = [
    ("goal_hit_rate", "GoalHit", 4, True),
    ("broken_rate", "Broken", 4, False),
    ("path_similarity_score", "PathSim", 4, True),
    ("normalized_lcs_success", "nLCS", 4, True),
    ("edge_f1", "EdgeF1", 4, True),
    ("pred_over_gt_cost_ratio", "pred/GT", 3, False),
    ("dtw_km", "DTW(km)", 3, False),
]


def preflight(need_eval: bool) -> List[str]:
    """开跑前检查数据/配置是否齐 —— 缺文件时给一条清楚的清单。

    不做这一步的后果实测过：评测阶段会为每个 run x 每个数据集各抛一次
    FileNotFoundError traceback，刷十几屏，很难一眼看出"只是少拷了数据集"。
    """
    problems: List[str] = []
    for rel in (CONFIG, TRAIN_DATA, VAL_DATA):
        if not (PROJECT_ROOT / rel).exists():
            problems.append(f"缺少训练输入: {rel}")
    if need_eval:
        for short, data, config, cn, _ph in EVAL_SETS:
            if not (PROJECT_ROOT / config).exists():
                problems.append(f"缺少配置（{cn}）: {config}")
            if not (PROJECT_ROOT / data).exists():
                problems.append(f"缺少数据集（{cn}）: {data}")
    return problems


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="run the four structural ablations")
    parser.add_argument("--stage", choices=("all", "train", "eval", "report"), default="all")
    parser.add_argument("--only", nargs="*", default=None, help="只跑指定 run")
    parser.add_argument("--force", action="store_true", help="已完成/已存在的也重跑")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--quiet", action="store_true", help="不透传子进程输出")
    return parser.parse_args()


def target_epochs() -> int:
    return int(load_config(CONFIG, []).get("training.epochs", 20))


def completed_epochs(name: str) -> int:
    """history.json 里记了多少轮 —— 这是判断"训完了没有"的唯一可靠依据。"""
    path = RUNS_DIR / name / "history.json"
    if not path.exists():
        return 0
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0
    return len(records) if isinstance(records, list) else 0


def run(cmd: List[str], dry: bool, quiet: bool = False) -> int:
    """跑一条子命令；输出**实时透传**，同时留一份给失败时打印尾部。"""
    child_env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    print("    " + " ".join(cmd), flush=True)
    if dry:
        return 0

    started = time.time()
    proc = subprocess.Popen(
        cmd, cwd=str(PROJECT_ROOT), env=child_env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    tail: List[str] = []
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip()
        if not line:
            continue
        tail.append(line)
        if len(tail) > 60:
            tail.pop(0)
        if not quiet:
            print(f"      | {line}", flush=True)
    proc.wait()
    took = time.time() - started

    if proc.returncode == 0:
        print(f"    OK  {took:.0f}s", flush=True)
    else:
        print(f"    FAILED rc={proc.returncode} ({took:.0f}s)", flush=True)
        for line in tail[-25:]:
            print(f"      ! {line}", flush=True)
    return proc.returncode


# ---------------------------------------------------------------------------
def train_variants(args, selected) -> int:
    want = target_epochs()
    print("=" * 78)
    print(f"阶段 1/3 · 训练（{len(selected)} 个 run，串行，每个 {want} epoch）")
    print("=" * 78)
    failures = 0
    for name, overrides, note in selected:
        run_dir = RUNS_DIR / name
        done = completed_epochs(name)
        has_last = (run_dir / "last.pt").exists()
        print(f"\n[{name}] {note}")
        print(f"    已完成 {done}/{want} epoch")

        cmd = [args.python, "scripts/train.py", "--config", CONFIG, "--name", name,
               "--data", TRAIN_DATA, "--val-data", VAL_DATA, "--device", args.device]
        for key, value in overrides.items():
            cmd += ["--set", f"{key}={value}"]

        if done >= want and not args.force:
            print("    SKIP（已达目标 epoch）")
            continue
        if 0 < done < want and has_last and not args.force:
            # 中断续训：从 last.pt 接着跑，不是重头来
            print(f"    续训 {want - done} 个 epoch（从 last.pt）")
            cmd += ["--resume", str(run_dir / "last.pt"),
                    "--extra-epochs", str(want - done)]
            failures += run(cmd, args.dry_run, args.quiet) != 0
        else:
            failures += run(cmd, args.dry_run, args.quiet) != 0
    return failures


def eval_variants(args, selected) -> int:
    print()
    print("=" * 78)
    print("阶段 2/3 · 评测（每个 run x 3 个测试集，strict 2/3 deterministic）")
    print("=" * 78)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0
    for name, _overrides, _note in selected:
        ckpt = RUNS_DIR / name / "best.pt"
        if not ckpt.exists():
            print(f"\n[{name}] SKIP：没有 best.pt（先跑 --stage train）")
            failures += 1
            continue
        for short, data, config, cn, _ph in EVAL_SETS:
            out = OUT_DIR / f"{name}__{short}.json"
            print(f"\n[{name}] {cn}  ({short})")
            if out.exists() and not args.force:
                print(f"    SKIP（已有 {out.name}）")
                continue
            cmd = [args.python, "scripts/evaluate.py", "--config", config,
                   "--checkpoint", str(ckpt), "--data", data, "--out", str(out),
                   "--beam-width", "3", "--device", args.device, "--no-progress"]
            failures += run(cmd, args.dry_run, args.quiet) != 0
    return failures


# ---------------------------------------------------------------------------
def fmt(value, digits: int) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if number != number:
        return "n/a"
    return f"{number:.{digits}f}"


def write_report(selected) -> None:
    print()
    print("=" * 78)
    print("阶段 3/3 · 汇总报告")
    print("=" * 78)
    baseline = "ab_full"
    lines: List[str] = []
    add = lines.append
    add("# 结构消融实验报告")
    add("")
    add("由 `scripts/run_ablation_suite.py` 自动生成，协议见 `docs/ABLATION_GUIDE.md`。")
    add("")
    add("口径：strict multi, top_k=2, beam=3, deterministic；数据：成都主训练集配方，20 epoch。")
    add("")

    for short, _data, _cfg, cn, placeholder in EVAL_SETS:
        add(f"## {cn}（{short}）")
        add("")
        add("| run | 变体 | " + " | ".join(n for _k, n, _d, _h in METRICS) + " |")
        add("|---|---|" + "---|" * len(METRICS))
        rows = {}
        for name, _ov, note in selected:
            path = OUT_DIR / f"{name}__{short}.json"
            if not path.exists():
                add(f"| {name} | {note} | " + " | ".join("—" for _ in METRICS) + " |")
                continue
            metrics = json.loads(path.read_text(encoding="utf-8"))["metrics"]
            rows[name] = metrics
            values = []
            for key, _n, digits, _h in METRICS:
                if placeholder and key in ("path_similarity_score", "normalized_lcs_success"):
                    values.append("—")
                else:
                    values.append(fmt(metrics.get(key), digits))
            add(f"| {name} | {note} | " + " | ".join(values) + " |")

        if baseline in rows and len(rows) > 1:
            add("")
            add("**相对 ab_full 的变化（Δ）**")
            add("")
            add("| run | " + " | ".join(n for _k, n, _d, _h in METRICS) + " |")
            add("|---|" + "---|" * len(METRICS))
            base = rows[baseline]
            for name, _ov, _note in selected:
                if name == baseline or name not in rows:
                    continue
                cells = []
                for key, _n, digits, higher_better in METRICS:
                    try:
                        delta = float(rows[name][key]) - float(base[key])
                    except (TypeError, ValueError):
                        cells.append("—")
                        continue
                    mark = " ▲" if (delta > 0) == higher_better and abs(delta) > 1e-9 else (
                        " ▼" if abs(delta) > 1e-9 else "")
                    cells.append(f"{delta:+.{digits}f}{mark}")
                add(f"| {name} | " + " | ".join(cells) + " |")
        add("")

    report = OUT_DIR / "ABLATION_REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"    写入 {report.relative_to(PROJECT_ROOT)}")


def main() -> int:
    args = parse_args()
    selected = [v for v in VARIANTS if args.only is None or v[0] in args.only]
    if not selected:
        raise SystemExit("no run selected; check --only")

    print(f"消融套件：{len(selected)} 个 run x {len(EVAL_SETS)} 个测试集   "
          f"目标 {target_epochs()} epoch/run")
    for name, overrides, _note in selected:
        pretty = " ".join(f"--set {k}={v}" for k, v in overrides.items()) or "(无覆盖 = Full)"
        print(f"  {name:<18} {pretty}")
    print()

    problems = preflight(need_eval=args.stage in ("all", "eval"))
    if problems:
        print("！开跑前检查未通过：")
        for line in problems:
            print(f"    - {line}")
        print()
        print("评测需要 3 个测试集（缺少的可以从训练机拷过来）：")
        print("    data/didi/graph/chengdu/{train,val,test_1000}.pkl")
        print("    data/didi/graph/chengdu_long/{test_1000}.pkl  + graph_global.pkl")
        print("    data/didi/graph/xian/{test_1000}.pkl          + graph_global.pkl")
        print()
        print("或者：只在这台机器上训练，把 outputs/runs/ab_*/ 拷到有数据的机器上评测")
        print("      （python scripts/run_ablation_suite.py --stage eval）")
        return 2

    failures = 0
    if args.stage in ("all", "train"):
        failures += train_variants(args, selected)
    if args.stage in ("all", "eval"):
        failures += eval_variants(args, selected)
    if args.stage in ("all", "report") and not args.dry_run:
        write_report(selected)

    print()
    print(f"完成。失败/缺失 {failures} 项。")
    if failures == 0 and args.stage in ("all", "report"):
        print("报告：outputs/ablation/ABLATION_REPORT.md")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
