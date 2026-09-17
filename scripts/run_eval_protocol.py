"""按 ``docs/EVAL_PROTOCOL.md`` 跑完整套评测矩阵。

协议要点（详见文档）：

    模型   M0 base      = outputs/runs/didi_chengdu_loss_improved/best.pt
           M1 finetuned = outputs/runs/didi_chengdu_large_finetune/best.pt
    数据集 D1..D6       = 固定 test_1000 / shuffled_od_1000（各 1000 条）
    口径   R3/R8/R16    = strict multi top_k=2，beam = 3 / 8 / 16，deterministic

    8 组 × 2 模型 = 16 次评测，产物统一落 ``outputs/eval/<dataset>__beam<N>__<model>.json``。

评测结果**不写进** ``outputs/runs/<run>/``：run 目录只放训练产出（权重/曲线/配置），
评测是跨模型的横向对比，混在一起会让"同模型多口径"和"同口径多模型"分不开。

用法::

    python scripts/run_eval_protocol.py                 # 补跑缺失的
    python scripts/run_eval_protocol.py --dry-run       # 只打印计划
    python scripts/run_eval_protocol.py --force         # 全部重跑
    python scripts/run_eval_protocol.py --only M1 --beams 3 --datasets D1 D2 D3
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

#: 两个模型必须成对评测，任何一张表都不允许只出现其中一个
MODELS = {
    "M0": ("base", "outputs/runs/didi_chengdu_loss_improved/best.pt"),
    "M1": ("finetuned", "outputs/runs/didi_chengdu_large_finetune/best.pt"),
}

#: D# -> (文件名用的短名, pkl 路径, 该数据集要配的 config)
#: config 决定坐标文件（DTW）与 graph_global（paths.data_dir），成都要用成都的、西安要用西安的
DATASETS = {
    "D1": ("chengdu_test1000", "data/didi/graph/chengdu/test_1000.pkl", "configs/didi_chengdu.yaml"),
    "D2": ("chengdu_long_test1000", "data/didi/graph/chengdu_long/test_1000.pkl", "configs/didi_chengdu.yaml"),
    "D3": ("xian_test1000", "data/didi/graph/xian/test_1000.pkl", "configs/didi_xian.yaml"),
    "D4": ("chengdu_shuffled1000", "data/didi/graph/chengdu/shuffled_od_1000.pkl", "configs/didi_chengdu.yaml"),
    "D5": ("chengdu_long_shuffled1000", "data/didi/graph/chengdu_long/shuffled_od_1000.pkl", "configs/didi_chengdu.yaml"),
    "D6": ("xian_shuffled1000", "data/didi/graph/xian/shuffled_od_1000.pkl", "configs/didi_xian.yaml"),
}

#: (数据集, beam 宽度, 是否额外跑传统基线)  —— 与协议 §5 的矩阵一一对应
MATRIX = [
    ("D1", 3, True),
    ("D2", 3, True),
    ("D2", 8, False),
    ("D2", 16, False),
    ("D3", 3, True),
    ("D4", 3, False),
    ("D5", 3, False),
    ("D6", 3, False),
]

#: D4–D6 的 GT 是 dijkstra_placeholder，PathSim 无意义，报告里要标出来
PLACEHOLDER_GT = {"D4", "D5", "D6"}

OUT_DIR = PROJECT_ROOT / "outputs" / "eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="run the paper evaluation protocol")
    parser.add_argument("--force", action="store_true", help="已存在的也重跑")
    parser.add_argument("--dry-run", action="store_true", help="只打印将执行什么")
    parser.add_argument("--only", nargs="*", default=None, choices=sorted(MODELS),
                        help="只跑指定模型，如 --only M1")
    parser.add_argument("--datasets", nargs="*", default=None, choices=sorted(DATASETS))
    parser.add_argument("--beams", nargs="*", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def cell_path(dataset_key: str, beam: int, model_slug: str) -> Path:
    short = DATASETS[dataset_key][0]
    return OUT_DIR / f"{short}__beam{beam}__{model_slug}.json"


def main() -> int:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    models = [k for k in sorted(MODELS) if args.only is None or k in args.only]
    cells = [
        (d, b, bl) for d, b, bl in MATRIX
        if (args.datasets is None or d in args.datasets)
        and (args.beams is None or b in args.beams)
    ]
    if not cells:
        raise SystemExit("no cell selected; check --datasets / --beams")

    total = len(cells) * len(models)
    print(f"协议矩阵: {len(cells)} 组 x {len(models)} 模型 = {total} 次评测")
    for key in models:
        slug, ckpt = MODELS[key]
        if not (PROJECT_ROOT / ckpt).exists():
            raise SystemExit(f"checkpoint missing for {key}: {ckpt}")
    print()

    index = {
        "protocol": "docs/EVAL_PROTOCOL.md",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": {k: {"slug": MODELS[k][0], "checkpoint": MODELS[k][1]} for k in models},
        "runs": [],
    }

    done = skipped = failed = 0
    for cell_i, (dataset_key, beam, with_baselines) in enumerate(cells, start=1):
        short, data_path, config = DATASETS[dataset_key]
        for key in models:
            slug, ckpt = MODELS[key]
            out = cell_path(dataset_key, beam, slug)
            label = f"[{cell_i}/{len(cells)}] {dataset_key} {short} beam={beam} {key}({slug})"

            if out.exists() and not args.force:
                print(f"{label}\n    SKIP (exists: {out.name})", flush=True)
                skipped += 1
                index["runs"].append({
                    "dataset": dataset_key, "data_file": data_path, "beam": beam,
                    "model": key, "model_slug": slug, "baselines": with_baselines,
                    "out": str(out.relative_to(PROJECT_ROOT)), "status": "skipped",
                    "placeholder_gt": dataset_key in PLACEHOLDER_GT,
                })
                continue

            cmd = [
                args.python, "scripts/evaluate.py",
                "--config", config,
                "--checkpoint", ckpt,
                "--data", data_path,
                "--out", str(out),
                "--beam-width", str(beam),
                "--device", args.device,
                "--no-progress",
            ]
            if with_baselines:
                cmd.append("--baselines")

            print(f"{label}\n    {' '.join(cmd)}", flush=True)
            if args.dry_run:
                continue

            started = time.time()
            # 子进程按 UTF-8 输出（脚本里有中文），父进程必须显式按 UTF-8 解码；
            # 不指定的话 Windows 会拿 GBK 去解，直接 UnicodeDecodeError 把整个协议打断。
            child_env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
            proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True,
                                  text=True, encoding="utf-8", errors="replace",
                                  env=child_env)
            elapsed = time.time() - started
            status = "ok" if proc.returncode == 0 and out.exists() else "failed"
            if status == "ok":
                done += 1
                tail = [line for line in (proc.stdout or "").strip().splitlines() if line.strip()][-2:]
                print(f"    OK  {elapsed:.0f}s  {out.name}", flush=True)
                for line in tail:
                    print(f"      {line.strip()}", flush=True)
            else:
                failed += 1
                print(f"    FAILED rc={proc.returncode} ({elapsed:.0f}s)", flush=True)
                print(((proc.stderr or "") + (proc.stdout or ""))[-1500:], flush=True)

            index["runs"].append({
                "dataset": dataset_key, "data_file": data_path, "beam": beam,
                "model": key, "model_slug": slug, "baselines": with_baselines,
                "out": str(out.relative_to(PROJECT_ROOT)), "status": status,
                "seconds": round(elapsed, 1),
                "placeholder_gt": dataset_key in PLACEHOLDER_GT,
            })

    index["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    index["summary"] = {"done": done, "skipped": skipped, "failed": failed}
    if not args.dry_run:
        (OUT_DIR / "protocol_index.json").write_text(
            json.dumps(index, indent=1, ensure_ascii=False), encoding="utf-8"
        )

    print()
    print(f"完成: {done} 跑完 / {skipped} 跳过 / {failed} 失败")
    if not args.dry_run:
        print(f"索引: {OUT_DIR / 'protocol_index.json'}")
        print(f"下一步: python scripts/report_eval_protocol.py")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
