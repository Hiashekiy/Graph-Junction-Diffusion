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
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
from urllib.parse import unquote, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = Path(__file__).resolve().parent / "static"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class RunInfo:
    id: str
    label: str
    path: Path
    checkpoint: Path


@dataclass(frozen=True)
class DatasetInfo:
    id: str
    label: str
    path: Path
    size_mb: float
    group: str = ""          # 所属子目录（controlled / long / oldv1 / mixed / smoke …）
    relative: str = ""       # 相对仓库根的可读路径，给面板当提示用


def _relative_id(path: Path, base: Path) -> str:
    return path.resolve().relative_to(base.resolve()).as_posix()


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
        found[run_id] = RunInfo(run_id, run_id.replace("/", " / "), run_dir, checkpoint)
    return found


def discover_datasets(root: Path = PROJECT_ROOT) -> Dict[str, DatasetInfo]:
    """Discover every dataset below ``data/``.

    Datasets live in per-family sub-directories (``controlled`` / ``long`` /
    ``oldv1`` / ``mixed`` / ``smoke``), so the scan is recursive and the listing
    comes back grouped.  The id stays the bare file name: existing
    ``eval*.json`` / ``mp_*.json`` artefacts record their dataset by file name,
    and the id is what the browser sends back on every request.
    """
    data_root = root / "data"
    found: Dict[str, DatasetInfo] = {}
    if not data_root.exists():
        return found
    for path in sorted(data_root.rglob("*.pkl")):
        dataset_id = path.name
        if dataset_id in found:  # never let one group shadow another
            dataset_id = _relative_id(path, data_root)
        label = path.stem.replace("_", " ")
        group = path.parent.name if path.parent != data_root else ""
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


def _dataset_id_from_path(value: str) -> str:
    return Path(str(value).replace("\\", "/")).name


def _dataset_from_eval_name(name: str) -> Optional[str]:
    stem = Path(name).stem.lower()
    if "oldv1" in stem:
        return "oldv1_test.pkl"
    if "long" in stem:
        return "controlled_long.pkl"
    if "test" in stem:
        return "controlled_test.pkl"
    return None


def _finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


def _metric_row(
    run_id: str,
    dataset_id: str,
    decoding: str,
    source: Path,
    metrics: Mapping[str, Any],
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    keys = (
        "num_queries",
        "goal_hit_rate",
        "optimal_path_rate",
        "success_cost_ratio",
        "loop_rate",
        "broken_rate",
        "mean_elapsed",
        "soft_goal_reachability",
        "coverage_rate",
        "optimal_coverage_rate",
    )
    row: Dict[str, Any] = {
        "model": run_id,
        "dataset": dataset_id,
        "decoding": decoding,
        "source": source.name,
    }
    row.update({key: metrics.get(key) for key in keys})
    if extra:
        row.update(extra)
    return _finite(row)


def discover_metrics(
    runs: Mapping[str, RunInfo], root: Path = PROJECT_ROOT
) -> List[Dict[str, Any]]:
    """Read compact metric summaries without returning per-query records."""
    rows: List[Dict[str, Any]] = []
    for run_id, run in runs.items():
        for path in sorted(run.path.glob("eval*.json")):
            # Flow-step files are ablations, not distinct trained models.  Keeping
            # them out avoids silently mixing inference configurations.
            if "_flow" in path.stem:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            metrics = payload.get("metrics")
            dataset_id = _dataset_from_eval_name(path.name)
            if not isinstance(metrics, Mapping) or dataset_id is None:
                continue
            decoding = "multi" if "multi" in path.stem else "single"
            extra = {}
            for key in ("coverage_rate", "optimal_coverage_rate"):
                if key in payload:
                    extra[key] = payload[key]
            rows.append(_metric_row(run_id, dataset_id, decoding, path, metrics, extra))

        for path in sorted(run.path.glob("mp_*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            results = payload.get("results") or []
            if not results or not isinstance(results[0], Mapping):
                continue
            dataset_id = _dataset_id_from_path(payload.get("data", ""))
            result = results[0]
            metrics = result.get("multi_best")
            if not dataset_id or not isinstance(metrics, Mapping):
                continue
            label = f"multi k={payload.get('top_k', '?')} / {payload.get('null_policy', 'stop')}"
            rows.append(
                _metric_row(
                    run_id,
                    dataset_id,
                    label,
                    path,
                    metrics,
                    {
                        "coverage_rate": result.get("coverage_rate"),
                        "optimal_coverage_rate": result.get("optimal_coverage_rate"),
                    },
                )
            )

    # Prefer the newest file when a run/dataset/decoding tuple occurs twice.
    unique: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in rows:
        unique[(row["model"], row["dataset"], row["decoding"])] = row
    return sorted(unique.values(), key=lambda row: (row["dataset"], row["model"], row["decoding"]))


def _graph_payload(sample: Any) -> Dict[str, Any]:
    import networkx as nx

    graph = sample.graph
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
    return {"nodes": nodes, "edges": edges}


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
    return _finite(
        {
            "rank": rank,
            "nodes": node_ids,
            "edges": edges,
            "status": status,
            "reason": reason,
            "cost": int(cost),
            "log_prob": log_prob,
        }
    )


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
        self._dataset_cache: Dict[str, Any] = {}
        self._model_key: Optional[str] = None
        self._model_bundle: Optional[Tuple[Any, Any, Any, Any]] = None
        self._lock = threading.RLock()

    def catalog(self) -> Dict[str, Any]:
        return {
            "models": [
                {"id": item.id, "label": item.label}
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
            "metrics": discover_metrics(self.runs, self.root),
        }

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
            if decode == "single":
                decoded = decode_flat(sample, chain["z0"], decision_offset=0, candidate_offset=0)
                routes = [_route_payload(decoded, 1, sample.graph)]
                summary = {
                    "coverage": decoded.status == "goal",
                    "num_finished": 1,
                    "num_goal_paths": int(decoded.status == "goal"),
                    "pruned": 0,
                }
            else:
                top_k = max(1, min(int(request.get("top_k", 2)), 6))
                beam_width = max(1, min(int(request.get("beam_width", 64)), 256))
                display_paths = max(1, min(int(request.get("display_paths", 8)), 24))
                null_policy = str(request.get("null_policy", "stop"))
                decoded = decode_multi_path(
                    sample,
                    candidate_prob,
                    top_k=top_k,
                    beam_width=beam_width,
                    null_policy=null_policy,
                )
                routes = [
                    _route_payload(path, rank + 1, sample.graph)
                    for rank, path in enumerate(decoded.finished[:display_paths])
                ]
                summary = decoded.summary()

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
                    },
                    "graph": _graph_payload(sample),
                    "routes": routes,
                    "summary": summary,
                }
            if include_diffusion:
                response["diffusion"] = _diffusion_payload(
                    sample, chain["trace"], batch.candidate_owner
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
