"""Builders: 从 config 造出 model / diffusion / optimizer / data (实施指南第 28 节).

所有超参数的默认值都来自 ``configs/graph_flow.yaml``；这个模块只负责把配置翻
译成对象，不包含任何训练逻辑。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn

from src.data.dataset_builder import build_dataset, split_dataset
from src.data.dataset import GraphQueryDataset
from src.diffusion.categorical import CategoricalDiffusion
from src.diffusion.schedule import NoiseSchedule
from src.models.denoiser import GraphFlowDenoiser
from src.utils.config import Config


def get_device(spec: str = "auto") -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config asks for cuda but torch.cuda.is_available() is False")
    return device


# ---------------------------------------------------------------------------
def build_schedule(config: Config) -> NoiseSchedule:
    diffusion_cfg = config.section("diffusion")
    return NoiseSchedule(
        T=int(diffusion_cfg.get("T", 50)),
        schedule=str(diffusion_cfg.get("schedule", "linear")),
        beta_start=float(diffusion_cfg.get("beta_start", 0.02)),
        beta_end=float(diffusion_cfg.get("beta_end", 0.20)),
    )


def build_diffusion(config: Config) -> CategoricalDiffusion:
    return CategoricalDiffusion(
        build_schedule(config),
        base_noise=str(config.get("diffusion.base_noise", "uniform")),
    )


def model_kwargs(config: Config) -> Dict[str, Any]:
    model_cfg = config.section("model")
    time_cfg = config.get("time", {})
    return {
        "d_model": int(model_cfg.get("d_model", 128)),
        "num_node_types": int(model_cfg.get("node_types", 4)),
        "num_edge_states": int(model_cfg.get("edge_states", 2)),
        "ffn_hidden": int(model_cfg.get("ffn_hidden", 256)),
        "dropout": float(model_cfg.get("dropout", 0.0)),
        # P2-2：time 配置项现在真正传进模型；取值未实现会在构造时报错，
        # 而不是被静默忽略。
        "d_time": int(time_cfg.get("d_time", model_cfg.get("d_model", 128))),
        "time_encoding": str(time_cfg.get("encoding", "sinusoidal")),
        "time_conditioning": str(time_cfg.get("conditioning", "adaln")),
    }


def build_model(config: Config, device: Optional[torch.device] = None) -> GraphFlowDenoiser:
    model = GraphFlowDenoiser(**model_kwargs(config))
    if device is not None:
        model = model.to(device)
    return model


def build_optimizer(config: Config, model: nn.Module) -> torch.optim.Optimizer:
    training_cfg = config.section("training")
    name = str(training_cfg.get("optimizer", "adamw")).lower()
    lr = float(training_cfg.get("lr", 1e-4))
    weight_decay = float(training_cfg.get("weight_decay", 1e-4))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    raise ValueError(f"unknown optimizer {name!r}")


# ---------------------------------------------------------------------------
def build_datasets(config: Config) -> Dict[str, GraphQueryDataset]:
    data_cfg = config.section("data")
    split_cfg = config.get("split", {})
    fractions = {
        "train": float(split_cfg.get("train", 0.8)),
        "val": float(split_cfg.get("val", 0.1)),
        "test": float(split_cfg.get("test", 0.1)),
    }
    num_nodes = data_cfg.get("num_nodes", [20, 40])
    dataset = build_dataset(
        num_samples=int(data_cfg.get("num_samples", 0)) or _auto_num_samples(config),
        graph_type=str(data_cfg.get("graph_type", "er")),
        num_nodes=[int(num_nodes[0]), int(num_nodes[1])] if isinstance(num_nodes, (list, tuple)) else int(num_nodes),
        min_od_distance=int(data_cfg.get("min_od_distance", 5)),
        seed=int(config.get("seed", 0)),
        queries_per_graph=int(data_cfg.get("num_queries_per_graph", 4)),
        generator_cfg=dict(data_cfg.get("generator", {}) or {}),
        weighted=bool(data_cfg.get("weighted", False)),
        component_fallback=bool(data_cfg.get("component_fallback", True)),
    )
    splits = split_dataset(dataset, fractions, seed=int(config.get("seed", 0)))
    empty = [name for name, split in splits.items() if len(split) == 0]
    if empty:
        raise RuntimeError(
            f"empty dataset split(s): {empty}. With {len(dataset)} queries the "
            "fractions cannot populate every split; increase data.num_samples or "
            "reduce the number of splits (an empty val split would silently disable "
            "validation)."
        )
    return splits


def _auto_num_samples(config: Config) -> int:
    """config 没写 num_samples 时给一个可用默认值。"""
    return int(config.get("data.num_samples", 256))


def run_directory(config: Config, create: bool = True) -> Path:
    output_dir = Path(str(config.get("paths.output_dir", "outputs/runs")))
    run_name = str(config.get("paths.run_name", "graph_flow"))
    run_dir = output_dir / run_name
    if create:
        run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
