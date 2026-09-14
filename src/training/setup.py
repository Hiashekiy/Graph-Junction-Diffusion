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
    # Weighted 扩展：model.use_edge_cost 控制 EdgeCostEncoder / k_cost_proj /
    # v_cost_proj 是否被实例化。老配置里没有这个键 -> False -> 参数集合与改动前
    # 完全一致（这是必须写死的 backward-compatible default）。
    edge_cost_cfg = model_cfg.get("edge_cost", {}) or {}
    normalization = str(edge_cost_cfg.get("normalization", "graph_mean"))
    if normalization != "graph_mean":
        raise NotImplementedError(
            f"model.edge_cost.normalization={normalization!r} is not implemented "
            "(only graph_mean). Do NOT use min-max / z-score here: any normalisation "
            "with a shift changes the ordering of paths with different hop counts, "
            "while the objective is the plain sum of edge costs."
        )
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
        # 一个 reverse step 内部的图信息交流轮数（1 = 旧行为）。
        "flow_steps": int(model_cfg.get("flow_steps", 1)),
        "slot_embedding": bool(model_cfg.get("flow_slot_embedding", True)),
        "slot_scale": float(model_cfg.get("flow_slot_scale", 1.0)),
        "use_edge_cost": bool(model_cfg.get("use_edge_cost", False)),
        "edge_cost_hidden": int(
            edge_cost_cfg.get("hidden_dim", model_cfg.get("d_model", 128))
        ),
    }


def build_model(config: Config, device: Optional[torch.device] = None) -> GraphFlowDenoiser:
    kwargs = model_kwargs(config)
    model = GraphFlowDenoiser(**kwargs)
    # weighted 数据集 + 不接受 cost 输入的模型 = 方案第 17 节要求的"cost 消融"，
    # 这是合法配置（它正是用来证明模型真的在利用 edge weight 的对照），但必须显式
    # 提示，否则很容易被误当成一次正常的 weighted 实验。
    if not kwargs["use_edge_cost"] and bool(config.get("data.weighted", False)):
        print(
            "[build_model] data.weighted=true but model.use_edge_cost=false -> "
            "this is the COST-ABLATED configuration (the model cannot see edge "
            "cost); it is only meaningful as an ablation baseline.",
            flush=True,
        )
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
def generator_config(data_cfg: Config) -> Dict[str, Any]:
    """把 config 里的数据生成参数整理成 build_dataset 需要的 ``generator_cfg``。

    支持两类生成器：
      * ``controlled_junction``：难度混合 / 结构模式 / source 比例 / branch 参数；
      * 其它（er/ba/ws/...）：直接透传 ``data.generator``。
    """
    base = dict(data_cfg.get("generator", {}) or {})
    graph_type = str(data_cfg.get("graph_type", "er"))
    if graph_type != "controlled_junction":
        return base

    cfg: Dict[str, Any] = dict(base)
    if "difficulty_mix" in data_cfg:
        cfg["difficulty_mix"] = {
            str(key): float(value)
            for key, value in data_cfg.get("difficulty_mix").items()
        }
    if "structure_mix" in data_cfg:
        cfg["structure_mix"] = {
            str(key): float(value) for key, value in data_cfg.get("structure_mix").items()
        }
    source_cfg = data_cfg.get("source", {}) or {}
    cfg["source_forced_probability"] = float(
        source_cfg.get("forced_probability", 0.70)
    )
    branch_cfg = data_cfg.get("branch", {}) or {}
    if branch_cfg:
        cfg["branch"] = {
            key: list(value) for key, value in branch_cfg.items() if value is not None
        }
    return cfg


def edge_weight_config(data_cfg: Config) -> Dict[str, Any]:
    """把 ``data.edge_weight`` 节取成普通 dict（缺省 = 生成器的默认 U(1, 10)）。"""
    value = data_cfg.get("edge_weight", None)
    if value is None:
        return {}
    if isinstance(value, Config):
        return value.to_dict()
    return dict(value)


def build_datasets(
    config: Config, progress_every: int = 500
) -> Dict[str, GraphQueryDataset]:
    data_cfg = config.section("data")
    split_cfg = config.get("split", {})
    fractions = {
        "train": float(split_cfg.get("train", 0.8)),
        "val": float(split_cfg.get("val", 0.1)),
        "test": float(split_cfg.get("test", 0.1)),
    }
    num_nodes = data_cfg.get("num_nodes", [20, 40])
    generator_cfg = generator_config(data_cfg)
    dataset = build_dataset(
        num_samples=int(data_cfg.get("num_samples", 0)) or _auto_num_samples(config),
        graph_type=str(data_cfg.get("graph_type", "er")),
        num_nodes=[int(num_nodes[0]), int(num_nodes[1])] if isinstance(num_nodes, (list, tuple)) else int(num_nodes),
        min_od_distance=int(data_cfg.get("min_od_distance", 5)),
        seed=int(config.get("seed", 0)),
        queries_per_graph=int(data_cfg.get("num_queries_per_graph", 4)),
        generator_cfg=generator_cfg,
        weighted=bool(data_cfg.get("weighted", False)),
        component_fallback=bool(data_cfg.get("component_fallback", True)),
        progress_every=progress_every,
        edge_weight=edge_weight_config(data_cfg),
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
