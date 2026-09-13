"""Config 测试（实施指南第 28 节：配置必须可被命令行覆盖）。"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.utils.config import Config, apply_overrides, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "graph_flow.yaml"


def test_defaults_are_kept_when_key_is_missing():
    cfg = Config({"data": {"num_samples": 8}})
    assert cfg.get("data.num_samples", 256) == 8
    assert cfg.get("data.not_there", 256) == 256


def test_section_get_keeps_the_default():
    """``cfg.section("data").get(key, default)`` 是实际用到的写法，默认值不能丢。"""
    cfg = Config({"data": {"num_samples": 8}})
    section = cfg.section("data")
    assert section.get("num_samples", 256) == 8
    assert section.get("missing", 256) == 256


def test_attribute_access_and_nested_section():
    cfg = Config({"model": {"d_model": 64}, "seed": 3})
    assert cfg.model.d_model == 64
    assert cfg.get("model.d_model") == 64
    assert cfg.seed == 3
    assert isinstance(cfg.model, Config)


def test_to_dict_returns_plain_dict():
    cfg = Config({"a": {"b": 1}})
    plain = cfg.to_dict()
    assert plain == {"a": {"b": 1}}
    assert isinstance(plain["a"], dict)


def test_apply_overrides_parses_scalars_and_lists():
    data = {"data": {"num_samples": 256}}
    apply_overrides(
        data,
        [
            "data.num_samples=32",
            "data.num_nodes=[18, 28]",
            "training.lr=3e-4",
            "training.amp=false",
        ],
    )
    assert data["data"]["num_samples"] == 32
    assert data["data"]["num_nodes"] == [18, 28]
    assert data["training"]["lr"] == 3e-4
    assert data["training"]["amp"] is False


def test_real_config_loads_with_overrides():
    cfg = load_config(
        CONFIG_PATH,
        ["data.num_samples=32", "data.num_nodes=[18, 28]", "data.min_od_distance=3"],
    )
    assert isinstance(cfg, Config)
    assert cfg.get("data.num_samples", 0) == 32
    assert cfg.get("data.num_nodes") == [18, 28]
    assert cfg.get("data.min_od_distance") == 3
    # 没被覆盖的键仍然是默认值
    assert cfg.section("data").get("graph_type", "er") == "er"
    assert cfg.section("model").get("d_model", 128) == 128


def test_real_config_matches_guide_defaults():
    """实施指南第 28 节列出的第一版默认值。"""
    with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    assert raw["model"]["d_model"] == 128
    assert raw["model"]["node_types"] == 4
    assert raw["model"]["edge_states"] == 2
    assert raw["model"]["ffn_hidden"] == 256
    assert raw["time"]["encoding"] == "sinusoidal"
    assert raw["time"]["d_time"] == 128
    assert raw["time"]["conditioning"] == "adaln"
    assert raw["diffusion"]["T"] == 50
    assert raw["diffusion"]["schedule"] == "linear"
    assert raw["diffusion"]["beta_start"] == 0.02
    assert raw["diffusion"]["beta_end"] == 0.20
    assert raw["diffusion"]["base_noise"] == "uniform"
    assert raw["training"]["optimizer"] == "adamw"
    assert raw["training"]["grad_clip"] == 1.0
    assert raw["loss"]["x0_ce"] == 1.0
    assert raw["evaluation"]["stochastic_sampling"] is True


def test_bad_override_is_rejected():
    with pytest.raises(ValueError):
        apply_overrides({}, ["not_a_pair"])
