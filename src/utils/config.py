"""Configuration handling.

A small wrapper around a nested dict so that configs can be written as plain
YAML and still be read with attribute access:

    cfg = load_config("configs/graph_flow.yaml")
    cfg.model.d_model          # 128
    cfg.get("training.lr", 1e-4)

Command line overrides use dotted paths:

    --set training.lr=3e-4 --set model.d_model=64
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import yaml

_MISSING = object()


class Config(Mapping):
    """Read-only nested mapping with attribute access.

    嵌套的 dict 在取出来时会被包成 ``Config``（``__getitem__`` 与 ``get`` 都是），
    所以 ``cfg.section("data").get("num_samples", 256)`` 的行为和顶层一致。

    注意：直接继承 Mapping 会让 ``.get`` 用 Mapping 的默认实现（默认值是 None），
    从而**静默吞掉**调用方给的默认值 —— 这里显式覆写 ``get``，并且用
    ``values()`` / ``items()`` 让 ``dict(cfg)`` 仍然得到普通 dict。
    """

    def __init__(self, data: Optional[Mapping[str, Any]] = None):
        object.__setattr__(self, "_data", dict(data or {}))

    @staticmethod
    def _wrap(value: Any) -> Any:
        return Config(value) if isinstance(value, Mapping) else value

    # -- mapping protocol -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self._wrap(self._data[key])

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def values(self):
        return [self._wrap(value) for value in self._data.values()]

    def items(self):
        return [(key, self._wrap(value)) for key, value in self._data.items()]

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._data:
            return self._wrap(self._data[name])
        raise AttributeError(f"config has no key {name!r}")

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Config({self._data!r})"

    # -- helpers ----------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        """Dotted lookup: ``cfg.get("training.lr", 1e-4)``。

        没有点号时按顶层键查找，所以 ``cfg.get("lr")`` 也能用。
        """
        if "." not in key:
            value = self._data.get(key, _MISSING)
            if value is _MISSING:
                return default
            return self._wrap(value)
        node: Any = self._data
        for part in key.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return self._wrap(node)

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._data)

    def section(self, key: str) -> "Config":
        """取一个子配置节；不存在或不是 mapping 时抛错。"""
        value = self.get(key, _MISSING)
        if value is _MISSING or not isinstance(value, Config):
            raise TypeError(f"{key!r} is not a config section")
        return value


def _parse_scalar(text: str) -> Any:
    """把命令行字符串解析成 YAML 标量 / 列表。

    用 YAML 解析而不是 ``int()``/``float()`` 手工尝试，这样 ``[18, 28]`` 这种
    列表覆盖才能生效（直接当字符串传下去会被 ``int()`` 悄悄吞掉）。

    注意 YAML 1.1 把 ``3e-4`` 当**字符串**（科学计数法需要 ``3.0e-4``），所以
    对解析结果仍是字符串的情况再补一次 ``float()`` 尝试。
    """
    lowered = text.strip().lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("none", "null", "~"):
        return None
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError:
        return text
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def set_inplace(data: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node = data
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value


def flatten_overrides(overrides: Optional[Iterable[Any]]) -> List[str]:
    """把 argparse 的 ``--set`` 值摊平成一维 ``key=value`` 字符串列表。

    两种写法都会遇到：

    * ``action="append"`` -> ``["a.b=1", "c.d=2"]``（已经是扁平的字符串）；
    * ``nargs="*"``       -> ``[["a.b=1"], ["c.d=2"]]``（嵌套列表）。

    不区分就会出事：对扁平字符串列表做 ``for group in overrides for item in group``
    会**按字符拆开**（``"data.num_samples=400"`` 变成 ``'d','a','t','a',...``），
    报错信息是 ``override 'd' is not in key=value form``。这个坑踩过两次
    （scripts/train.py、scripts/generate_dataset.py），所以统一走这个函数。
    """
    flat: List[str] = []
    for item in overrides or []:
        if isinstance(item, str):
            flat.append(item)
        else:
            flat.extend(str(part) for part in item)
    return flat


def apply_overrides(cfg: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    """Apply ``key.sub=value`` strings (values are parsed as YAML scalars)."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override {item!r} is not in key=value form")
        key, _, raw = item.partition("=")
        set_inplace(cfg, key.strip(), _parse_scalar(raw))
    return cfg


def load_config(path: str | Path, overrides: Optional[Iterable[str]] = None) -> Config:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"config root must be a mapping, got {type(data).__name__}")
    apply_overrides(data, overrides or [])
    return Config(data)
