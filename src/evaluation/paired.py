"""同一批 query 上的**配对**比较（逐条对比，而不是比两个率）。

验证/测试集只有 300 条，``goal_hit`` 的标准差约 ±0.03，单看两个率（0.6153 vs
0.6833）分不清是"真的更好"还是随机波动。两次评测跑的是**同一批 query**，所以可以
逐条配对：

    both    : 两个都达标
    only_a  : 只有 A 达标
    only_b  : 只有 B 达标
    neither : 都不达标

携带信息的只有 ``only_a`` / ``only_b``（不一致对）。在"两者无差异"的原假设下，
不一致对应当各占一半，于是可以用 McNemar 的**精确**版本（二项检验）给出 p 值。
这里不依赖 scipy：``p = 2 * P(X <= min(only_a, only_b))``，``X ~ Binomial(n, 0.5)``。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence

METRICS = ("goal_hit", "optimal", "broken", "loop")


def exact_mcnemar_p(only_a: int, only_b: int) -> float:
    """McNemar 精确检验（two-sided）p 值；无不一致对时返回 1.0（无信息）。"""
    total = int(only_a) + int(only_b)
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, k) for k in range(0, min(only_a, only_b) + 1))
    return min(1.0, 2.0 * tail / (2 ** total))


def flag(record: Dict[str, Any], metric: str) -> bool:
    """把一条 ``SampleRecord.to_dict()`` 变成"是否达标"的布尔值。"""
    if metric == "goal_hit":
        return bool(record["goal_hit"])
    if metric == "optimal":
        return bool(record["optimal"])
    if metric == "broken":
        return record["status"] == "broken"
    if metric == "loop":
        return record["status"] == "loop"
    raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")


def paired_counts(
    records_a: Sequence[Dict[str, Any]],
    records_b: Sequence[Dict[str, Any]],
    metric: str,
    indices: Sequence[int],
) -> Dict[str, Any]:
    """在 ``indices`` 指定的 query 上做配对统计。

    ``delta`` 是 ``rate_b - rate_a``。``p_value`` 只对"达标率差异"有意义；
    对 ``broken`` / ``loop`` 这类"越低越好"的指标，看 ``delta`` 的符号即可。
    """
    both = only_a = only_b = neither = 0
    for index in indices:
        hit_a = flag(records_a[index], metric)
        hit_b = flag(records_b[index], metric)
        if hit_a and hit_b:
            both += 1
        elif hit_a:
            only_a += 1
        elif hit_b:
            only_b += 1
        else:
            neither += 1
    total = len(indices)
    rate_a = (both + only_a) / total if total else float("nan")
    rate_b = (both + only_b) / total if total else float("nan")
    return {
        "n": total,
        "both": both,
        "only_a": only_a,
        "only_b": only_b,
        "neither": neither,
        "rate_a": rate_a,
        "rate_b": rate_b,
        "delta": rate_b - rate_a,
        "p_value": exact_mcnemar_p(only_a, only_b),
    }


__all__ = ["METRICS", "exact_mcnemar_p", "flag", "paired_counts"]
