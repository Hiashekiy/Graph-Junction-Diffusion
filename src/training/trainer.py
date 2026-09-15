"""Trainer (实施指南第 18-22、26 节).

第一版的训练循环是**整条 reverse chain 的 teacher-forced 展开**：

    forward trajectory -> H_T 初始化一次 -> t=T..1 连续 step -> 每步 CE -> 平均

每个 epoch 结束会在 val split 上：
    1. 算 teacher-forced 单步 accuracy（debug metric）；
    2. 跑完整 reverse chain 算 Goal Hit / Optimal / Cost Ratio / Loop / Broken。

模型选择主指标默认是 goal hit rate，而不是 decision accuracy（指南第 24 节）。

**best.pt 的选择指标可配置**（方案第 15 节）：

    training.selection_metric: goal_hit_rate        （默认，旧行为）
                              path_similarity_score （真实 DiDi 数据用）
    training.selection_mode:   max | min            （默认 max）

真实数据上 GoalHit 会较早饱和，而"路径与真实司机路线有多像"还在继续改善，
所以 DiDi 配置改用 ``path_similarity_score``（未到达 goal 的 query 记 0）。
两个键的默认值都写死成旧行为，老配置读出来完全不变。
"""

from __future__ import annotations

import contextlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from src.data.collate import collate_samples, iter_batches
from src.diffusion.categorical import CategoricalDiffusion
from src.evaluation.evaluator import evaluate_dataset, records_to_dicts
from src.models.denoiser import GraphFlowDenoiser
from src.training.checkpoint import save_checkpoint
from src.training.losses import LossWeights, recurrent_reverse_loss

#: Path NLL + Sampled NULL 的拆分日志字段（loss_type="ce" 时全是 NaN，会被跳过）
SAMPLED_NULL_METRICS = (
    "path_nll",
    "sampled_null_loss",
    "active_branch_acc",
    "mean_gt_branch_prob",
    "sampled_null_acc",
    "mean_sampled_null_prob",
    "pred_active_rate",
    "mean_num_active",
    "mean_num_sampled_null",
    # 饱和式 NULL 专属：p(NULL) 已经到 rho_null 的 decision 占比。它到 1.0 就说明
    # 这一项已经"失去梯度"，再涨 lambda_null_local 也没用，该去调 rho_null。
    "null_saturation_rate",
)

#: 多轨迹集合损失（trajectory.enabled=false 时全是 NaN，会被跳过）。
#:
#: 这些是**诊断 L_traj 到底在干什么**的唯一窗口：只看 trajectory_loss 掉下来
#: 分不清是"成功轨迹拿到质量了"还是"失败代价被压掉了"。至少要一起看
#: traj_success_mass（越大越好）与 traj_fail_*_mass（越小越好）。
TRAJECTORY_METRICS = (
    "trajectory_loss",
    "traj_success_loss",
    "traj_similarity_loss",
    "traj_failure_loss",
    # 集合质量分布：success_mass + failure_mass == 1
    "traj_success_mass",
    "traj_failure_mass",
    # 候选池规模
    "traj_num_candidates",
    "traj_num_success",
    "traj_num_failure",
    # 失败类型细分（按集合质量，不是按条数）
    "traj_fail_null_mass",
    "traj_fail_loop_mass",
    "traj_fail_dead_mass",
    "traj_fail_broken_mass",
    # 成功轨迹与 GT 的平均相似度（nLCS，1.0 = 完全一致）
    "traj_mean_success_nlcs",
    # miner 在截断到 max_success / max_failure **之前**挖到多少条 —— mining
    # budget 消融用：如果 raw_* 一直贴着上限，说明 beam 再放大还有东西可挖。
    "traj_raw_finished",
    "traj_raw_success",
    "traj_raw_null",
    "traj_raw_loop",
    "traj_raw_dead_end",
    "traj_raw_broken",
    # 被剔除的空 trace（模型零 decision，不可学）。持续偏高 = corridor 的 OD 太浅，
    # 多轨迹项本身没信息可学，此时该去查数据而不是调超参。
    "traj_raw_no_decision",
)

#: 上面两组一起进 history.json / 控制台（getattr 取不到或 NaN 的自动跳过）
EXTRA_TRAIN_METRICS = SAMPLED_NULL_METRICS + TRAJECTORY_METRICS


def _try_load_coordinates(config) -> Optional[Dict[str, Any]]:
    """尽力加载 ``data.coords_file``；失败/缺失返回 None（DTW 自动降级为 NaN）。"""
    if config is None:
        return None
    raw = config.get("data.coords_file", None)
    if not raw:
        return None
    from pathlib import Path as _Path

    from src.data import didi_dataset as _didi

    path = _Path(str(raw))
    if not path.is_absolute():
        path = _Path(__file__).resolve().parents[2] / path
    if not path.exists():
        print(f"[trainer] data.coords_file not found ({path}) -> DTW disabled", flush=True)
        return None
    # 补缺失节点（原始坐标只覆盖 ~96%），否则部分样本的 DTW 会静默变 NaN
    graph_path = config.get("paths.data_dir", None)
    graph_file = None
    if graph_path:
        candidate = _Path(str(graph_path)) / "graph_global.pkl"
        if not candidate.is_absolute():
            candidate = _Path(__file__).resolve().parents[2] / candidate
        if candidate.exists():
            graph_file = candidate
    try:
        coordinates, _stats = _didi.load_node_coordinates_filled(path, graph_file)
        return coordinates
    except Exception as error:  # noqa: BLE001 - 可选依赖，失败不是错误，但要出声
        print(
            f"[trainer] could not load coordinates ({type(error).__name__}: {error}) "
            "-> DTW disabled",
            flush=True,
        )
        return None


class Trainer:
    def __init__(
        self,
        model: GraphFlowDenoiser,
        diffusion: CategoricalDiffusion,
        optimizer: torch.optim.Optimizer,
        train_dataset,
        val_dataset=None,
        config=None,
        device: torch.device | str = "cpu",
        run_dir: Optional[str | Path] = None,
        generator: Optional[torch.Generator] = None,
    ):
        self.model = model
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = torch.device(device)
        # 防御：模型必须和 batch 同类型设备（比较 device.type，这样 "cuda" 与
        # "cuda:0" 不会被误判为不同设备）
        parameter = next(self.model.parameters(), None)
        if parameter is not None and parameter.device.type != self.device.type:
            raise ValueError(
                f"model is on {parameter.device} but Trainer.device={self.device}; "
                "move the model before constructing the Trainer"
            )
        self.generator = generator
        self.run_dir = Path(run_dir) if run_dir is not None else None
        if self.run_dir is not None:
            self.run_dir.mkdir(parents=True, exist_ok=True)

        training_cfg = config.section("training") if config is not None else None
        loss_cfg = config.section("loss") if config is not None else None
        eval_cfg = config.section("evaluation") if config is not None else None

        self.batch_size = int(training_cfg.get("batch_size", 8)) if training_cfg else 8
        self.epochs = int(training_cfg.get("epochs", 50)) if training_cfg else 50
        self.grad_clip = float(training_cfg.get("grad_clip", 1.0)) if training_cfg else 1.0
        self.max_bptt_steps = int(training_cfg.get("max_bptt_steps", 0)) if training_cfg else 0
        self.log_every = int(training_cfg.get("log_every", 1)) if training_cfg else 1
        self.eval_every = int(training_cfg.get("eval_every", 1)) if training_cfg else 1
        self.use_amp = bool(training_cfg.get("amp", False)) if training_cfg else False
        self.seed = int(config.get("seed", 0)) if config is not None else 0

        self.weights = LossWeights.from_config(config)
        self.weights.validate()
        # km-based DTW 需要节点经纬度（可选文件）。训练期间也把它带上，这样
        # history.json 里的 val 指标与最终评测口径一致；加载失败就静默降级
        # （DTW 记 NaN），绝不让一个可选地理文件把训练搞挂。
        self.coordinates = _try_load_coordinates(config)

        self.stochastic_sampling = (
            bool(eval_cfg.get("stochastic_sampling", True)) if eval_cfg else True
        )
        self.eval_batch_size = (
            int(eval_cfg.get("batch_size", self.batch_size)) if eval_cfg else self.batch_size
        )
        self.eval_max_steps = int(eval_cfg.get("max_steps", 0)) if eval_cfg else 0

        # ---- 验证 / 选 best.pt 用的解码口径 ----------------------------------
        #
        # 默认值**逐位等于 evaluate_dataset() 的默认形参**（single / 非 strict /
        # beam 64），所以没写这些键的旧 config（controlled_unweighted /
        # controlled_weighted）的验证行为一个字节都没变。
        #
        # 为什么必须可配：best.pt 是按验证集指标挑的。如果验证还在用 single、而最终
        # 推理用 strict 多分支，碰到"single 后期变差、strict 后期反而继续提升"时就会
        # 选中错误的 checkpoint —— 这个坑本项目真实踩过。让 scripts/evaluate.py 与这里
        # 读**同一组 config 键**，口径就不可能再漂。
        #
        # 键名与 scripts/evaluate.py 的命令行参数一一对应：
        #   decode <-> --decode, top_k <-> --top-k, beam_width <-> --beam-width,
        #   null_policy <-> --null-policy,
        #   filter_dead_branches <-> --filter-dead-branches,
        #   strict_decode <-> --strict-decode
        self.eval_decode = (
            str(eval_cfg.get("decode", "single")).lower() if eval_cfg else "single"
        )
        if self.eval_decode not in ("single", "multi"):
            raise ValueError(
                f"evaluation.decode={self.eval_decode!r} is not supported "
                "(choose one of 'single' | 'multi')"
            )
        self.eval_strict_decode = (
            bool(eval_cfg.get("strict_decode", False)) if eval_cfg else False
        )
        self.eval_top_k = int(eval_cfg.get("top_k", 2)) if eval_cfg else 2
        self.eval_beam_width = int(eval_cfg.get("beam_width", 64)) if eval_cfg else 64
        self.eval_null_policy = (
            str(eval_cfg.get("null_policy", "stop")) if eval_cfg else "stop"
        )
        self.eval_filter_dead_branches = (
            bool(eval_cfg.get("filter_dead_branches", False)) if eval_cfg else False
        )
        if self.eval_strict_decode and self.eval_decode != "multi":
            # strict 只存在于多分支解码器里。配错的话会静默地按 single 跑，
            # 训练全程以为自己用的是 strict 2/3 —— 必须直接报错。
            raise ValueError(
                "evaluation.strict_decode=true 需要 evaluation.decode=multi"
                f"（当前 decode={self.eval_decode!r}）"
            )

        # P2-2：AMP 真正落地。只有 cuda + 显式开启才启用，并且把 scaler 状态
        # 一起交给 optimizer（已创建的 scaler 也能在 CPU 上安全存在）。
        self.amp_enabled = bool(self.use_amp and self.device.type == "cuda")
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=True) if self.amp_enabled else None
        )

        self.history: List[Dict[str, Any]] = []
        self.global_step = 0
        self.start_epoch = 0
        self.best_metric = float("-inf")

        # ---- best.pt 的选择指标（方案第 15 节）---------------------------
        # 默认值必须写死成 goal_hit_rate / max：老配置里没有这两个键，读出来就是
        # 旧行为，旧 run 的 best.pt 语义逐位不变。
        # 真实 DiDi 数据用 path_similarity_score：GoalHit 在真实数据上会较早饱和，
        # 而"路径像不像司机走的那条"还在继续改善；而且该指标对没到 goal 的 query
        # 记 0，不会让"没到终点但前半段很像"拿到虚高分。
        self.selection_metric = (
            str(training_cfg.get("selection_metric", "goal_hit_rate"))
            if training_cfg
            else "goal_hit_rate"
        )
        self.selection_mode = (
            str(training_cfg.get("selection_mode", "max")).lower()
            if training_cfg
            else "max"
        )
        if self.selection_mode not in ("max", "min"):
            raise ValueError(
                f"training.selection_mode={self.selection_mode!r} is not "
                "'max' or 'min'"
            )

    # ------------------------------------------------------------------
    def _autocast(self):
        """训练用的 autocast 上下文（未开 AMP 时是 no-op）。"""
        if not self.amp_enabled:
            return contextlib.nullcontext()
        return torch.amp.autocast("cuda", dtype=torch.float16)

    # ------------------------------------------------------------------
    def train_epoch(self, epoch: int) -> Dict[str, Any]:
        self.model.train()
        batches = iter_batches(
            list(self.train_dataset),
            batch_size=self.batch_size,
            shuffle=True,
            seed=self.seed + epoch,
        )
        print(f"[epoch {epoch}] {len(batches)} batches", flush=True)

        total_loss = 0.0
        total_ce = 0.0
        total_goal = 0.0
        total_soft_goal = 0.0
        total_acc = 0.0
        extra_total: Dict[str, float] = {}
        extra_count: Dict[str, int] = {}
        start = time.time()
        for batch_index, samples in enumerate(batches):
            batch = collate_samples(samples, device=self.device)
            # P2-2：training.amp 现在真的控制 autocast + GradScaler，
            # 不再是一个只被读取、不生效的配置项。
            with self._autocast():
                out = recurrent_reverse_loss(
                    self.model,
                    self.diffusion,
                    batch,
                    weights=self.weights,
                    generator=self.generator,
                    max_steps=self.diffusion.T,
                    truncate_every=self.max_bptt_steps,
                )
            self.optimizer.zero_grad(set_to_none=True)
            if self.scaler is not None:
                self.scaler.scale(out.loss).backward()
                if self.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                out.loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                self.optimizer.step()
            self.global_step += 1

            total_loss += float(out.loss.detach())
            total_ce += float(out.ce_loss.detach())
            total_goal += float(out.goal_loss.detach())
            total_soft_goal += float(out.soft_goal_mean)
            total_acc += out.final_accuracy
            for metric_name in EXTRA_TRAIN_METRICS:
                metric_value = getattr(out, metric_name, None)
                if metric_value is None or metric_value != metric_value:
                    continue  # 该 loss_type 不产出这个字段
                extra_total[metric_name] = (
                    extra_total.get(metric_name, 0.0) + float(metric_value)
                )
                extra_count[metric_name] = extra_count.get(metric_name, 0) + 1
            if (batch_index + 1) % self.log_every == 0:
                # 拆开的日志：只看总 loss 分不清是 CE 没学好还是 Goal reachability
                # 没起来（第二轮修订第十二条）。
                #
                # traj / null_sat 只在对应目标真的启用时才有值（否则是 NaN），
                # 所以按需拼，不让不带这些项的 run 刷一屏 "nan"。
                bits = []
                for label, name, fmt in (
                    ("null_sat", "null_saturation_rate", ".3f"),
                    ("traj", "trajectory_loss", ".3f"),
                    ("sm", "traj_success_mass", ".3f"),
                    ("ncand", "traj_num_candidates", ".0f"),
                ):
                    value = getattr(out, name, None)
                    if value is None or value != value:
                        continue
                    bits.append(f"{label}={float(value):{fmt}} ")
                print(
                    f"[epoch {epoch}] batch {batch_index + 1}/{len(batches)} "
                    f"loss={float(out.loss.detach()):.4f} "
                    f"ce={float(out.ce_loss.detach()):.4f} "
                    f"goal={float(out.goal_loss.detach()):.4f} "
                    f"soft_goal={out.soft_goal_mean:.4f} "
                    f"x0_acc={out.final_accuracy:.3f} "
                    f"act_acc={getattr(out, 'active_branch_acc', float('nan')):.3f} "
                    f"gt_p={getattr(out, 'mean_gt_branch_prob', float('nan')):.3f} "
                    f"{''.join(bits)}"
                    f"({time.time() - start:.1f}s)",
                    flush=True,
                )

        batches_done = max(len(batches), 1)
        # 统一加 train_ 前缀，和 train_loss / train_x0_acc 一致，也避免和
        # 评测侧的指标名混淆
        record = {
            f"train_{key}": extra_total[key] / max(extra_count[key], 1)
            for key in extra_total
        }
        return {
            **record,
            "train_loss": total_loss / batches_done,
            "train_ce_loss": total_ce / batches_done,
            "train_goal_loss": total_goal / batches_done,
            "train_soft_goal": total_soft_goal / batches_done,
            # "对**全部** decision 等权"的 teacher-forced 准确率。
            # 用 Path NLL + Sampled NULL 时它不是训练目标（模型只被监督 active +
            # 采样到的 NULL），把它当 headline 会严重误读 —— 实测它会掉到 0.05，
            # 而真正该看的 active_branch_acc 在同期从 0.37 涨到 0.54。
            # 所以新目标下改名成 all_decision_acc（信息保留、语义显式），
            # 并且不进主日志。
            ("train_all_decision_acc" if self.weights.is_sampled_null else "train_x0_acc"): (
                total_acc / batches_done
            ),
            "train_seconds": time.time() - start,
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, Any]:
        if self.val_dataset is None or len(self.val_dataset) == 0:
            return {}
        # 口径完全由 config 的 `evaluation.*` 决定（见 __init__ 里的说明）：
        # 验证、选 best.pt、最终 evaluate.py 三者必须用同一把尺子。
        report = evaluate_dataset(
            self.model,
            self.diffusion,
            self.val_dataset,
            batch_size=self.eval_batch_size,
            stochastic=self.stochastic_sampling,
            device=self.device,
            generator=self.generator,
            max_steps=self.eval_max_steps or None,
            progress=False,
            weights=self.weights,
            coordinates=self.coordinates,
            decode=self.eval_decode,
            top_k=self.eval_top_k,
            beam_width=self.eval_beam_width,
            null_policy=self.eval_null_policy,
            filter_dead_branches=self.eval_filter_dead_branches,
            strict_decode=self.eval_strict_decode,
        )
        metrics = dict(report.metrics)
        if self.weights.is_sampled_null:
            # 同上：这两个是"全部 decision"口径的旧 debug 指标，改名后保留
            metrics["val_all_decision_acc"] = report.debug.get("accuracy", float("nan"))
            metrics["val_all_decision_ce"] = report.debug.get("loss", float("nan"))
        else:
            metrics["val_x0_acc"] = report.debug.get("accuracy", float("nan"))
            metrics["val_one_step_loss"] = report.debug.get("loss", float("nan"))
        metrics["val_one_step_soft_goal"] = report.debug.get("soft_goal", float("nan"))
        if self.run_dir is not None:
            with open(self.run_dir / f"val_records_epoch{epoch}.json", "w", encoding="utf-8") as handle:
                json.dump(records_to_dicts(report.records), handle, indent=1)
        return metrics

    # ------------------------------------------------------------------
    def _history_path(self) -> Optional[Path]:
        return None if self.run_dir is None else self.run_dir / "history.json"

    def _load_history(self) -> List[Dict[str, Any]]:
        """把磁盘上已有的 history.json 读回来（resume 时保持完整曲线）。

        没有这一步的话，续训会用新列表覆盖 history.json，前面几轮的曲线就永久丢了
        （真实踩过：第一段 1..20 轮的逐轮记录被第二段覆盖）。
        """
        path = self._history_path()
        if path is None or not path.exists():
            return []
        try:
            with open(path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
        except (json.JSONDecodeError, OSError):
            return []
        if not isinstance(loaded, list):
            return []
        # 只保留 epoch <= start_epoch 的旧记录，避免重复
        keep = [
            record
            for record in loaded
            if isinstance(record, dict)
            and int(record.get("epoch", 0)) <= self.start_epoch
        ]
        return sorted(keep, key=lambda record: int(record.get("epoch", 0)))

    def fit(self, epochs: Optional[int] = None) -> List[Dict[str, Any]]:
        epochs = int(epochs if epochs is not None else self.epochs)
        if not self.history:
            self.history = self._load_history()
            if self.history:
                print(
                    f"loaded {len(self.history)} previous epoch records from history.json",
                    flush=True,
                )
        for epoch in range(self.start_epoch + 1, epochs + 1):
            record = self.train_epoch(epoch)
            if epoch % self.eval_every == 0:
                record.update(self.validate(epoch))
            # 显式记录真实 epoch 号：resume 之后 history 是拼接的列表，
            # 靠列表下标推 epoch 会算错。
            record["epoch"] = epoch
            self.history.append(record)
            self._log(epoch, record)
            self._maybe_save(epoch, record)
        return self.history

    # ------------------------------------------------------------------
    #: 不进主日志的字段（仍然完整写进 history.json）。
    #: 这些是"对全部 decision 等权"的旧 debug 口径 —— 用 Path NLL + Sampled NULL 时
    #: 模型只被监督 active + 采样到的 NULL，把它们当 headline 会误导。
    LOG_HIDDEN = ("train_all_decision_acc", "val_all_decision_acc", "val_all_decision_ce")

    def _log(self, epoch: int, record: Dict[str, Any]) -> None:
        message = f"[epoch {epoch}] " + " ".join(
            f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in record.items()
            if key not in self.LOG_HIDDEN
        )
        print(message, flush=True)
        if self.run_dir is not None:
            with open(self.run_dir / "history.json", "w", encoding="utf-8") as handle:
                json.dump(self.history, handle, indent=1)

    def _selection_score(self, record: Dict[str, Any]) -> Optional[float]:
        """把 ``record`` 里的选择指标转成"越大越好"的分值。

        ``None`` 表示"这一轮不更新 best.pt"。两种情况：

        * 配置的就是默认的 ``goal_hit_rate`` —— 走**历史回退链**
          （``goal_hit_rate`` -> ``-train_loss``），旧 run 的 best.pt 逐位不变；
        * 配置了别的指标（例如真实数据的 ``path_similarity_score``）而这一轮
          没算出来（``eval_every`` 没到、或 val 里没有观测 GT）—— **直接跳过**。
          退回 ``-train_loss`` 会把"这一轮没评测"当成"表现变好了"，是很隐蔽的
          选模型 bug。
        """
        if self.selection_metric == "goal_hit_rate":
            metric = record.get("goal_hit_rate")
            if metric is None:
                metric = -record.get("train_loss", float("inf"))
        else:
            metric = record.get(self.selection_metric)
            if metric is None:
                return None
        try:
            value = float(metric)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        return value if self.selection_mode == "max" else -value

    def _maybe_save(self, epoch: int, record: Dict[str, Any]) -> None:
        if self.run_dir is None:
            return
        save_checkpoint(
            self.run_dir / "last.pt",
            self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            global_step=self.global_step,
            best_metric=self.best_metric,
            model_config=self.config.to_dict().get("model") if self.config else None,
            diffusion_config=self.config.to_dict().get("diffusion") if self.config else None,
        )
        score = self._selection_score(record)
        if score is None:
            return
        if score > self.best_metric:
            self.best_metric = float(score)
            save_checkpoint(
                self.run_dir / "best.pt",
                self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                global_step=self.global_step,
                best_metric=self.best_metric,
                model_config=self.config.to_dict().get("model") if self.config else None,
                diffusion_config=self.config.to_dict().get("diffusion") if self.config else None,
            )
            print(
                f"[epoch {epoch}] new best "
                f"({self.selection_metric}/{self.selection_mode}="
                f"{record.get(self.selection_metric, record.get('goal_hit_rate'))})",
                flush=True,
            )
