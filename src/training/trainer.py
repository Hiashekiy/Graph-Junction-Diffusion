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
    except Exception:  # noqa: BLE001 - 可选依赖，失败不是错误
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

        self.weights = LossWeights(
            x0_ce=float(loss_cfg.get("x0_ce", 1.0)) if loss_cfg else 1.0,
            null_weight=float(loss_cfg.get("null_weight", 1.0)) if loss_cfg else 1.0,
            active_weight=float(loss_cfg.get("active_weight", 1.0)) if loss_cfg else 1.0,
            goal_reach_weight=float(loss_cfg.get("goal_reach_weight", 0.1))
            if loss_cfg
            else 0.1,
            goal_reach_eps=float(loss_cfg.get("goal_reach_eps", 1e-8)) if loss_cfg else 1e-8,
            goal_timestep_weighting=str(
                loss_cfg.get("goal_timestep_weighting", "alpha_bar")
            )
            if loss_cfg
            else "alpha_bar",
            goal_horizon_cap=(
                int(loss_cfg.get("goal_horizon_cap"))
                if loss_cfg and loss_cfg.get("goal_horizon_cap") is not None
                else None
            ),
        )
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
            if (batch_index + 1) % self.log_every == 0:
                # 拆开的日志：只看总 loss 分不清是 CE 没学好还是 Goal reachability
                # 没起来（第二轮修订第十二条）。
                print(
                    f"[epoch {epoch}] batch {batch_index + 1}/{len(batches)} "
                    f"loss={float(out.loss.detach()):.4f} "
                    f"ce={float(out.ce_loss.detach()):.4f} "
                    f"goal={float(out.goal_loss.detach()):.4f} "
                    f"soft_goal={out.soft_goal_mean:.4f} "
                    f"x0_acc={out.final_accuracy:.3f} "
                    f"({time.time() - start:.1f}s)",
                    flush=True,
                )

        batches_done = max(len(batches), 1)
        return {
            "train_loss": total_loss / batches_done,
            "train_ce_loss": total_ce / batches_done,
            "train_goal_loss": total_goal / batches_done,
            "train_soft_goal": total_soft_goal / batches_done,
            "train_x0_acc": total_acc / batches_done,
            "train_seconds": time.time() - start,
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, Any]:
        if self.val_dataset is None or len(self.val_dataset) == 0:
            return {}
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
        )
        metrics = dict(report.metrics)
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
    def _log(self, epoch: int, record: Dict[str, Any]) -> None:
        message = f"[epoch {epoch}] " + " ".join(
            f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in record.items()
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
