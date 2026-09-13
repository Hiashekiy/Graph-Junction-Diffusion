"""Trainer (实施指南第 18-22、26 节).

第一版的训练循环是**整条 reverse chain 的 teacher-forced 展开**：

    forward trajectory -> H_T 初始化一次 -> t=T..1 连续 step -> 每步 CE -> 平均

每个 epoch 结束会在 val split 上：
    1. 算 teacher-forced 单步 accuracy（debug metric）；
    2. 跑完整 reverse chain 算 Goal Hit / Optimal / Cost Ratio / Loop / Broken。

模型选择主指标默认是 goal hit rate，而不是 decision accuracy（指南第 24 节）。
"""

from __future__ import annotations

import json
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
        )
        self.stochastic_sampling = (
            bool(eval_cfg.get("stochastic_sampling", True)) if eval_cfg else True
        )
        self.eval_batch_size = (
            int(eval_cfg.get("batch_size", self.batch_size)) if eval_cfg else self.batch_size
        )
        self.eval_max_steps = int(eval_cfg.get("max_steps", 0)) if eval_cfg else 0

        self.history: List[Dict[str, Any]] = []
        self.global_step = 0
        self.start_epoch = 0
        self.best_metric = float("-inf")

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
        total_acc = 0.0
        start = time.time()
        for batch_index, samples in enumerate(batches):
            batch = collate_samples(samples, device=self.device)
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
            out.loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.global_step += 1

            total_loss += float(out.loss.detach())
            total_acc += out.final_accuracy
            if (batch_index + 1) % self.log_every == 0:
                print(
                    f"[epoch {epoch}] batch {batch_index + 1}/{len(batches)} "
                    f"loss={float(out.loss.detach()):.4f} "
                    f"x0_acc={out.final_accuracy:.3f} "
                    f"({time.time() - start:.1f}s)",
                    flush=True,
                )

        return {
            "train_loss": total_loss / max(len(batches), 1),
            "train_x0_acc": total_acc / max(len(batches), 1),
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
        )
        metrics = dict(report.metrics)
        metrics["val_x0_acc"] = report.debug.get("accuracy", float("nan"))
        metrics["val_one_step_loss"] = report.debug.get("loss", float("nan"))
        if self.run_dir is not None:
            with open(self.run_dir / f"val_records_epoch{epoch}.json", "w", encoding="utf-8") as handle:
                json.dump(records_to_dicts(report.records), handle, indent=1)
        return metrics

    # ------------------------------------------------------------------
    def fit(self, epochs: Optional[int] = None) -> List[Dict[str, Any]]:
        epochs = int(epochs if epochs is not None else self.epochs)
        for epoch in range(self.start_epoch + 1, epochs + 1):
            record = self.train_epoch(epoch)
            if epoch % self.eval_every == 0:
                record.update(self.validate(epoch))
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
        metric = record.get("goal_hit_rate")
        if metric is None:
            metric = -record.get("train_loss", float("inf"))
        if metric > self.best_metric:
            self.best_metric = float(metric)
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
            print(f"[epoch {epoch}] new best (metric={self.best_metric:.4f})", flush=True)
