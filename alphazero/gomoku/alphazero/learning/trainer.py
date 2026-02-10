from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.amp import GradScaler, autocast
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from gomoku.alphazero.learning.dataset import GameSample, ReplayDataset, _safe_priority
from gomoku.core.gomoku import Gomoku
from gomoku.utils.config.loader import TrainingConfig
from gomoku.utils.progress import make_progress

__all__ = ["AlphaZeroTrainer"]


@dataclass(slots=True)
class AlphaZeroTrainer:
    """Trainer that performs AlphaZero policy/value updates."""

    train_cfg: TrainingConfig
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    game: Gomoku

    # Internal state fields (init=False)
    device: torch.device = field(init=False)
    scaler: GradScaler = field(init=False)
    use_channels_last: bool = field(init=False)
    enable_tf32: bool = field(init=False)

    def __post_init__(self) -> None:
        """Initialize trainer runtime state."""
        self.device = next(self.model.parameters()).device
        self.use_channels_last = bool(
            getattr(self.train_cfg, "use_channels_last", False)
        )
        self.enable_tf32 = bool(getattr(self.train_cfg, "enable_tf32", True))

        # Scaler must persist state across steps, initialize once here
        self.scaler = GradScaler("cuda", enabled=(self.device.type == "cuda"))

    def _make_dataset(self, samples: Sequence[GameSample] | Dataset) -> Dataset:
        """Wrap input into a dataset if it is not already one."""
        if isinstance(samples, Dataset):
            return samples
        return ReplayDataset(samples=samples, game=self.game)

    def _make_loader(
        self, dataset: Dataset, sampler: WeightedRandomSampler | None = None
    ) -> DataLoader:
        """Create a training DataLoader."""
        num_workers = max(0, int(getattr(self.train_cfg, "dataloader_num_workers", 1)))
        if num_workers == 0:
            return DataLoader(
                dataset,
                batch_size=self.train_cfg.batch_size,
                shuffle=(sampler is None),
                sampler=sampler,
                num_workers=0,
                pin_memory=(self.device.type == "cuda"),
            )
        prefetch_factor = max(
            2, int(getattr(self.train_cfg, "dataloader_prefetch_factor", 2))
        )
        return DataLoader(
            dataset,
            batch_size=self.train_cfg.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=(self.device.type == "cuda"),
            persistent_workers=True,
            prefetch_factor=prefetch_factor,
        )

    def train_one_iteration(
        self,
        samples: Sequence[GameSample] | Dataset,
        *,
        progress_label: str | None = None,
    ) -> dict[str, float]:
        """Run one training iteration and return averaged metrics.

        Parameters
        ----------
        samples :
            GameSample sequence or Dataset.
        progress_label :
            Optional label for progress reporting (currently unused).

        Returns
        -------
        dict[str, float]
            Dictionary of average loss metrics.
        """
        self.model.train()

        # Enable TF32 on Ampere+ GPUs for perf when allowed
        torch.backends.cuda.matmul.allow_tf32 = self.enable_tf32
        torch.backends.cudnn.allow_tf32 = self.enable_tf32

        per_cfg = self.train_cfg.priority_replay
        use_per = bool(per_cfg and getattr(per_cfg, "enabled", False))
        per_alpha = float(per_cfg.alpha) if use_per else 0.0
        per_beta = float(per_cfg.beta) if use_per else 0.0
        per_eps = float(per_cfg.epsilon) if use_per else 0.0

        dataset = self._make_dataset(samples)
        if hasattr(dataset, "include_priority"):
            # Drop priority tensor when PER is disabled to keep 3-tuples
            dataset.include_priority = use_per
        if use_per and hasattr(dataset, "return_index"):
            dataset.return_index = True

        sampler: WeightedRandomSampler | None = None
        per_weight_sum: float | None = None
        per_N: int | None = None

        if use_per and hasattr(dataset, "priorities"):
            priorities = np.asarray(dataset.priorities, dtype=np.float32)
            safe_p = np.maximum(priorities, 0.0) + per_eps
            weights = safe_p**per_alpha
            per_weight_sum = float(weights.sum())
            per_N = int(len(weights))
            if per_weight_sum <= 0 or per_N <= 0:
                use_per = False
            else:
                sampler = WeightedRandomSampler(
                    weights=torch.tensor(weights, dtype=torch.float32),
                    num_samples=per_N,
                    replacement=True,
                )

        loader = self._make_loader(dataset, sampler=sampler)

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_samples = 0
        total_batches = 0

        progress = make_progress(
            total=self.train_cfg.num_epochs * len(loader),
            desc=progress_label,
            unit="batch",
            disable=progress_label is None,
            dynamic_ncols=True,
        )

        for _ in range(self.train_cfg.num_epochs):
            for batch in loader:
                if use_per:
                    if getattr(dataset, "return_index", False):
                        (
                            batch_indices,
                            states,
                            policy_targets,
                            value_targets,
                            priorities,
                        ) = batch
                    else:
                        states, policy_targets, value_targets, priorities = batch
                    priorities = priorities.to(self.device, non_blocking=True)
                else:
                    states, policy_targets, value_targets = batch

                if self.use_channels_last:
                    states = states.to(
                        self.device,
                        non_blocking=True,
                        memory_format=torch.channels_last,
                    )
                else:
                    states = states.to(self.device, non_blocking=True)

                policy_targets = policy_targets.to(self.device, non_blocking=True)
                value_targets = value_targets.to(self.device, non_blocking=True)

                with autocast(device_type=self.device.type, dtype=torch.float16):
                    out_policy, out_value = self.model(states)

                    # policy_loss = -∑_i π_i log p_i (visit-based cross entropy)
                    log_probs = F.log_softmax(out_policy, dim=1)
                    policy_losses = -(policy_targets * log_probs).sum(dim=1)

                    # value_loss = (v_pred - z)^2 (scalar MSE)
                    value_losses = F.mse_loss(
                        out_value.squeeze(), value_targets.squeeze(), reduction="none"
                    )

                    combined_losses = policy_losses + value_losses

                    if use_per and per_weight_sum and per_N:
                        # Importance correction using sampling probabilities
                        w = torch.pow(priorities + per_eps, per_alpha)
                        prob = w / per_weight_sum
                        imp_w = torch.pow(1.0 / (prob * per_N), per_beta)
                        imp_w = imp_w / imp_w.max().clamp(min=1e-12)
                        combined_losses = combined_losses * imp_w

                    loss = combined_losses.mean()

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                batch_size = states.size(0)
                policy_loss = policy_losses.mean()
                value_loss = value_losses.mean()

                total_samples += batch_size
                total_loss += float(loss.item()) * batch_size
                total_policy_loss += float(policy_loss.item()) * batch_size
                total_value_loss += float(value_loss.item()) * batch_size
                total_batches += 1
                progress.update(1)

                # Update PER priorities with fresh TD-error estimates (value head)
                if use_per and getattr(dataset, "priorities", None) is not None:
                    with torch.no_grad():
                        td_errors = torch.abs(
                            out_value.detach().squeeze()
                            - value_targets.detach().squeeze()
                        )
                        new_priorities = (td_errors + per_eps).cpu().tolist()
                    if getattr(dataset, "return_index", False):
                        for idx_val, p_val in zip(
                            batch_indices, new_priorities, strict=False
                        ):
                            dataset.priorities[int(idx_val)] = _safe_priority(
                                float(p_val)
                            )

        progress.close()

        # Return averaged metrics; guard against empty loaders
        if total_samples == 0:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}

        return {
            "loss": total_loss / total_samples,
            "policy_loss": total_policy_loss / total_samples,
            "value_loss": total_value_loss / total_samples,
            "last_batch_loss": float(loss.item()) if total_batches else 0.0,
            "last_policy_loss": float(policy_loss.item()) if total_batches else 0.0,
            "last_value_loss": float(value_loss.item()) if total_batches else 0.0,
        }
