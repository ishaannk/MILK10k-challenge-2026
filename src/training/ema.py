"""Exponential moving average of model weights.

An EMA copy of the weights is close to free and, on a dataset this small, its
validation curve is usually both higher and far less jumpy than the raw weights.
That stability matters here for a second reason: we tune per-class thresholds on
the validation predictions, and thresholds fitted to a noisy checkpoint
generalise worse than thresholds fitted to a smoothed one.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class ModelEMA:
    """Maintain a shadow copy of ``model`` updated as an exponential moving average.

    Parameters
    ----------
    model:
        The live model. A deep copy is taken immediately.
    decay:
        EMA decay. 0.999 over ~150 steps/epoch gives an effective window of
        roughly a thousand steps.
    warmup_steps:
        Ramp the decay in from 0 over this many updates, so the EMA does not stay
        anchored to the random initialisation early on.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, warmup_steps: int = 100, device: str | None = None):
        self.module = copy.deepcopy(model).eval()
        for param in self.module.parameters():
            param.requires_grad_(False)
        if device is not None:
            self.module.to(device)
        self.decay = decay
        self.warmup_steps = max(0, warmup_steps)
        self.updates = 0

    def _current_decay(self) -> float:
        if self.warmup_steps == 0:
            return self.decay
        # Standard timm-style ramp: early updates track the live model closely.
        return self.decay * (1.0 - torch.exp(torch.tensor(-float(self.updates) / self.warmup_steps)).item())

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        decay = self._current_decay()
        ema_state = self.module.state_dict()
        for key, value in model.state_dict().items():
            shadow = ema_state[key]
            if shadow.dtype.is_floating_point:
                shadow.mul_(decay).add_(value.detach().to(shadow.device), alpha=1.0 - decay)
            else:
                # Integer buffers (e.g. num_batches_tracked) are copied, not averaged.
                shadow.copy_(value)

    def state_dict(self) -> dict:
        return {"module": self.module.state_dict(), "updates": self.updates, "decay": self.decay}

    def load_state_dict(self, state: dict) -> None:
        self.module.load_state_dict(state["module"])
        self.updates = state.get("updates", 0)
        self.decay = state.get("decay", self.decay)
