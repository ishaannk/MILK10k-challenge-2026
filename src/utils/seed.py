"""Reproducibility helpers: global seeding and deterministic backends."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True, strict: bool = False) -> None:
    """Seed every RNG we depend on and configure cuDNN.

    Parameters
    ----------
    seed:
        Base seed for ``random``, ``numpy`` and ``torch`` (CPU + all CUDA devices).
    deterministic:
        Turn on ``cudnn.deterministic`` and turn off ``cudnn.benchmark``. Costs a
        little throughput but makes runs comparable, which is what you want when
        an experiment's whole purpose is an A/B against the previous one.
    strict:
        Additionally call ``torch.use_deterministic_algorithms(True)``. This
        raises if any op lacks a deterministic kernel, so it is opt-in: some
        interpolation and pooling kernels have no deterministic variant.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Let cuDNN autotune convolutions; fine when images are a fixed size.
        torch.backends.cudnn.benchmark = True

    if strict:
        # Required by some cuBLAS GEMM kernels to behave deterministically.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` giving each worker a distinct, stable seed.

    Without this, every worker inherits the parent's numpy seed and augmentation
    streams correlate across workers.
    """
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """A CPU generator to hand to ``DataLoader(generator=...)`` for stable shuffling."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
