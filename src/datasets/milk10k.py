"""Paired-image PyTorch ``Dataset`` and ``DataLoader`` factories for MILK10k.

One dataset item == one **lesion**, which is also the unit of prediction and of
the submission file. Each item can carry:

* ``derm``  - the dermoscopic image tensor
* ``clin``  - the clinical close-up image tensor
* ``meta``  - the encoded tabular feature vector
* ``target``- an 11-dim one-hot float vector (absent for the test split)

Which image tensors are materialised is controlled by ``mode``, so the same class
backs all three model stages without loading pixels the model will not consume:

===============  ==============================  ==========================
mode             loads                           used by
===============  ==============================  ==========================
``dermoscopic``  dermoscopic only                Stage 1 baseline
``clinical``     clinical only                   ablation
``both``         both images                     Stages 2 and 3
===============  ==============================  ==========================
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from ..constants import CLASSES, NUM_CLASSES
from ..utils.config import Config
from ..utils.logging_utils import get_logger
from ..utils.seed import make_generator, seed_worker
from .metadata import MetadataProcessor
from .transforms import build_transforms_from_config

LOGGER = get_logger(__name__)

# Reading JPEGs with OpenCV inside DataLoader workers: cap OpenCV's own thread
# pool, otherwise every worker spawns N threads and they thrash the CPU.
cv2.setNumThreads(0)

VALID_MODES = {"dermoscopic", "clinical", "both"}


class MILK10kDataset(Dataset):
    """Lesion-level dataset over paired clinical/dermoscopic images.

    Parameters
    ----------
    frame:
        Lesion-level table from :func:`src.datasets.metadata.build_lesion_table`.
        Must contain ``lesion_id``, ``isic_id_clin`` and ``isic_id_derm``; label
        columns are required only when ``has_targets`` is true.
    image_root:
        Directory holding ``<lesion_id>/<isic_id>.jpg`` subfolders.
    transforms:
        ``{"clinical": Compose, "dermoscopic": Compose}``. Applied independently to
        the two images of a pair -- they are genuinely different photographs, so
        correlating their augmentations would only throw away variety.
    mode:
        One of :data:`VALID_MODES`.
    meta_features:
        Pre-encoded ``(n_lesions, dim)`` array aligned row-for-row with ``frame``.
        Pass ``None`` to omit metadata from the batch.
    has_targets:
        Whether ``frame`` carries the 11 label columns.
    image_layout:
        ``"nested"`` for MILK10k's ``<lesion_id>/<isic_id>.jpg``, or ``"flat"`` for
        ``<isic_id>.jpg`` directly under ``image_root``. The flat layout is what the
        external ISIC challenge archives ship, so supporting both lets the same
        Dataset back external pretraining without a copy or symlink farm.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        image_root: str | Path,
        transforms: dict[str, Callable] | None = None,
        mode: str = "dermoscopic",
        meta_features: np.ndarray | None = None,
        has_targets: bool = True,
        image_layout: str = "nested",
    ):
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(VALID_MODES)}, got {mode!r}")
        if image_layout not in ("nested", "flat"):
            raise ValueError(f"image_layout must be 'nested' or 'flat', got {image_layout!r}")

        self.frame = frame.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transforms = transforms or {}
        self.mode = mode
        self.image_layout = image_layout
        self.has_targets = has_targets

        if meta_features is not None and len(meta_features) != len(self.frame):
            raise ValueError(
                f"meta_features has {len(meta_features)} rows but frame has {len(self.frame)}"
            )
        self.meta_features = None if meta_features is None else np.asarray(meta_features, dtype=np.float32)

        self.lesion_ids = self.frame["lesion_id"].to_numpy()
        self._clin_ids = self.frame["isic_id_clin"].to_numpy()
        self._derm_ids = self.frame["isic_id_derm"].to_numpy()

        if has_targets:
            missing = [c for c in CLASSES if c not in self.frame.columns]
            if missing:
                raise ValueError(f"has_targets=True but label columns are missing: {missing}")
            self.targets = self.frame[CLASSES].to_numpy(dtype=np.float32)
            # Integer class index, used for stratification and balanced sampling.
            self.labels = self.targets.argmax(axis=1)
        else:
            self.targets = None
            self.labels = None

    def __len__(self) -> int:
        return len(self.frame)

    # -- image IO -----------------------------------------------------------
    def _load_image(self, lesion_id: str, isic_id: str) -> np.ndarray:
        """Read one JPEG as an RGB uint8 array.

        Raises on a missing/corrupt file rather than substituting a blank image:
        silently training on black frames is far worse than failing loudly.
        """
        path = (
            self.image_root / f"{isic_id}.jpg"
            if self.image_layout == "flat"
            else self.image_root / str(lesion_id) / f"{isic_id}.jpg"
        )
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _apply(self, image: np.ndarray, modality: str) -> torch.Tensor:
        transform = self.transforms.get(modality)
        if transform is None:
            # No pipeline configured: hand back a bare CHW float tensor.
            return torch.from_numpy(image.transpose(2, 0, 1).copy()).float().div_(255.0)
        return transform(image=image)["image"]

    # -- item ---------------------------------------------------------------
    def __getitem__(self, index: int) -> dict[str, Any]:
        lesion_id = self.lesion_ids[index]
        sample: dict[str, Any] = {"index": index, "lesion_id": str(lesion_id)}

        if self.mode in ("dermoscopic", "both"):
            image = self._load_image(lesion_id, self._derm_ids[index])
            sample["derm"] = self._apply(image, "dermoscopic")

        if self.mode in ("clinical", "both"):
            image = self._load_image(lesion_id, self._clin_ids[index])
            sample["clin"] = self._apply(image, "clinical")

        if self.meta_features is not None:
            sample["meta"] = torch.from_numpy(self.meta_features[index])

        if self.targets is not None:
            sample["target"] = torch.from_numpy(self.targets[index])
            sample["label"] = int(self.labels[index])

        return sample

    # -- class statistics ---------------------------------------------------
    def class_counts(self) -> np.ndarray:
        """Per-class lesion counts, used for loss weighting and sampling."""
        if self.labels is None:
            return np.zeros(NUM_CLASSES, dtype=np.int64)
        return np.bincount(self.labels, minlength=NUM_CLASSES)


def collate_lesions(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate lesion dicts, stacking tensors and keeping ids as a plain list.

    The default collate would try to tensorise the string ``lesion_id``; we need
    those ids intact to write the submission file.
    """
    out: dict[str, Any] = {
        "lesion_id": [item["lesion_id"] for item in batch],
        "index": torch.tensor([item["index"] for item in batch], dtype=torch.long),
    }
    for key in ("derm", "clin", "meta", "target"):
        if key in batch[0]:
            out[key] = torch.stack([item[key] for item in batch])
    if "label" in batch[0]:
        out["label"] = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return out


def make_balanced_sampler(
    labels: np.ndarray,
    power: float = 0.5,
    num_samples: int | None = None,
    seed: int = 42,
) -> WeightedRandomSampler:
    """Class-rebalancing sampler with a tunable strength.

    Sample weight is ``(1 / class_count) ** power``:

    * ``power = 0`` -> the natural distribution (no rebalancing)
    * ``power = 1`` -> fully balanced; every class appears equally often
    * ``0 < power < 1`` -> a compromise

    Full balancing is rarely optimal here: with ``MAL_OTH`` at 9 lesions out of
    5,240, ``power=1`` would show those 9 images hundreds of times per epoch and
    overfit them hard. A square-root-ish default (0.5) lifts the rare classes
    substantially while keeping some of the real prior. Since macro-F1 weights all
    11 classes equally, some rebalancing is clearly worth it -- this parameter is
    the dial, and is worth sweeping.
    """
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    class_weight = (1.0 / counts) ** float(power)
    weights = class_weight[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=int(num_samples or len(labels)),
        replacement=True,
        generator=make_generator(seed),
    )


def build_dataloader(
    dataset: MILK10kDataset,
    cfg: Config,
    train: bool,
    sampler: Any | None = None,
) -> DataLoader:
    """Construct a ``DataLoader`` with sane, reproducible defaults."""
    loader_cfg = cfg.get("loader", Config())
    num_workers = int(loader_cfg.get("num_workers", 8))
    batch_size = int(cfg.train.batch_size if train else cfg.train.get("eval_batch_size", cfg.train.batch_size))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(train and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=bool(loader_cfg.get("pin_memory", True)),
        drop_last=bool(train and loader_cfg.get("drop_last", True)),
        collate_fn=collate_lesions,
        # persistent_workers keeps worker processes alive between epochs, which
        # matters here: epochs are short, so respawning would be a real overhead.
        persistent_workers=num_workers > 0 and bool(loader_cfg.get("persistent_workers", True)),
        prefetch_factor=int(loader_cfg.get("prefetch_factor", 4)) if num_workers > 0 else None,
        worker_init_fn=seed_worker,
        generator=make_generator(int(cfg.get("seed", 42))),
    )


def build_datasets(
    cfg: Config,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame | None = None,
    processor: MetadataProcessor | None = None,
) -> tuple[MILK10kDataset, MILK10kDataset | None, MetadataProcessor | None]:
    """Build train/valid datasets, fitting the metadata processor on train only.

    Returns ``(train_ds, valid_ds, processor)``. ``processor`` is ``None`` when the
    configured model does not consume metadata, which keeps Stage 1/2 runs from
    carrying an unused artefact.
    """
    mode = cfg.data.get("mode", "dermoscopic")
    image_root = cfg.data.train_image_root
    layout = cfg.data.get("image_layout", "nested")
    use_meta = bool(cfg.model.get("use_metadata", False))

    if use_meta and processor is None:
        processor = MetadataProcessor(
            use_monet=bool(cfg.data.get("use_monet", True)),
            use_manipulation=bool(cfg.data.get("use_manipulation", True)),
        ).fit(train_frame)

    train_meta = processor.transform(train_frame) if use_meta else None
    train_ds = MILK10kDataset(
        train_frame,
        image_root,
        transforms=build_transforms_from_config(cfg, train=True),
        mode=mode,
        meta_features=train_meta,
        has_targets=True,
        image_layout=layout,
    )

    valid_ds = None
    if valid_frame is not None:
        valid_meta = processor.transform(valid_frame) if use_meta else None
        valid_ds = MILK10kDataset(
            valid_frame,
            image_root,
            transforms=build_transforms_from_config(cfg, train=False),
            mode=mode,
            meta_features=valid_meta,
            has_targets=True,
            image_layout=layout,
        )

    LOGGER.info(
        "Datasets built | mode=%s train=%d valid=%s meta_dim=%s",
        mode,
        len(train_ds),
        len(valid_ds) if valid_ds is not None else "-",
        processor.dim if processor is not None else "-",
    )
    return train_ds, valid_ds, processor
