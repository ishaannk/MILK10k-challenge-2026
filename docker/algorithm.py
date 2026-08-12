#!/usr/bin/env python
"""ISIC challenge container entrypoint.

Contract (see https://github.com/ImageMarkup/isic-algorithm-example):

* score every JPEG reachable under ``/images``
* write CSV with a header row to **stdout**

MILK10k predicts one row per **lesion** from a *pair* of images (clinical close-up
+ dermoscopic), while the container is handed a directory of JPEGs. Both plausible
layouts are therefore handled:

``/images/<lesion_id>/<isic_id>.jpg``  (the released MILK10k layout)
    Images are grouped by their parent directory into lesions. Which of the pair is
    dermoscopic is not encoded in the filename, so it is decided by a small image
    heuristic (see :func:`pick_dermoscopic`) and both orderings are averaged for the
    dual-encoder model, which makes the result invariant to guessing wrong.

``/images/<isic_id>.jpg``  (flat)
    Each file is treated as its own lesion and used as the dermoscopic view. Only
    the dermoscopy-only models contribute; the dual-encoder model is fed the same
    image in both towers, which is a degraded but sane fallback.

All progress/diagnostic output goes to **stderr** so that stdout stays a clean CSV.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.constants import CLASSES, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from src.inference.predictor import ensemble_probabilities  # noqa: E402
from src.inference.tta import TTA_POLICIES, apply_view  # noqa: E402
from src.models.lesion_net import LesionNet  # noqa: E402
from src.validation.thresholds import apply_strategy  # noqa: E402

cv2.setNumThreads(1)

IMAGES_DIR = Path("/images")
BUNDLE_DIR = Path(__file__).resolve().parent / "weights_bundle"
BATCH_SIZE = int(8)


def log(message: str) -> None:
    """Diagnostics to stderr; stdout is reserved for the CSV."""
    print(message, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Image handling
# ---------------------------------------------------------------------------
def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unreadable image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def preprocess(image: np.ndarray, size: int) -> np.ndarray:
    """Resize to ``size`` and normalise exactly as the eval transform does."""
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    array = resized.astype(np.float32) / 255.0
    array = (array - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    return array.transpose(2, 0, 1)


def pick_dermoscopic(paths: list[Path]) -> tuple[Path, Path]:
    """Guess which of a lesion's two images is the dermoscopic one.

    Returns ``(dermoscopic, clinical)``. Dermoscopy is captured through a contact
    lens, so relative to a clinical close-up it is typically lower in saturation
    variance across the frame and more uniformly illuminated — the lesion fills the
    field rather than sitting in surrounding skin. The heuristic scores each image by
    the standard deviation of its per-tile mean brightness: clinical photographs vary
    more across the frame.

    This only breaks the tie for ordering; ``predict_lesions`` averages both
    orderings for the dual-encoder model, so a wrong guess costs nothing.
    """
    if len(paths) == 1:
        return paths[0], paths[0]

    scores = []
    for path in paths[:2]:
        image = cv2.cvtColor(load_rgb(path), cv2.COLOR_RGB2GRAY)
        small = cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32)
        scores.append(float(small.std()))
    # Lower spatial brightness variation -> more likely dermoscopic.
    return (paths[0], paths[1]) if scores[0] <= scores[1] else (paths[1], paths[0])


def discover_lesions(root: Path) -> tuple[list[str], dict[str, tuple[Path, Path]], bool]:
    """Group the JPEGs under ``root`` into lesions.

    Returns ``(lesion_ids, {lesion_id: (derm_path, clin_path)}, nested)``.
    """
    files = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}
    )
    if not files:
        raise SystemExit(f"No JPEG images found under {root}")

    by_parent: dict[Path, list[Path]] = {}
    for path in files:
        by_parent.setdefault(path.parent, []).append(path)

    # Nested layout iff images sit in per-lesion subdirectories (not root) and those
    # directories hold small groups (a pair), rather than one big flat dump.
    nested = all(parent != root for parent in by_parent) and max(
        len(v) for v in by_parent.values()
    ) <= 3

    lesions: dict[str, tuple[Path, Path]] = {}
    if nested:
        for parent, paths in by_parent.items():
            lesions[parent.name] = pick_dermoscopic(sorted(paths))
    else:
        for path in files:
            lesions[path.stem] = (path, path)

    lesion_ids = sorted(lesions)
    log(f"Found {len(files)} image(s) -> {len(lesion_ids)} lesion(s) (layout: {'nested' if nested else 'flat'})")
    return lesion_ids, lesions, nested


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def build_from_manifest(run: dict, weight_file: Path, device: torch.device) -> LesionNet:
    spec = run["model"]
    model = LesionNet(
        backbone=spec["backbone"],
        clinical_backbone=spec.get("clinical_backbone"),
        pretrained=False,           # weights come from the bundle, never the network
        use_clinical=spec["use_clinical"],
        use_metadata=spec["use_metadata"],
        meta_dim=0,
        fusion=spec["fusion"],
        fusion_dim=spec["fusion_dim"],
        head_hidden=spec["head_hidden"],
        head_dropout=spec["head_dropout"],
        drop_rate=0.0,
        drop_path_rate=0.0,
        share_encoder=spec["share_encoder"],
    )
    state = torch.load(weight_file, map_location="cpu", weights_only=True)
    # Bundle is fp16 to keep the image small; run inference in fp32 on CPU.
    state = {k: (v.float() if v.is_floating_point() else v) for k, v in state.items()}
    model.load_state_dict(state)
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_run(
    run: dict,
    lesion_ids: list[str],
    lesions: dict[str, tuple[Path, Path]],
    device: torch.device,
    tta_policy: str,
) -> np.ndarray:
    """Average all folds of one run over all TTA views. Returns ``(n_lesions, 11)``."""
    size = run["image_size"]
    use_clinical = run["model"]["use_clinical"]
    views = TTA_POLICIES.get(tta_policy, TTA_POLICIES["d4"])

    models = [build_from_manifest(run, BUNDLE_DIR / name, device) for name in run["folds"]]
    log(f"  {run['name']}: {len(models)} fold(s), {size}px, clinical={use_clinical}, {len(views)} TTA view(s)")

    logits_sum = np.zeros((len(lesion_ids), len(CLASSES)), dtype=np.float64)

    for start in range(0, len(lesion_ids), BATCH_SIZE):
        chunk = lesion_ids[start : start + BATCH_SIZE]
        derm = np.stack([preprocess(load_rgb(lesions[i][0]), size) for i in chunk])
        derm_t = torch.from_numpy(derm).to(device)
        clin_t = None
        if use_clinical:
            clin = np.stack([preprocess(load_rgb(lesions[i][1]), size) for i in chunk])
            clin_t = torch.from_numpy(clin).to(device)

        accumulated = torch.zeros(len(chunk), len(CLASSES), dtype=torch.float32, device=device)
        n_passes = 0
        for model in models:
            for hflip, vflip, rot in views:
                d = apply_view(derm_t, hflip, vflip, rot)
                if use_clinical:
                    c = apply_view(clin_t, hflip, vflip, rot)
                    # Average both pair orderings so a wrong dermoscopic/clinical
                    # guess in pick_dermoscopic cannot change the answer.
                    accumulated += model(d, c).float() + model(c, d).float()
                    n_passes += 2
                else:
                    accumulated += model(d).float()
                    n_passes += 1
        logits_sum[start : start + len(chunk)] = (accumulated / n_passes).cpu().numpy()

        if start % (BATCH_SIZE * 10) == 0:
            log(f"    {min(start + BATCH_SIZE, len(lesion_ids))}/{len(lesion_ids)}")

    # Averaging in logit space preserves confident tails, which per-class
    # thresholding depends on; sigmoid is applied once at the end.
    return 1.0 / (1.0 + np.exp(-logits_sum))


def main() -> None:
    manifest = json.loads((BUNDLE_DIR / "manifest.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    lesion_ids, lesions, _ = discover_lesions(IMAGES_DIR)

    per_run = [
        predict_run(run, lesion_ids, lesions, device, run.get("tta", "d4"))
        for run in manifest["runs"]
    ]

    blend = manifest.get("blend") or {}
    method = blend.get("method", "mean")

    if len(per_run) == 1:
        probs = per_run[0]
    elif blend.get("per_class_weights"):
        # Per-class weights: each class gets its own mixture of the runs.
        weights = np.asarray(blend["per_class_weights"], dtype=np.float64)
        probs = np.zeros_like(per_run[0])
        for c in range(len(CLASSES)):
            columns = [p[:, [c]] for p in per_run]
            probs[:, [c]] = ensemble_probabilities(columns, weights=weights[:, c], method=method)
    else:
        probs = ensemble_probabilities(per_run, weights=blend.get("weights"), method=method)

    thresholds = blend.get("thresholds")
    strategy = blend.get("strategy", "raw")
    if thresholds is not None and strategy != "raw":
        # Re-map probabilities so the challenge's fixed 0.5 cut reproduces the
        # OOF-tuned per-class decision boundaries.
        probs = apply_strategy(probs, strategy=strategy, thresholds=np.asarray(thresholds))
    log(f"Blend: method={method} strategy={strategy} runs={len(per_run)}")

    # --- CSV to stdout
    writer = sys.stdout
    writer.write("image," + ",".join(CLASSES) + "\n")
    for i, lesion_id in enumerate(lesion_ids):
        writer.write(lesion_id + "," + ",".join(f"{v:.6f}" for v in probs[i]) + "\n")
    writer.flush()
    log(f"Wrote {len(lesion_ids)} row(s) to stdout")


if __name__ == "__main__":
    main()
