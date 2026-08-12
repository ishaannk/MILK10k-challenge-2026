"""Challenge-level constants for ISIC MILK10k.

Everything here is dictated by the official challenge definition, so it lives in
one place and is imported rather than re-typed. In particular ``CLASSES`` is the
exact column order the submission CSV must use.
"""

from __future__ import annotations

# The 11 diagnostic categories, in official submission column order.
CLASSES: list[str] = [
    "AKIEC",    # Actinic keratosis / intraepidermal carcinoma
    "BCC",      # Basal cell carcinoma
    "BEN_OTH",  # Other benign proliferations
    "BKL",      # Benign keratinocytic lesion
    "DF",       # Dermatofibroma
    "INF",      # Inflammatory and infectious conditions
    "MAL_OTH",  # Other malignant proliferations
    "MEL",      # Melanoma
    "NV",       # Melanocytic nevus
    "SCCKA",    # Squamous cell carcinoma / keratoacanthoma
    "VASC",     # Vascular lesions and haemorrhage
]
NUM_CLASSES: int = len(CLASSES)
CLASS_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(CLASSES)}

# Human-readable names, used in reports and confusion-matrix axes.
CLASS_DESCRIPTIONS: dict[str, str] = {
    "AKIEC": "Actinic keratosis / intraepidermal carcinoma",
    "BCC": "Basal cell carcinoma",
    "BEN_OTH": "Other benign proliferation",
    "BKL": "Benign keratinocytic lesion",
    "DF": "Dermatofibroma",
    "INF": "Inflammatory / infectious",
    "MAL_OTH": "Other malignant proliferation",
    "MEL": "Melanoma",
    "NV": "Melanocytic nevus",
    "SCCKA": "Squamous cell carcinoma / keratoacanthoma",
    "VASC": "Vascular / haemorrhage",
}

# Classes the challenge treats as malignant. Not scored directly, but useful for
# reporting a clinically meaningful secondary metric.
MALIGNANT_CLASSES: list[str] = ["AKIEC", "BCC", "MAL_OTH", "MEL", "SCCKA"]

# `image_type` values in the metadata CSV, mapped to the short names used
# throughout the codebase.
IMAGE_TYPE_CLINICAL = "clinical: close-up"
IMAGE_TYPE_DERMOSCOPIC = "dermoscopic"
MODALITIES: list[str] = ["clinical", "dermoscopic"]

# The submission threshold is fixed by the organisers: probabilities are
# binarised at >= 0.5 before the macro-F1 is computed. Any per-class threshold
# tuning we do must therefore be folded back into the probabilities themselves.
SUBMISSION_THRESHOLD: float = 0.5

# Categorical metadata vocabularies. Held fixed (rather than inferred per split)
# so that a model trained on one fold can score any other file without column
# drift. `unknown` absorbs NaNs and unseen values.
SITE_VOCAB: list[str] = [
    "trunk",
    "head_neck_face",
    "lower_extremity",
    "upper_extremity",
    "hand",
    "foot",
    "genital",
    "unknown",
]
SEX_VOCAB: list[str] = ["male", "female", "unknown"]
SKIN_TONE_VOCAB: list[str] = ["0", "1", "2", "3", "4", "5", "unknown"]

# MONET concept-annotation columns. Present in both train and test metadata, so
# they are legitimate model inputs.
MONET_COLUMNS: list[str] = [
    "MONET_ulceration_crust",
    "MONET_hair",
    "MONET_vasculature_vessels",
    "MONET_erythema",
    "MONET_pigmented",
    "MONET_gel_water_drop_fluid_dermoscopy_liquid",
    "MONET_skin_markings_pen_ink_purple_pen",
]

# ImageNet statistics; all timm backbones we use are ImageNet-pretrained.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
