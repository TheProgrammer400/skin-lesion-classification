#!/usr/bin/env python3
"""Predict skin-lesion classes for dermoscopic images with the trained ensemble.

The script reproduces the inference path used in the study:

    image -> optional dermoscope-border crop -> 224 x 224 resize -> ImageNet normalisation
          -> EfficientNet-B4 / ConvNeXt-Base / DenseNet-121 backbones (computed once)
          -> MC Dropout on the classification heads (30 passes per model, 90 samples)
          -> averaged probabilities -> optional temperature scaling

Because the backbones contain no active dropout, sampling only the heads on features computed once is
mathematically identical to running 30 full forward passes per model, and about 25 times faster.

Examples
--------
    python predict.py --image path/to/lesion.jpg --weights-dir checkpoints
    python predict.py --image path/to/folder --weights-dir checkpoints --output predictions.csv
    python predict.py --image lesion.jpg --weights-dir checkpoints --folds all --no-border-crop

Expected weights layout (one sub-folder per cross-validation fold)::

    checkpoints/fold0/efficientnet_b4.pth
    checkpoints/fold0/convnext_base.pth
    checkpoints/fold0/densenet121.pth
    ...

Research code only. The output is not a medical diagnosis and must not be used for clinical decisions.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# --------------------------------------------------------------------------- settings used in the study
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_NAMES = {
    "akiec": "Actinic keratosis / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesion",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus",
    "vasc": "Vascular lesion",
}
MALIGNANT = ["akiec", "bcc", "mel"]
MODELS = ["efficientnet_b4", "convnext_base", "densenet121"]
IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DROPOUT = 0.4
MC_SAMPLES = 30
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Border detector and crop policy (fixed before the confirmatory border-crop experiment)
DARK_LEVEL = 40          # grey level below which a pixel counts as dark
BORDER_MIN_FRAC = 0.10   # minimum share of the image covered by dark regions touching the edge
CORNER_BOX = 16          # corner patch size (pixels, at 224 x 224)
CORNER_DARK_MIN = 0.60   # a corner is dark if at least this share of its patch is dark
MIN_DARK_CORNERS = 3     # an image is "bordered" if at least this many corners are dark
CROP_PURITY = 0.995      # the crop square must be at least this share field-of-view pixels
CROP_MIN_SIDE = 0.40     # never crop below this fraction of the image side


# --------------------------------------------------------------------------- model definition
def make_head(in_dim: int, n_classes: int = len(CLASSES), p: float = DROPOUT) -> nn.Sequential:
    """Classification head: LayerNorm -> Dropout -> Linear(512) -> GELU -> Dropout -> Linear(7)."""
    return nn.Sequential(
        nn.LayerNorm(in_dim), nn.Dropout(p), nn.Linear(in_dim, 512),
        nn.GELU(), nn.Dropout(p / 2), nn.Linear(512, n_classes),
    )


class SkinLesionModel(nn.Module):
    """timm backbone with global average pooling followed by the classification head."""

    def __init__(self, model_name: str):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0, global_pool="avg")
        self.feat_dim = self.backbone.num_features
        self.classifier = make_head(self.feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))


def find_checkpoint(weights_dir: Path, fold: int, model_name: str) -> Path | None:
    for candidate in (weights_dir / f"fold{fold}" / f"{model_name}.pth",
                      weights_dir / f"fold{fold}" / "focal_ls" / f"{model_name}.pth"):
        if candidate.is_file():
            return candidate
    return None


def available_folds(weights_dir: Path) -> list[int]:
    folds = []
    for d in sorted(weights_dir.glob("fold*")):
        suffix = d.name[4:]
        if d.is_dir() and suffix.isdigit() and all(find_checkpoint(weights_dir, int(suffix), m) for m in MODELS):
            folds.append(int(suffix))
    return folds


def load_model(path: Path, model_name: str, device: torch.device) -> SkinLesionModel:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model = SkinLesionModel(model_name)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


# --------------------------------------------------------------------------- preprocessing
def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def resize(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)


def border_measure(img224: np.ndarray) -> tuple[float, int, np.ndarray]:
    """Share of dark pixels connected to the image edge, number of dark corners, and the edge mask."""
    grey = cv2.GaussianBlur(cv2.cvtColor(np.ascontiguousarray(img224), cv2.COLOR_RGB2GRAY), (3, 3), 0)
    dark = (grey < DARK_LEVEL).astype(np.uint8)
    _, labels = cv2.connectedComponents(dark, connectivity=8)
    touching = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    edge = np.isin(labels, touching[touching != 0])
    b = CORNER_BOX
    corners = [dark[:b, :b], dark[:b, -b:], dark[-b:, :b], dark[-b:, -b:]]
    return float(edge.mean()), int(sum(c.mean() >= CORNER_DARK_MIN for c in corners)), edge


def crop_box(img224: np.ndarray) -> tuple[int, int, int, int, str]:
    """Largest centred square inside the field of view (coordinates at 224 x 224)."""
    _, _, edge = border_measure(img224)
    field = ~edge
    h, w = field.shape
    if field.mean() < 0.2:
        return 0, 0, w, h, "not cropped: field of view too small"
    ys, xs = np.nonzero(field)
    cy, cx = int(round(ys.mean())), int(round(xs.mean()))
    for half in range(min(h, w) // 2, 0, -1):
        y0, y1, x0, x1 = max(cy - half, 0), min(cy + half, h), max(cx - half, 0), min(cx + half, w)
        if field[y0:y1, x0:x1].mean() >= CROP_PURITY:
            if min(y1 - y0, x1 - x0) < CROP_MIN_SIDE * min(h, w):
                return 0, 0, w, h, "not cropped: crop would be too small"
            return x0, y0, x1, y1, "cropped"
    return 0, 0, w, h, "not cropped: no clean square found"


def preprocess(path: Path, border_crop: bool) -> tuple[np.ndarray, str]:
    """Return the 224 x 224 RGB image fed to the models and a short border-handling note."""
    raw = load_rgb(path)
    small = resize(raw)
    if not border_crop:
        return small, "border crop disabled"
    coverage, dark_corners, _ = border_measure(small)
    if not (coverage >= BORDER_MIN_FRAC and dark_corners >= MIN_DARK_CORNERS):
        return small, "no border detected"
    x0, y0, x1, y1, status = crop_box(small)
    if status != "cropped":
        return small, f"border detected, {status}"
    h, w = raw.shape[:2]
    sx, sy = w / IMG_SIZE, h / IMG_SIZE
    patch = raw[int(round(y0 * sy)):int(round(y1 * sy)), int(round(x0 * sx)):int(round(x1 * sx))]
    return resize(patch), "border detected and cropped"


def to_tensor(img224: np.ndarray) -> torch.Tensor:
    x = (img224.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


# --------------------------------------------------------------------------- inference
@torch.no_grad()
def mc_samples(model: SkinLesionModel, x: torch.Tensor, device: torch.device, n: int, seed: int) -> torch.Tensor:
    """Backbone once (float16 autocast on GPU, as in the study), then n stochastic passes of the head."""
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        features = model.backbone(x.to(device))
    features = features.half().float()              # features were stored in float16 in the study
    model.classifier.train()                        # dropout on (LayerNorm behaves the same in both modes)
    torch.manual_seed(seed)
    samples = torch.stack([F.softmax(model.classifier(features), dim=1) for _ in range(n)])
    model.classifier.eval()
    return samples.cpu()                            # (n, batch, classes)


def summarise(samples: torch.Tensor, temperature: float) -> dict[str, np.ndarray]:
    s = samples.numpy().astype(np.float64)          # (S, N, C)
    mean = s.mean(0)
    pred = mean.argmax(1)
    variance = s[:, np.arange(s.shape[1]), pred].var(0)
    entropy = -(mean * np.log(mean + 1e-12)).sum(1)
    expected_entropy = -(s * np.log(s + 1e-12)).sum(2).mean(0)
    probs = mean
    if temperature != 1.0:
        logits = np.log(mean + 1e-12) / temperature
        logits -= logits.max(1, keepdims=True)
        probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    return {"probs": probs, "mc_variance": variance, "entropy": entropy, "mutual_information": entropy - expected_entropy}


def collect_images(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    if target.is_dir():
        return sorted(p for p in target.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    raise FileNotFoundError(f"No image or folder found at {target}")


def parse_folds(value: str, weights_dir: Path) -> list[int]:
    found = available_folds(weights_dir)
    if not found:
        raise FileNotFoundError(
            f"No complete fold of checkpoints found in '{weights_dir}'. Expected files such as "
            f"{weights_dir / 'fold0' / 'efficientnet_b4.pth'} (see 'Pretrained weights' in README.md).")
    if value == "all":
        return found
    folds = [int(v) for v in value.split(",")]
    missing = [k for k in folds if k not in found]
    if missing:
        raise FileNotFoundError(f"Checkpoints for fold(s) {missing} not found in '{weights_dir}'. Available: {found}")
    return folds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--image", required=True, type=Path, help="image file or folder of images")
    ap.add_argument("--weights-dir", default=Path("checkpoints"), type=Path, help="folder containing fold*/ checkpoints")
    ap.add_argument("--folds", default="0",
                    help="fold ensemble(s) to use, e.g. '0', '0,1' or 'all'. Each fold ensemble was evaluated on its own; "
                         "averaging several folds is an extension that was not evaluated in the study")
    ap.add_argument("--mc-samples", default=MC_SAMPLES, type=int, help="MC Dropout passes per model")
    ap.add_argument("--temperature", default=1.0, type=float,
                    help="temperature for calibration (fitted per fold on validation data; 1.0 = uncalibrated)")
    ap.add_argument("--no-border-crop", action="store_true", help="disable the dermoscope-border crop")
    ap.add_argument("--batch-size", default=16, type=int)
    ap.add_argument("--seed", default=42, type=int, help="seed for MC Dropout sampling")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", type=Path, help="optional .csv or .json file for the predictions")
    args = ap.parse_args()

    device = torch.device(args.device)
    try:
        folds = parse_folds(args.folds, args.weights_dir)
        paths = collect_images(args.image)
    except (FileNotFoundError, ValueError) as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    if not paths:
        print(f"No images found in {args.image}", file=sys.stderr)
        return 1

    images, notes = [], []
    for p in paths:
        img, note = preprocess(p, border_crop=not args.no_border_crop)
        images.append(to_tensor(img))
        notes.append(note)

    all_samples = []                                # list of (S, N, C) tensors, one per fold x model
    for fold in folds:
        for mi, name in enumerate(MODELS):
            model = load_model(find_checkpoint(args.weights_dir, fold, name), name, device)
            batches = [mc_samples(model, torch.stack(images[i:i + args.batch_size]), device,
                                  args.mc_samples, args.seed + 1000 * fold + 17 * mi)
                       for i in range(0, len(images), args.batch_size)]
            all_samples.append(torch.cat(batches, dim=1))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    result = summarise(torch.cat(all_samples, dim=0), args.temperature)
    mal_idx = [CLASSES.index(c) for c in MALIGNANT]

    rows = []
    for i, path in enumerate(paths):
        p = result["probs"][i]
        top = np.argsort(-p)
        rows.append({
            "image": str(path),
            "prediction": CLASSES[top[0]],
            "prediction_name": CLASS_NAMES[CLASSES[top[0]]],
            "confidence": round(float(p[top[0]]), 4),
            "p_malignant": round(float(p[mal_idx].sum()), 4),
            "mc_variance": round(float(result["mc_variance"][i]), 5),
            "entropy": round(float(result["entropy"][i]), 4),
            "mutual_information": round(float(result["mutual_information"][i]), 4),
            "border": notes[i],
            **{f"p_{c}": round(float(p[j]), 4) for j, c in enumerate(CLASSES)},
        })
        print(f"\n{path.name}")
        print(f"  prediction : {CLASS_NAMES[CLASSES[top[0]]]} ({CLASSES[top[0]]}), confidence {p[top[0]]:.3f}")
        print("  top 3      : " + ", ".join(f"{CLASSES[j]} {p[j]:.3f}" for j in top[:3]))
        print(f"  P(akiec + bcc + mel) {p[mal_idx].sum():.3f} | MC variance {result['mc_variance'][i]:.4f} | "
              f"entropy {result['entropy'][i]:.3f} | {notes[i]}")

    print(f"\n{len(paths)} image(s); fold ensemble(s) {folds}; {len(all_samples) * args.mc_samples} MC samples per image; "
          f"temperature {args.temperature}. Research use only - not a medical diagnosis.")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix.lower() == ".json":
            args.output.write_text(json.dumps(rows, indent=2))
        else:
            with open(args.output, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
