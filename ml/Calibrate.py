"""
Find per-class logit biases that improve segmentation IoU.

If a model over-predicts foreground (recall >> precision), subtracting a
small bias from each foreground class's logit makes it less eager and
usually raises IoU. (The v4 model needed none; check v5 the same way.) Biases are fitted on a random pixel sample of the val
split, then reported on the FULL val and test splits, before and after.
The test numbers are the honest ones, because test was not used to fit.

Usage:  python ml\\calibrate.py
Output: ml\\output\\calibration_v5.json
"""

import json
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import FloorplanDataset


NUM_CLASSES = 6
CLASS_NAMES = [
    "background", "wall", "door", "window", "stairs_steps", "railing",
]

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT = BASE_DIR / "output" / "best_unet_v5.pth"
OUTPUT_JSON = BASE_DIR / "output" / "calibration_v5.json"

FIT_SPLIT = "val"
REPORT_SPLITS = ("val", "test")

# Fraction of pixels kept per val image for the bias search.
SAMPLE_FRACTION = 0.10

# Search grid for each foreground class's bias (added to its logit).
GRID = np.arange(-4.0, 1.01, 0.25)
ROUNDS = 3
SEED = 42

# Set to an int (e.g. 100) for a quick test, None for all images.
MAX_IMAGES = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"


# ---------------- METRICS ----------------

def confusion_from(logits, labels, bias, chunk=4_000_000):
    """Confusion matrix from cached (N, C) logits with a bias applied."""
    confusion = torch.zeros(
        (NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=logits.device
    )

    for start in range(0, logits.shape[0], chunk):
        scores = logits[start:start + chunk].float() + bias
        predictions = scores.argmax(dim=1)
        targets = labels[start:start + chunk].long()

        confusion += torch.bincount(
            NUM_CLASSES * targets + predictions,
            minlength=NUM_CLASSES * NUM_CLASSES,
        ).reshape(NUM_CLASSES, NUM_CLASSES)

    return confusion


def metrics(confusion):
    confusion = confusion.float().cpu()

    tp = confusion.diag()
    actual = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)

    iou = tp / (actual + predicted - tp).clamp_min(1)
    precision = tp / predicted.clamp_min(1)
    recall = tp / actual.clamp_min(1)

    present = actual[1:] > 0
    mean_iou = (
        iou[1:][present].mean().item() if present.any() else 0.0
    )

    return iou, precision, recall, mean_iou


# ---------------- DATA COLLECTION ----------------

def load_model():
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)

    print(f"Checkpoint : {CHECKPOINT.name}")
    print(f"Epoch      : {checkpoint.get('epoch')}")
    print(f"Saved mIoU : {checkpoint.get('mean_foreground_iou')}")

    model = smp.Unet(
        encoder_name=checkpoint.get("encoder", "resnet34"),
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def make_loader(split):
    dataset = FloorplanDataset(split, training=False)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


@torch.no_grad()
def collect_samples(model, split):
    """Run the model once and keep a random subset of pixel logits."""
    loader = make_loader(split)
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(SEED)

    logit_chunks = []
    label_chunks = []

    for index, (image, mask) in enumerate(
        tqdm(loader, desc=f"{split} (collect)", unit="img")
    ):
        if MAX_IMAGES is not None and index >= MAX_IMAGES:
            break

        image = image.to(DEVICE)

        with torch.autocast(
            device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP
        ):
            logits = model(image)

        flat = logits[0].float().reshape(NUM_CLASSES, -1)
        labels = mask.reshape(-1).to(DEVICE)

        keep = int(labels.numel() * SAMPLE_FRACTION)
        chosen = torch.randperm(
            labels.numel(), device=DEVICE, generator=generator
        )[:keep]

        logit_chunks.append(flat[:, chosen].T.half().contiguous())
        label_chunks.append(labels[chosen].to(torch.uint8))

    return torch.cat(logit_chunks), torch.cat(label_chunks)


# ---------------- BIAS SEARCH ----------------

def search_biases(logits, labels):
    """Coordinate descent over the foreground class biases."""
    bias = torch.zeros(NUM_CLASSES, device=logits.device)

    _, _, _, best = metrics(confusion_from(logits, labels, bias))
    print(f"\nStarting sampled mean IoU (no bias): {best:.4f}")

    for round_number in range(1, ROUNDS + 1):
        improved = False

        for class_index in range(1, NUM_CLASSES):
            best_value = bias[class_index].item()

            for value in GRID:
                trial = bias.clone()
                trial[class_index] = float(value)

                _, _, _, score = metrics(
                    confusion_from(logits, labels, trial)
                )

                if score > best + 1e-6:
                    best = score
                    best_value = float(value)
                    improved = True

            bias[class_index] = best_value

        print(
            f"Round {round_number}: sampled mean IoU {best:.4f}  "
            f"bias={[round(b, 2) for b in bias.tolist()]}"
        )

        if not improved:
            break

    return bias, best


# ---------------- FULL EVALUATION ----------------

@torch.no_grad()
def evaluate_full(model, split, bias):
    """Score every image, with and without the bias, in one pass."""
    loader = make_loader(split)

    raw = torch.zeros(
        (NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=DEVICE
    )
    calibrated = torch.zeros_like(raw)

    shaped_bias = bias.view(1, NUM_CLASSES, 1, 1)

    for index, (image, mask) in enumerate(
        tqdm(loader, desc=f"{split} (full)", unit="img")
    ):
        if MAX_IMAGES is not None and index >= MAX_IMAGES:
            break

        image = image.to(DEVICE)
        target = mask.to(DEVICE, dtype=torch.long).reshape(-1)

        with torch.autocast(
            device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP
        ):
            logits = model(image).float()

        for confusion, scores in (
            (raw, logits),
            (calibrated, logits + shaped_bias),
        ):
            prediction = scores.argmax(dim=1).reshape(-1)
            confusion += torch.bincount(
                NUM_CLASSES * target + prediction,
                minlength=NUM_CLASSES * NUM_CLASSES,
            ).reshape(NUM_CLASSES, NUM_CLASSES)

    return raw, calibrated


def print_comparison(split, raw, calibrated):
    iou_a, prec_a, rec_a, miou_a = metrics(raw)
    iou_b, prec_b, rec_b, miou_b = metrics(calibrated)

    print(f"\n=== {split.upper()} (full split) ===")
    print(
        f"{'class':14s} {'IoU before':>10s} {'IoU after':>10s} "
        f"{'Prec b->a':>13s} {'Recall b->a':>13s}"
    )

    for i, name in enumerate(CLASS_NAMES):
        print(
            f"{name:14s} {iou_a[i]:10.4f} {iou_b[i]:10.4f} "
            f"{prec_a[i]:6.3f}->{prec_b[i]:5.3f} "
            f"{rec_a[i]:6.3f}->{rec_b[i]:5.3f}"
        )

    print(
        f"Mean foreground IoU: {miou_a:.4f} -> {miou_b:.4f} "
        f"({miou_b - miou_a:+.4f})"
    )

    return miou_a, miou_b


# ---------------- MAIN ----------------

def main():
    model = load_model()

    logits, labels = collect_samples(model, FIT_SPLIT)
    print(f"Sampled pixels for fitting: {labels.numel():,}")

    bias, fitted_score = search_biases(logits, labels)

    del logits, labels
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    results = {}

    for split in REPORT_SPLITS:
        try:
            raw, calibrated = evaluate_full(model, split, bias)
        except FileNotFoundError as error:
            print(f"\nSkipping {split}: {error}")
            continue

        before, after = print_comparison(split, raw, calibrated)
        results[split] = {"before": before, "after": after}

    payload = {
        "checkpoint": CHECKPOINT.name,
        "class_names": CLASS_NAMES,
        "logit_bias": [round(b, 4) for b in bias.tolist()],
        "fit_split": FIT_SPLIT,
        "sampled_fit_mean_iou": fitted_score,
        "results": results,
    }

    OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nLogit bias ({', '.join(CLASS_NAMES)}):")
    print(f"  {payload['logit_bias']}")
    print(f"Saved: {OUTPUT_JSON}")
    print(
        "\nNote: val numbers are optimistic (biases were fitted on val). "
        "The test numbers are the unbiased check."
    )
    print(
        "To use at inference: logits = model(x); "
        "logits = logits + bias.view(1, 6, 1, 1); pred = logits.argmax(1)"
    )


if __name__ == "__main__":
    main()