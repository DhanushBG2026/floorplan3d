"""
Run the trained floor-plan segmentation model on your own images.

Examples (from the folder that contains ml\\):
    python ml\\predict.py plan.png
    python ml\\predict.py plan1.jpg plan2.png --tta
    python ml\\predict.py my_plans_folder --out results --no-stairs
    python ml\\predict.py plan.png --merge-railing
    python ml\\predict.py plan.png --no-clean --save-raw

For every input it writes, next to each other in the output folder:
    <name>_mask.png     class IDs (0 bg, 1 wall, 2 door, 3 window, 4 stairs,
                        5 railing),
                        saved with a colour palette, at the ORIGINAL size
    <name>_overlay.png  the plan with the coloured mask blended on top
    <name>_raw_mask.png only with --save-raw: the mask before cleanup

Cleanup (postprocess.py, needs scipy) removes stray doors/windows/railings,
smooths the edges and turns straight walls, doors, windows and railings
into exact rectangles. Use --no-clean to switch it off.
"""

import argparse
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from PIL import Image

from dataset import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, resize_and_pad

try:
    import postprocess
except ImportError as error:  # scipy missing, or postprocess.py not here
    postprocess = None
    POSTPROCESS_ERROR = error


NUM_CLASSES = 6
CLASS_NAMES = [
    "background", "wall", "door", "window", "stairs_steps", "railing",
]

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = BASE_DIR / "output" / "best_unet_v5.pth"
DEFAULT_OUTPUT = BASE_DIR / "predictions"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# Same colours as prepare_dataset.py.
PALETTE = [
    0, 0, 0,          # 0: background
    220, 50, 50,      # 1: wall
    40, 180, 90,      # 2: door
    50, 120, 230,     # 3: window
    255, 165, 0,      # 4: stairs / steps
    180, 70, 220,     # 5: railing
]

# Cleanup runs at the plan's own resolution, but never above this many
# pixels on the long side (bigger plans are cleaned at this size and the
# result is scaled back up, which keeps the straight edges sharp).
MAX_WORK_SIDE = 2048

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"


# ---------------- MODEL ----------------

def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    model = smp.Unet(
        encoder_name=checkpoint.get("encoder", "resnet34"),
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Checkpoint : {Path(checkpoint_path).name}")
    print(f"Epoch      : {checkpoint.get('epoch')}")
    print(f"Device     : {DEVICE}")

    return model


@torch.no_grad()
def predict_probabilities(model, tensor, tta):
    """Softmax probabilities, optionally averaged over flips."""

    def run(batch):
        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=USE_AMP,
        ):
            logits = model(batch)
        return torch.softmax(logits.float(), dim=1)

    probabilities = run(tensor)

    if tta:
        probabilities = probabilities + run(tensor.flip(3)).flip(3)
        probabilities = probabilities + run(tensor.flip(2)).flip(2)
        probabilities = probabilities / 3.0

    return probabilities


# ---------------- PRE / POST PROCESSING ----------------

def preprocess(image):
    """Resize + pad exactly like training, then normalise."""
    padded = resize_and_pad(image, IMAGE_SIZE, is_mask=False)

    array = np.asarray(padded, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD

    tensor = torch.from_numpy(
        array.transpose(2, 0, 1).astype(np.float32).copy()
    )

    return tensor.unsqueeze(0).to(DEVICE)


def padding_box(original_size):
    """Where the resized plan sits inside the padded canvas."""
    original_width, original_height = original_size
    target_width, target_height = IMAGE_SIZE

    scale = min(
        target_width / original_width,
        target_height / original_height,
    )

    new_width = max(1, round(original_width * scale))
    new_height = max(1, round(original_height * scale))

    left = (target_width - new_width) // 2
    top = (target_height - new_height) // 2

    return left, top, new_width, new_height


def probabilities_to_labels(probabilities, original_size):
    """
    Crop the padding off the model's probability maps, resize them
    (smoothly, not nearest-neighbour) to the working size and take the
    most likely class per pixel. Smooth resizing of the probabilities gives
    much cleaner edges than resizing the finished label image.

    Returns (labels uint8 (H, W), probs float32 (C, H, W), work_size).
    """
    original_width, original_height = original_size
    left, top, new_width, new_height = padding_box(original_size)

    probabilities = probabilities[:, :, top:top + new_height,
                                  left:left + new_width]

    long_side = max(original_width, original_height)
    shrink = min(1.0, MAX_WORK_SIDE / long_side)
    work_width = max(1, int(round(original_width * shrink)))
    work_height = max(1, int(round(original_height * shrink)))

    probabilities = F.interpolate(
        probabilities, size=(work_height, work_width),
        mode="bilinear", align_corners=False,
    )

    labels = probabilities.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    probs = probabilities[0].cpu().numpy().astype(np.float32)

    return labels, probs, (work_width, work_height)


def finish_labels(labels, hide_stairs, merge_railing):
    """Optional class removal / merging, applied after cleanup."""
    labels = labels.copy()

    if hide_stairs:
        labels[labels == 4] = 0

    if merge_railing:
        labels[labels == 5] = 1

    return labels


def to_original_size(labels, original_size):
    if (labels.shape[1], labels.shape[0]) == tuple(original_size):
        return labels
    return np.asarray(
        Image.fromarray(labels).resize(
            original_size, Image.Resampling.NEAREST
        )
    )


def save_mask(prediction, path):
    mask = Image.fromarray(prediction.astype(np.uint8), mode="P")
    mask.putpalette(PALETTE)
    mask.save(path)


def save_overlay(image, prediction, path):
    output = np.asarray(image.convert("RGB")).astype(np.float32).copy()
    colours = np.array(PALETTE, dtype=np.float32).reshape(-1, 3)[prediction]
    foreground = prediction > 0

    output[foreground] = 0.5 * output[foreground] + 0.5 * colours[foreground]

    Image.fromarray(output.astype(np.uint8)).save(path)


# ---------------- MAIN ----------------

def collect_inputs(paths):
    files = []

    for item in paths:
        item = Path(item)

        if item.is_dir():
            files.extend(
                sorted(
                    p for p in item.iterdir()
                    if p.suffix.lower() in IMAGE_SUFFIXES
                )
            )
        elif item.is_file():
            files.append(item)
        else:
            print(f"Skipping (not found): {item}")

    return files


def process_image(model, path, output_dir, tta, hide_stairs, merge_railing,
                  clean=True, save_raw=False):
    image = Image.open(path).convert("RGB")
    original_size = image.size  # (width, height)

    tensor = preprocess(image)
    probabilities = predict_probabilities(model, tensor, tta)

    labels, probs, _ = probabilities_to_labels(probabilities, original_size)
    stem = Path(path).stem

    if save_raw:
        raw = finish_labels(labels, hide_stairs, merge_railing)
        save_mask(
            to_original_size(raw, original_size),
            output_dir / f"{stem}_raw_mask.png",
        )

    if clean:
        report = {}
        labels = postprocess.clean_mask(labels, probs=probs, report=report)
        removed = {
            k: v for k, v in report.items()
            if k not in ("rects", "run_lengths") and v
        }
        if removed:
            print(f"  cleanup removed (pixels): {removed}")

    labels = finish_labels(labels, hide_stairs, merge_railing)
    prediction = to_original_size(labels, original_size)

    save_mask(prediction, output_dir / f"{stem}_mask.png")
    save_overlay(image, prediction, output_dir / f"{stem}_overlay.png")

    total = prediction.size
    shares = ", ".join(
        f"{CLASS_NAMES[c]} {100 * (prediction == c).sum() / total:.2f}%"
        for c in range(1, NUM_CLASSES)
    )
    print(f"{Path(path).name}: {shares}")


def main():
    parser = argparse.ArgumentParser(
        description="Segment walls, doors, windows and stairs in floor plans."
    )
    parser.add_argument(
        "inputs", nargs="+", help="Image files and/or folders of images."
    )
    parser.add_argument(
        "--checkpoint", default=str(DEFAULT_CHECKPOINT),
        help="Model checkpoint (default: output/best_unet_v5.pth).",
    )
    parser.add_argument(
        "--out", default=str(DEFAULT_OUTPUT),
        help="Output folder (default: ml/predictions).",
    )
    parser.add_argument(
        "--tta", action="store_true",
        help="Average with horizontal/vertical flips (slower, a bit better).",
    )
    parser.add_argument(
        "--no-stairs", action="store_true",
        help="Remove the stairs class from the output.",
    )
    parser.add_argument(
        "--merge-railing", action="store_true",
        help="Output railings as walls (class 1) instead of their own class.",
    )
    parser.add_argument(
        "--no-clean", action="store_true",
        help="Skip the cleanup / straightening step (raw model output).",
    )
    parser.add_argument(
        "--save-raw", action="store_true",
        help="Also save the mask before cleanup as <name>_raw_mask.png.",
    )
    args = parser.parse_args()

    files = collect_inputs(args.inputs)

    if not files:
        raise SystemExit("No input images found.")

    clean = not args.no_clean
    if clean and postprocess is None:
        raise SystemExit(
            "Cleanup needs postprocess.py next to predict.py and scipy "
            f"installed (pip install scipy). Details: {POSTPROCESS_ERROR}\n"
            "Or run with --no-clean."
        )

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint)

    for path in files:
        try:
            process_image(
                model, path, output_dir, args.tta, args.no_stairs,
                args.merge_railing, clean=clean, save_raw=args.save_raw,
            )
        except Exception as error:
            print(f"FAILED {Path(path).name}: {error}")

    print(f"\nResults saved in: {output_dir}")


if __name__ == "__main__":
    main()