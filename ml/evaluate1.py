from pathlib import Path

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

# Splits to score. Add "train" to compare train vs val (over/underfitting).
# Use "test" only for the final number.
SPLITS = ("val", "test")

# Evaluate at most this many images per split (None = all).
MAX_IMAGES = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"


@torch.no_grad()
def evaluate(model, split):
    # training=False: full 1024x1024 image, no crops or augmentation.
    dataset = FloorplanDataset(split, training=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    confusion = torch.zeros(
        (NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=DEVICE
    )

    for index, (image, mask) in enumerate(
        tqdm(loader, desc=split, unit="img")
    ):
        if MAX_IMAGES is not None and index >= MAX_IMAGES:
            break

        image = image.to(DEVICE)
        mask = mask.to(DEVICE, dtype=torch.long)

        with torch.autocast(
            device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP
        ):
            logits = model(image)

        prediction = logits.argmax(dim=1).reshape(-1)
        target = mask.reshape(-1)

        encoded = NUM_CLASSES * target + prediction
        confusion += torch.bincount(
            encoded, minlength=NUM_CLASSES * NUM_CLASSES
        ).reshape(NUM_CLASSES, NUM_CLASSES)

    confusion = confusion.float().cpu()
    tp = confusion.diag()
    actual = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)

    iou = tp / (actual + predicted - tp).clamp_min(1)
    dice = 2 * tp / (actual + predicted).clamp_min(1)
    precision = tp / predicted.clamp_min(1)
    recall = tp / actual.clamp_min(1)

    print(f"\n=== {split.upper()} ({len(dataset)} images in split) ===")
    print(
        f"{'class':14s} {'IoU':>7s} {'Dice':>7s} "
        f"{'Prec':>7s} {'Recall':>7s} {'GT pixels':>14s}"
    )
    for i, name in enumerate(CLASS_NAMES):
        print(
            f"{name:14s} {iou[i]:7.4f} {dice[i]:7.4f} "
            f"{precision[i]:7.4f} {recall[i]:7.4f} {int(actual[i]):14,d}"
        )
    print(f"Mean foreground IoU : {iou[1:].mean():.4f}")
    print(f"Mean foreground Dice: {dice[1:].mean():.4f}")


def main():
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

    for split in SPLITS:
        evaluate(model, split)


if __name__ == "__main__":
    main()