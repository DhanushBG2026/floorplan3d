from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import FloorplanDataset


# ---------------- CONFIGURATION ----------------

NUM_CLASSES = 6          # background, wall, door, window, stairs, railing
OLD_NUM_CLASSES = 5      # classes in the v4 checkpoint we fine-tune from

# Memory plan for a 4 GB GPU:
#   random 512x512 crops, batch 2, gradient accumulation 2 (effective 4).
# If you hit "CUDA out of memory": BATCH_SIZE = 1 and ACCUM_STEPS = 4,
# or ENCODER = "resnet18".
BATCH_SIZE = 2
ACCUM_STEPS = 2

EPOCHS = 20
LEARNING_RATE = 1e-4
PATIENCE = 6
NUM_WORKERS = 2
SEED = 42

ENCODER = "resnet34"

CLASS_NAMES = [
    "background",
    "wall",
    "door",
    "window",
    "stairs_steps",
    "railing",
]

# Order matches CLASS_NAMES. Railing is thin and rare, so it gets a
# slightly higher weight than wall.
CLASS_WEIGHTS = [0.4, 1.0, 2.0, 1.5, 2.0, 2.5]

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# v5 = 6-class model. Earlier checkpoints are never overwritten.
BASE_CHECKPOINT = OUTPUT_DIR / "best_unet_v4.pth"
BEST_PATH = OUTPUT_DIR / "best_unet_v5.pth"
LAST_PATH = OUTPUT_DIR / "last_unet_v5.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"


def build_model(pretrained=True):
    return smp.Unet(
        encoder_name=ENCODER,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=NUM_CLASSES,
    )


def load_v4_weights(model, checkpoint_path):
    """
    Initialise a 6-class model from the 5-class v4 checkpoint.

    Every layer whose shape matches is copied as-is. The output layer has
    one more channel than v4, so the five old class channels are copied and
    the new railing channel starts from the wall channel (bias lowered by 1
    so early predictions stay "wall" until railings are actually learned).
    """
    state = torch.load(checkpoint_path, map_location=DEVICE)
    old_weights = state["model_state_dict"]
    new_weights = model.state_dict()

    copied = 0
    expanded = []

    for name, tensor in old_weights.items():
        if name not in new_weights:
            continue

        target = new_weights[name]

        if target.shape == tensor.shape:
            new_weights[name] = tensor
            copied += 1
        elif (
            tensor.shape[0] == OLD_NUM_CLASSES
            and target.shape[0] == NUM_CLASSES
            and tensor.shape[1:] == target.shape[1:]
        ):
            expanded_tensor = target.clone()
            expanded_tensor[:OLD_NUM_CLASSES] = tensor
            expanded_tensor[OLD_NUM_CLASSES] = tensor[1]  # start from wall

            if name.endswith("bias"):
                expanded_tensor[OLD_NUM_CLASSES] -= 1.0

            new_weights[name] = expanded_tensor
            expanded.append(name)

    if not expanded:
        raise RuntimeError(
            "Could not find the output layer to expand from "
            f"{OLD_NUM_CLASSES} to {NUM_CLASSES} classes."
        )

    model.load_state_dict(new_weights)

    print(
        f"Loaded {copied} layers from {checkpoint_path.name} "
        f"(epoch {state.get('epoch')}); expanded output layer: {expanded}"
    )


# ---------------- LOSS FUNCTION ----------------

def combined_loss(logits, targets, ce_criterion):
    """
    Weighted Cross-Entropy + foreground Dice.
    Targets contain integer class IDs from 0 to 5.
    """
    ce_loss = ce_criterion(logits, targets)

    probabilities = torch.softmax(logits.float(), dim=1)

    one_hot = F.one_hot(
        targets, num_classes=NUM_CLASSES
    ).permute(0, 3, 1, 2).float()

    # Exclude background from Dice loss.
    probabilities = probabilities[:, 1:]
    one_hot = one_hot[:, 1:]

    dimensions = (0, 2, 3)

    intersection = (probabilities * one_hot).sum(dim=dimensions)

    denominator = (
        probabilities.sum(dim=dimensions)
        + one_hot.sum(dim=dimensions)
    )

    # Smoothing of 1.0 stops classes that are absent from a batch from
    # producing a loss of ~1 even when the model correctly predicts none.
    dice_scores = (2.0 * intersection + 1.0) / (denominator + 1.0)

    dice_loss = 1.0 - dice_scores.mean()

    return 0.5 * ce_loss + 0.5 * dice_loss


# ---------------- METRICS ----------------

@torch.no_grad()
def update_confusion_matrix(confusion, predictions, targets):
    predictions = predictions.reshape(-1)
    targets = targets.reshape(-1)

    valid = (targets >= 0) & (targets < NUM_CLASSES)

    encoded = NUM_CLASSES * targets[valid] + predictions[valid]

    confusion += torch.bincount(
        encoded,
        minlength=NUM_CLASSES * NUM_CLASSES,
    ).reshape(NUM_CLASSES, NUM_CLASSES)

    return confusion


def calculate_metrics(confusion):
    confusion = confusion.float()

    true_positive = confusion.diag()
    actual = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)

    iou = true_positive / (
        actual + predicted - true_positive
    ).clamp_min(1)

    dice = (2 * true_positive) / (actual + predicted).clamp_min(1)

    # Mean over foreground classes that actually appear in the data.
    present = actual[1:] > 0

    if present.any():
        mean_iou = iou[1:][present].mean().item()
        mean_dice = dice[1:][present].mean().item()
    else:
        mean_iou = 0.0
        mean_dice = 0.0

    return (
        iou.cpu().tolist(),
        dice.cpu().tolist(),
        mean_iou,
        mean_dice,
    )


# ---------------- TRAINING ----------------

def main():
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True

    print("=" * 60)
    print("FLOORPLAN SEGMENTATION TRAINING")
    print("=" * 60)
    print(f"Device         : {DEVICE}")

    if DEVICE.type == "cuda":
        print(f"GPU            : {torch.cuda.get_device_name(0)}")
        print(
            f"GPU memory     : "
            f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB"
        )

    print(f"Encoder        : {ENCODER} (fine-tuning from v4)")
    print(f"Classes        : {NUM_CLASSES} {CLASS_NAMES}")
    print(
        f"Batch size     : {BATCH_SIZE} "
        f"(x{ACCUM_STEPS} accumulation = {BATCH_SIZE * ACCUM_STEPS})"
    )
    print(f"Epochs         : {EPOCHS}")
    print(f"Mixed precision: {USE_AMP}")
    print("=" * 60)

    # ---- Datasets ----

    print("\nLoading datasets...")

    train_dataset = FloorplanDataset("train")
    val_dataset = FloorplanDataset("val")

    print(f"Training images  : {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=USE_AMP,
        persistent_workers=NUM_WORKERS > 0,
        drop_last=False,
    )

    # Validation uses the full 1024x1024 image, one at a time.
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=USE_AMP,
        persistent_workers=NUM_WORKERS > 0,
    )

    # ---- Model ----

    model = build_model(pretrained=False).to(DEVICE)
    load_v4_weights(model, BASE_CHECKPOINT)

    weights = torch.tensor(
        CLASS_WEIGHTS,
        dtype=torch.float32,
        device=DEVICE,
    )

    ce_criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    best_iou = -1.0
    epochs_without_improvement = 0

    # ---- Epoch loop ----

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'=' * 60}")
        print(f"EPOCH {epoch}/{EPOCHS}")
        print(f"{'=' * 60}")

        # ==========================================
        # TRAINING
        # ==========================================

        model.train()
        train_loss_total = 0.0
        num_batches = len(train_loader)

        optimizer.zero_grad(set_to_none=True)

        train_progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{EPOCHS} [Train]",
            unit="batch",
            dynamic_ncols=True,
            mininterval=0.5,
        )

        for batch_idx, (images, masks) in enumerate(
            train_progress,
            start=1,
        ):
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, dtype=torch.long, non_blocking=True)

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=USE_AMP,
            ):
                logits = model(images)
                loss = combined_loss(logits, masks, ce_criterion)

            scaler.scale(loss / ACCUM_STEPS).backward()

            if batch_idx % ACCUM_STEPS == 0 or batch_idx == num_batches:
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss_total += loss.item()

            train_progress.set_postfix(
                loss=f"{loss.item():.4f}",
                avg_loss=f"{train_loss_total / batch_idx:.4f}",
                refresh=False,
            )

        train_progress.close()

        train_loss = train_loss_total / max(num_batches, 1)

        # ==========================================
        # VALIDATION
        # ==========================================

        model.eval()
        val_loss_total = 0.0

        confusion = torch.zeros(
            (NUM_CLASSES, NUM_CLASSES),
            dtype=torch.int64,
            device=DEVICE,
        )

        with torch.no_grad():
            val_progress = tqdm(
                val_loader,
                desc=f"Epoch {epoch}/{EPOCHS} [Val]",
                unit="batch",
                dynamic_ncols=True,
                mininterval=0.5,
            )

            for batch_idx, (images, masks) in enumerate(
                val_progress,
                start=1,
            ):
                images = images.to(DEVICE, non_blocking=True)
                masks = masks.to(
                    DEVICE, dtype=torch.long, non_blocking=True
                )

                with torch.autocast(
                    device_type=DEVICE.type,
                    dtype=torch.float16,
                    enabled=USE_AMP,
                ):
                    logits = model(images)
                    loss = combined_loss(logits, masks, ce_criterion)

                val_loss_total += loss.item()

                predictions = logits.argmax(dim=1)

                update_confusion_matrix(confusion, predictions, masks)

                val_progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    avg_loss=f"{val_loss_total / batch_idx:.4f}",
                    refresh=False,
                )

            val_progress.close()

        val_loss = val_loss_total / max(len(val_loader), 1)

        ious, dices, mean_iou, mean_dice = calculate_metrics(confusion)

        scheduler.step(mean_iou)

        # ==========================================
        # EPOCH SUMMARY
        # ==========================================

        print(f"\nEpoch {epoch}/{EPOCHS} Summary")
        print("-" * 45)

        print(f"Train loss            : {train_loss:.4f}")
        print(f"Validation loss       : {val_loss:.4f}")
        print(f"Mean foreground IoU   : {mean_iou:.4f}")
        print(f"Mean foreground Dice  : {mean_dice:.4f}")
        print(
            f"Learning rate         : "
            f"{optimizer.param_groups[0]['lr']:.7f}"
        )

        print("\nPer-class metrics:")

        for index, name in enumerate(CLASS_NAMES):
            print(
                f"  {name:14s} "
                f"IoU={ious[index]:.4f}  "
                f"Dice={dices[index]:.4f}"
            )

        # ==========================================
        # SAVE CHECKPOINTS
        # ==========================================

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "mean_foreground_iou": mean_iou,
            "mean_foreground_dice": mean_dice,
            "class_names": CLASS_NAMES,
            "num_classes": NUM_CLASSES,
            "encoder": ENCODER,
        }

        torch.save(checkpoint, LAST_PATH)

        if mean_iou > best_iou:
            best_iou = mean_iou
            epochs_without_improvement = 0

            torch.save(checkpoint, BEST_PATH)

            print(f"\nNew best model saved: {BEST_PATH.name}")
        else:
            epochs_without_improvement += 1

            print(
                f"\nNo improvement for "
                f"{epochs_without_improvement}/{PATIENCE} epochs."
            )

        if epochs_without_improvement >= PATIENCE:
            print(
                f"\nEarly stopping after "
                f"{PATIENCE} epochs without improvement."
            )
            break

    # ---- Final summary ----

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best foreground IoU : {best_iou:.4f}")
    print(f"Best checkpoint     : {BEST_PATH}")
    print(f"Latest checkpoint   : {LAST_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()