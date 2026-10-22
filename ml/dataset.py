from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


NUM_CLASSES = 6

# Every plan is resized/padded to this size (width, height).
IMAGE_SIZE = (1024, 1024)

# Training uses random crops of this size taken from the 1024 image,
# which keeps GPU memory low. Validation/inference use the full image.
CROP_SIZE = 512

# ImageNet statistics, required by the pretrained encoder.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resize_mask_keep_thin(mask, new_size, threshold=0.3):
    """
    Downscale a class-ID mask without losing thin structures.

    Nearest-neighbour resizing skips most pixels of 2-3 px wide lines
    (railings, thin walls), leaving them dashed or missing. Here each
    foreground class is area-averaged separately; a pixel takes the class
    with the highest coverage, if that coverage is at least `threshold`.
    """
    array = np.asarray(mask)
    new_width, new_height = new_size

    label = np.zeros((new_height, new_width), dtype=np.uint8)
    best = np.zeros((new_height, new_width), dtype=np.float32)

    for class_id in range(1, NUM_CLASSES):
        binary = Image.fromarray(
            ((array == class_id) * 255).astype(np.uint8)
        )
        score = np.asarray(
            binary.resize(new_size, Image.Resampling.BOX),
            dtype=np.float32,
        ) / 255.0

        better = (score >= threshold) & (score > best)
        label[better] = class_id
        best[better] = score[better]

    return Image.fromarray(label, mode="L")


def resize_and_pad(image, target_size, is_mask=False):
    """
    Resize while preserving aspect ratio, then pad to target size.

    Images use white padding.
    Masks use class 0 (background) padding.
    """
    target_width, target_height = target_size
    original_width, original_height = image.size

    scale = min(
        target_width / original_width,
        target_height / original_height,
    )

    new_width = max(1, round(original_width * scale))
    new_height = max(1, round(original_height * scale))

    if is_mask:
        image = resize_mask_keep_thin(image, (new_width, new_height))
    else:
        image = image.resize(
            (new_width, new_height), Image.Resampling.BILINEAR
        )

    if is_mask:
        # Same mode as the source mask ("P" or "L") so IDs are kept.
        canvas = Image.new(
            image.mode,
            (target_width, target_height),
            color=0,
        )

        palette = image.getpalette()
        if palette is not None:
            canvas.putpalette(palette)
    else:
        canvas = Image.new(
            "RGB",
            (target_width, target_height),
            color=(255, 255, 255),
        )

    left = (target_width - new_width) // 2
    top = (target_height - new_height) // 2

    canvas.paste(image, (left, top))

    return canvas


def random_crop(image, mask, size, tries=8):
    """
    Random square crop. Prefers crops that contain some foreground so
    the model does not train on blank paper margins too often.
    """
    height, width = mask.shape

    if height <= size and width <= size:
        return image, mask

    best = None

    for _ in range(tries):
        top = random.randint(0, max(0, height - size))
        left = random.randint(0, max(0, width - size))

        crop = mask[top:top + size, left:left + size]
        foreground = float((crop > 0).mean())

        if best is None or foreground > best[0]:
            best = (foreground, top, left)

        if foreground >= 0.01:
            break

    _, top, left = best

    return (
        image[top:top + size, left:left + size],
        mask[top:top + size, left:left + size],
    )


def augment(image, mask):
    """Flips, 90-degree rotations and brightness/contrast jitter."""
    if random.random() < 0.5:
        image, mask = image[:, ::-1], mask[:, ::-1]

    if random.random() < 0.5:
        image, mask = image[::-1], mask[::-1]

    k = random.randint(0, 3)
    if k:
        image, mask = np.rot90(image, k), np.rot90(mask, k)

    # Photometric jitter touches the image only.
    image = np.clip(
        image * random.uniform(0.8, 1.2) + random.uniform(-0.1, 0.1),
        0.0,
        1.0,
    )

    return (
        np.ascontiguousarray(image, dtype=np.float32),
        np.ascontiguousarray(mask),
    )


class FloorplanDataset(Dataset):
    def __init__(self, split, training=None):
        """
        training=None  -> crops and augmentation only for the train split.
        training=False -> always full image, no augmentation
                          (use this when evaluating the train split).
        """
        self.image_size = IMAGE_SIZE
        self.num_classes = NUM_CLASSES
        self.training = (
            (split == "train") if training is None else training
        )

        self.base_dir = Path(__file__).resolve().parent
        self.split_dir = self.base_dir / "data" / split

        if not self.split_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset split not found: {self.split_dir}\n"
                "Expected folders: ml/data/train, ml/data/val, ml/data/test"
            )

        self.mask_paths = sorted(self.split_dir.glob("*_mask.png"))

        if not self.mask_paths:
            raise FileNotFoundError(
                f"No *_mask.png files found in {self.split_dir}"
            )

        self.samples = []

        for mask_path in self.mask_paths:
            image_name = mask_path.name.replace("_mask.png", ".png")
            image_path = self.split_dir / image_name

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image missing for mask {mask_path.name}: "
                    f"{image_path}"
                )

            self.samples.append((image_path, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        # A size mismatch means the pair is misaligned before resizing.
        if image.size != mask.size:
            raise ValueError(
                f"Image/mask size mismatch for {image_path.name}: "
                f"{image.size} vs {mask.size}"
            )

        image = resize_and_pad(image, self.image_size, is_mask=False)
        mask = resize_and_pad(mask, self.image_size, is_mask=True)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = np.asarray(mask, dtype=np.uint8).copy()

        unique_labels = np.unique(mask_array)

        if not np.all(unique_labels < NUM_CLASSES):
            raise ValueError(
                f"Invalid class IDs in {mask_path}: "
                f"{unique_labels.tolist()}"
            )

        if self.training:
            image_array, mask_array = random_crop(
                image_array, mask_array, CROP_SIZE
            )
            image_array, mask_array = augment(image_array, mask_array)

        # Normalise with ImageNet statistics for the pretrained encoder.
        image_array = (image_array - IMAGENET_MEAN) / IMAGENET_STD

        image_tensor = torch.from_numpy(
            image_array.transpose(2, 0, 1).astype(np.float32).copy()
        )
        mask_tensor = torch.from_numpy(
            mask_array.astype(np.int64).copy()
        )

        return image_tensor, mask_tensor