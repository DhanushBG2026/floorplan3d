from pathlib import Path
import copy
import io
import re
import traceback
import xml.etree.ElementTree as ET

import cairosvg
import numpy as np
from PIL import Image
from tqdm import tqdm


# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------

DATASET = Path.home() / "Downloads" / "cubicasa5k" / "cubicasa5k"
OUTPUT_ROOT = Path(__file__).resolve().parent / "data"

SPLITS = ("train", "val", "test")

# None processes the complete split.
# Use 10 temporarily if you want a quick test.
LIMIT = None

# Save image+mask blends so alignment can be checked by eye.
SAVE_OVERLAYS = True
OVERLAYS_PER_SPLIT = 20

CLASS_NAMES = {
    0: "background",
    1: "wall",
    2: "door",
    3: "window",
    4: "stairs_steps",
    5: "railing",
}

# Palette colours are for viewing the masks.
# The actual pixel values remain class IDs 0-4.
PALETTE = [
    0, 0, 0,          # 0: Background - black
    220, 50, 50,      # 1: Wall - red
    40, 180, 90,      # 2: Door - green
    50, 120, 230,     # 3: Window - blue
    255, 165, 0, 
    180, 70, 220,     # 4: Stairs/steps - orange
]

GEOMETRY = {
    "path",
    "polygon",
    "rect",
    "line",
    "polyline",
    "circle",
    "ellipse",
    "use",
}

# Only these elements may be traversed. Everything else that is not
# geometry (text, tspan, image, foreignObject, ...) is dropped, because
# it would otherwise be rendered into EVERY class mask.
CONTAINERS = {"svg", "g", "switch", "a"}

SVG_NS = "http://www.w3.org/2000/svg"


# --------------------------------------------------
# CLASS DETECTION
# --------------------------------------------------

def normalise_class(value):
    """Normalize an SVG class name for reliable matching."""
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def class_id(element):
    """
    Map SVG classes to semantic label IDs.

    0 = background
    1 = wall
    2 = door
    3 = window
    4 = stairs/steps
    5 = railing
    """
    raw = element.attrib.get("class", "")
    name = normalise_class(raw)
    tokens = set(name.split())

    # Check railing BEFORE wall to prevent railing classes
    # containing both terms from being classified as walls.
    if "railing" in tokens or any(
        token.startswith("railing") for token in tokens
    ):
        return 5

    if "wall" in tokens:
        return 1

    if any(token.startswith("door") for token in tokens):
        return 2

    if any(token.startswith("window") for token in tokens):
        return 3

    if tokens.intersection({"stairs", "flight", "steps"}):
        return 4

    return None


# --------------------------------------------------
# SVG FILTERING
# --------------------------------------------------

def local_tag(element):
    """Return an SVG element's tag without its namespace."""
    return element.tag.split("}")[-1]


def filtered_svg(original_root, wanted):
    """
    Make a copy of the SVG containing geometry for one
    semantic class. Group classes are inherited by children
    unless a child supplies its own class.
    """
    root = copy.deepcopy(original_root)

    def clean(element, inherited_class=None):
        """Return True if `element` should be kept."""
        tag = local_tag(element)
        own_class = element.attrib.get("class")

        active_class = (
            own_class if own_class is not None else inherited_class
        )

        # Definitions and styles are needed by <use> and CSS.
        if tag in {"defs", "style"}:
            return True

        if tag in GEOMETRY:
            if own_class is not None:
                current_id = class_id(element)
            elif inherited_class is not None:
                temporary = ET.Element("g", {"class": inherited_class})
                current_id = class_id(temporary)
            else:
                current_id = None

            return current_id == wanted

        # Text, tspan, image, title, desc, metadata, etc. are never
        # part of a class mask.
        if tag not in CONTAINERS:
            return False

        # Containers: filter children, then drop if nothing is left.
        for child in list(element):
            if not clean(child, active_class):
                element.remove(child)

        # Keep the root <svg> even if empty so rendering still works.
        return tag == "svg" or len(element) > 0

    clean(root)
    return root


# --------------------------------------------------
# SVG DIMENSIONS AND RENDERING
# --------------------------------------------------

def _number(value):
    """Read a numeric SVG dimension such as 1200 or 1200px."""
    if value is None:
        return None

    match = re.match(
        r"^\s*([-+]?(?:\d*\.)?\d+(?:[eE][-+]?\d+)?)",
        str(value),
    )

    return float(match.group(1)) if match else None


def svg_size(root):
    """Return the SVG's (width, height) in its own units, or None."""
    view_box = root.attrib.get("viewBox")

    if view_box:
        parts = [p for p in re.split(r"[,\s]+", view_box.strip()) if p]
        try:
            numbers = [float(p) for p in parts]
        except ValueError:
            numbers = []
        if len(numbers) == 4 and numbers[2] > 0 and numbers[3] > 0:
            return numbers[2], numbers[3]

    width = _number(root.attrib.get("width"))
    height = _number(root.attrib.get("height"))

    if width and height:
        return width, height

    return None


def fit_root_to_image(root, size):
    """
    Map SVG user units 1:1 onto image pixels.

    CubiCasa5k's model.svg coordinates are in the pixel space of
    F1_scaled.png. The viewBox / width / height declared on the root
    <svg> do NOT describe that image (their aspect ratio is often
    20-30% off), so they are ignored and replaced.
    """
    width, height = size

    for attribute in ("x", "y"):
        root.attrib.pop(attribute, None)

    root.set("viewBox", f"0 0 {width} {height}")
    root.set("width", str(width))
    root.set("height", str(height))
    root.set("preserveAspectRatio", "xMinYMin meet")

    return root


def render_mask(svg_root, wanted, size):
    """Render one semantic class as a binary NumPy mask."""
    filtered_root = filtered_svg(svg_root, wanted)
    filtered_root = fit_root_to_image(filtered_root, size)

    svg_bytes = ET.tostring(
        filtered_root,
        encoding="utf-8",
        xml_declaration=True,
    )

    png_bytes = cairosvg.svg2png(
        bytestring=svg_bytes,
        output_width=size[0],
        output_height=size[1],
    )

    rendered = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    alpha = np.asarray(rendered)[:, :, 3]

    return alpha >= 128


# --------------------------------------------------
# IMAGE AND MASK OUTPUT
# --------------------------------------------------

def find_image(sample_dir):
    """
    The annotations live in the pixel space of F1_scaled.png, so that
    is the image to pair with the mask. Fall back to the original only
    if no scaled image exists.
    """
    for pattern in ("*_scaled.png", "*_original.png", "*.png"):
        candidates = sorted(sample_dir.glob(pattern))
        if candidates:
            return candidates[0]
    return None


def save_label_mask(labels, output_path):
    """Save class IDs with a visible palette."""
    mask = Image.fromarray(labels.astype(np.uint8), mode="P")
    mask.putpalette(PALETTE)
    mask.save(output_path)


def save_overlay(image, labels, output_path):
    """Blend the coloured labels over the plan to check alignment."""
    out = np.asarray(image).astype(np.float32).copy()
    colours = np.array(PALETTE, dtype=np.float32).reshape(-1, 3)[labels]
    foreground = labels > 0

    out[foreground] = 0.5 * out[foreground] + 0.5 * colours[foreground]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out.astype(np.uint8)).save(output_path)


def process_sample(relative_id, split, overlay=False):
    relative_id = relative_id.strip().replace("\\", "/").strip("/")
    sample_dir = DATASET / relative_id
    svg_path = sample_dir / "model.svg"

    if not svg_path.exists():
        raise FileNotFoundError(f"Missing SVG: {svg_path}")

    svg_root = ET.parse(svg_path).getroot()

    image_path = find_image(sample_dir)

    if image_path is None:
        raise FileNotFoundError(
            f"No PNG image found in {sample_dir}"
        )

    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    masks = {}
    for label_id in (1, 2, 3, 4, 5):
        masks[label_id] = render_mask(
            svg_root,
            label_id,
            (width, height),
        )

    labels = np.zeros((height, width), dtype=np.uint8)

    # Later assignments win where masks overlap.
    labels[masks[1]] = 1
    labels[masks[5]] = 5
    labels[masks[3]] = 3
    labels[masks[2]] = 2
    labels[masks[4]] = 4

    if not any(mask.any() for mask in masks.values()):
        raise ValueError("All rendered class masks are empty.")

    output_dir = OUTPUT_ROOT / split
    output_dir.mkdir(parents=True, exist_ok=True)

    output_name = relative_id.replace("/", "__").replace("\\", "__")

    image.save(output_dir / f"{output_name}.png")
    save_label_mask(
        labels,
        output_dir / f"{output_name}_mask.png",
    )

    if overlay:
        save_overlay(
            image,
            labels,
            OUTPUT_ROOT / "reports" / "overlays" / split
            / f"{output_name}_overlay.png",
        )

    return {
        CLASS_NAMES[label_id]: int((labels == label_id).sum())
        for label_id in range(6)
    }


# --------------------------------------------------
# DATASET PREPARATION
# --------------------------------------------------

def main():
    if not DATASET.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET}"
        )

    for split in SPLITS:
        split_file = DATASET / f"{split}.txt"

        if not split_file.exists():
            print(f"\nSkipping {split}: {split_file} not found.")
            continue

        entries = [
            line.strip()
            for line in split_file.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        if LIMIT is not None:
            entries = entries[:LIMIT]

        print(f"\n{'=' * 60}")
        print(f"Processing {split}: {len(entries)} samples")
        print(f"{'=' * 60}")

        failures = []
        totals = {name: 0 for name in CLASS_NAMES.values()}
        success_count = 0

        for index, relative_id in enumerate(
            tqdm(entries, desc=split)
        ):
            try:
                counts = process_sample(
                    relative_id,
                    split,
                    overlay=SAVE_OVERLAYS
                    and index < OVERLAYS_PER_SPLIT,
                )

                for name, count in counts.items():
                    totals[name] += count

                success_count += 1

            except Exception as exc:
                failures.append(
                    f"{relative_id}: {exc}\n"
                    f"{traceback.format_exc()}\n"
                )

        reports_dir = OUTPUT_ROOT / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        failure_file = reports_dir / f"{split}_failures.txt"
        failure_file.write_text(
            "\n".join(failures),
            encoding="utf-8",
        )

        print(f"\n{split.upper()} RESULTS")
        print(f"Successful samples: {success_count}")
        print(f"Failed samples:     {len(failures)}")

        print("\nLabel pixel counts:")
        for name, count in totals.items():
            print(f"  {name:15s}: {count:,}")

        print(f"Failure report: {failure_file}")

    print("\nDataset preparation complete.")


if __name__ == "__main__":
    main()