
from pathlib import Path
import copy
import io
import re
import xml.etree.ElementTree as ET

import cairosvg
import numpy as np
from PIL import Image


DATASET = Path.home() / "Downloads" / "cubicasa5k" / "cubicasa5k"
SAMPLE_DIR = DATASET / "colorful" / "10052"
SVG_PATH = SAMPLE_DIR / "model.svg"
IMAGE_PATH = SAMPLE_DIR / "F1_original.png"

OUTPUT_DIR = Path("ml") / "output"

# Class IDs
BACKGROUND = 0
WALL = 1
DOOR = 2
WINDOW = 3

CLASS_NAMES = {
    WALL: "wall",
    DOOR: "door",
    WINDOW: "window",
}

COLORS = {
    BACKGROUND: (0, 0, 0),
    WALL: (220, 40, 40),       # Red
    DOOR: (40, 180, 70),       # Green
    WINDOW: (40, 90, 230),     # Blue
}

TARGET_CLASSES = {
    WALL: re.compile(r"\bwall\b", re.IGNORECASE),
    DOOR: re.compile(r"\bdoors?\b", re.IGNORECASE),
    WINDOW: re.compile(r"\bwindows?\b", re.IGNORECASE),
}

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"


def local_tag(element):
    """Return an SVG element's tag without its namespace."""
    return element.tag.split("}")[-1]


def element_class(element):
    return element.attrib.get("class", "").strip()


def target_class(element):
    """Identify only explicit wall, door, or window groups."""
    if local_tag(element) != "g":
        return None

    label = element_class(element)

    for class_id, pattern in TARGET_CLASSES.items():
        if pattern.search(label):
            return class_id

    return None


def remove_from_parent(parent, child):
    parent.remove(child)


def filter_svg_for_class(root, wanted_class):
    """
    Keep only geometry belonging to the requested class.

    Nested target groups override their parent's class. Non-target
    groups retain their parent class, allowing nested SVG transforms
    and groups such as Panel/Glass to remain inside a target object.
    """
    filtered_root = copy.deepcopy(root)

    def prune(parent, active_class=None, is_root=False):
        for child in list(parent):
            tag = local_tag(child)

            # Definitions are retained so that references used by
            # retained geometry can still be resolved by the renderer.
            if tag in {"defs", "metadata", "title", "desc"}:
                continue

            child_target = target_class(child) if tag == "g" else None

            if child_target is not None:
                # A nested wall/door/window starts a new class region.
                if child_target != wanted_class:
                    remove_from_parent(parent, child)
                    continue

                prune(child, child_target)
                continue

            if tag == "g":
                prune(child, active_class)
                if len(child) == 0 and not (child.text or "").strip():
                    remove_from_parent(parent, child)
                continue

            # Keep geometric elements only while inside the requested
            # class. Ignore text and unrelated SVG elements.
            if tag in {
                "path", "polygon", "rect", "line", "polyline",
                "circle", "ellipse", "use"
            }:
                if active_class != wanted_class:
                    remove_from_parent(parent, child)
                continue

            # For other elements, recurse where appropriate.
            if len(child):
                prune(child, active_class)
            elif active_class != wanted_class:
                remove_from_parent(parent, child)

    # The root SVG is a container, not a labelled object.
    prune(filtered_root, None, is_root=True)
    return filtered_root


def render_class_mask(root, class_id, width, height):
    """Render one class to a binary mask using the SVG renderer."""
    filtered = filter_svg_for_class(root, class_id)

    svg_bytes = ET.tostring(
        filtered,
        encoding="utf-8",
        xml_declaration=True,
    )

    png_bytes = cairosvg.svg2png(
        bytestring=svg_bytes,
        output_width=width,
        output_height=height,
    )

    image = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    rgba = np.asarray(image)

    # A pixel belongs to the class if the filtered SVG draws visible
    # content there. This is a first-pass geometric mask.
    return rgba[:, :, 3] > 0


def save_color_mask(label_mask, path):
    rgb = np.zeros(
        (*label_mask.shape, 3),
        dtype=np.uint8,
    )

    for class_id, color in COLORS.items():
        rgb[label_mask == class_id] = color

    Image.fromarray(rgb).save(path)


def main():
    if not SVG_PATH.exists():
        raise FileNotFoundError(f"SVG not found: {SVG_PATH}")

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    reference = Image.open(IMAGE_PATH).convert("RGB")
    width, height = reference.size

    root = ET.parse(SVG_PATH).getroot()

    print("Sample:", SAMPLE_DIR)
    print("Reference dimensions:", width, "x", height)
    print("SVG viewBox:", root.attrib.get("viewBox"))

    masks = {}

    for class_id, class_name in CLASS_NAMES.items():
        print(f"Rendering {class_name} mask...")

        mask = render_class_mask(
            root,
            class_id,
            width,
            height,
        )

        masks[class_id] = mask

        print(
            f"  {class_name}: "
            f"{int(mask.sum()):,} foreground pixels"
        )

    # Start with background.
    label_mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    # Apply walls first, then openings/objects.
    # Doors and windows take priority where classes overlap.
    label_mask[masks[WALL]] = WALL
    label_mask[masks[DOOR]] = DOOR
    label_mask[masks[WINDOW]] = WINDOW

    # Save the single-channel label mask.
    labels_path = OUTPUT_DIR / "labels.png"
    Image.fromarray(label_mask).save(labels_path)

    # Save a color-coded annotation preview.
    preview_path = OUTPUT_DIR / "annotation_preview_v2.png"
    save_color_mask(label_mask, preview_path)

    # Blend annotations over the original plan.
    original = np.asarray(reference).astype(np.float32)

    color_image = np.zeros_like(original, dtype=np.float32)
    for class_id, color in COLORS.items():
        color_image[label_mask == class_id] = color

    foreground = label_mask != BACKGROUND
    overlay = original.copy()

    overlay[foreground] = (
        0.50 * original[foreground]
        + 0.50 * color_image[foreground]
    )

    overlay_path = OUTPUT_DIR / "overlay_v2.png"
    Image.fromarray(
        np.clip(overlay, 0, 255).astype(np.uint8)
    ).save(overlay_path)

    reference_path = OUTPUT_DIR / "reference.png"
    reference.save(reference_path)

    print("\nSaved files:")
    print("Reference:", reference_path)
    print("Labels:", labels_path)
    print("Preview:", preview_path)
    print("Overlay:", overlay_path)
    print("\nClass IDs: background=0, wall=1, door=2, window=3")
    print(
        "\nIMPORTANT: This is a first-pass mask generator. "
        "Inspect the output before training."
    )


if __name__ == "__main__":
    main()
