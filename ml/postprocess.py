"""
Clean up a predicted floor-plan label mask so it is sharp and easy to
turn into 3D geometry.

Pure numpy + scipy; no model needed. Class IDs:
    0 background, 1 wall, 2 door, 3 window, 4 stairs, 5 railing

Main entry point:
    clean_mask(labels, probs=None, config=None) -> cleaned labels (uint8)

`labels` is an (H, W) uint8 array at the ORIGINAL plan resolution.
`probs` is optional: an (6, H, W) float array of model probabilities at
the same resolution. When given, low-confidence door/window/railing/stairs
regions are removed too.

Every size below is written in pixels of a 1024-pixel-wide plan and is
scaled automatically to the real image size, so the same settings work for
small and large plans.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

BACKGROUND, WALL, DOOR, WINDOW, STAIRS, RAILING = range(6)
NUM_CLASSES = 6

S8 = np.ones((3, 3), dtype=bool)


@dataclass
class CleanConfig:
    # ---- tiny blobs (area, in px^2 of a 1024 plan) ----
    min_area: dict = field(default_factory=lambda: {
        WALL: 120, DOOR: 50, WINDOW: 80, STAIRS: 1200, RAILING: 50,
    })

    # ---- 1. remove ticks / "+" marks / jagged bumps from walls & railings
    # A wall pixel survives only if it belongs to a straight horizontal or
    # vertical run at least this long (px at 1024 scale).
    straight_run: int = 14
    # Leftover wall pixels (e.g. a diagonal wall) are kept only if the
    # leftover piece is bigger than this area; ticks and bumps are smaller.
    keep_leftover_area: int = 90

    # ---- 2. close hairline gaps inside wall lines (px, 1024 scale)
    close_gap: int = 5

    # ---- 3. doors / windows must sit on a wall
    attach_distance: int = 7        # px: max gap to a wall/railing
    min_opening_length: int = 10    # px: shortest door/window kept

    # ---- 4. railings must be connected to the building, or be long
    railing_attach_distance: int = 9
    railing_isolated_min_length: int = 260   # px, longest side of its box

    # ---- 5. low-confidence regions (only when probs are supplied)
    # Mean model probability a region needs to be kept. Deliberately low:
    # only clearly doubtful regions go. Raise for stricter cleaning.
    min_confidence: dict = field(default_factory=lambda: {
        DOOR: 0.50, WINDOW: 0.50, STAIRS: 0.50, RAILING: 0.50,
    })

    # ---- 6. boundary smoothing (runs before straightening) ----
    smooth_sigma: float = 1.0       # px at 1024 scale; 0 turns it off
    smooth_classes: tuple = (WALL, DOOR, WINDOW, STAIRS, RAILING)

    # ---- 7. straighten: replace every straight run by an exact rectangle
    straighten: bool = True
    straighten_classes: tuple = (WALL, DOOR, WINDOW, RAILING)
    # parallel rectangles whose centre lines differ by less than this are
    # moved onto one common line (px, 1024 scale)
    align_tolerance: float = 2.0
    # stairs blobs that fill at least this much of their bounding box are
    # turned into that rectangle
    stairs_fill_ratio: float = 0.80


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _scale(labels):
    return max(labels.shape) / 1024.0


def _px(value, scale, minimum=1):
    return max(minimum, int(round(value * scale)))


def _remove_small(mask, min_area):
    """Drop connected components smaller than min_area."""
    if not mask.any():
        return mask
    lab, n = ndi.label(mask, structure=S8)
    areas = ndi.sum(mask, lab, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = areas >= min_area
    return keep[lab]


def _line_open(mask, length):
    """Opening with a horizontal and a vertical line; union of both."""
    if length <= 1:
        return mask
    horizontal = ndi.binary_opening(
        mask, structure=np.ones((1, length), dtype=bool))
    vertical = ndi.binary_opening(
        mask, structure=np.ones((length, 1), dtype=bool))
    return horizontal | vertical


def _line_close(mask, length):
    """Bridge gaps shorter than `length` along each axis."""
    if length <= 1:
        return mask
    horizontal = ndi.binary_closing(
        mask, structure=np.ones((1, length), dtype=bool))
    vertical = ndi.binary_closing(
        mask, structure=np.ones((length, 1), dtype=bool))
    return horizontal | vertical


def estimate_thickness(mask):
    """
    Typical stroke thickness (pixels) of a line-like mask: the 85th
    percentile of the local width measured along the middle of the strokes.
    Returns 0 for an empty mask.
    """
    if not mask.any():
        return 0.0
    dist = ndi.distance_transform_edt(mask)
    ridge = (dist == ndi.maximum_filter(dist, size=3)) & (dist >= 1.0)
    return float(2.0 * np.percentile(dist[ridge], 85) - 1.0)


def run_length_for(mask, base, factor=1.7):
    """Straight-run length: at least `base`, and clearly longer than the
    stroke is thick (otherwise a thick wall would count as both a
    horizontal and a vertical run)."""
    return int(max(base, np.ceil(factor * estimate_thickness(mask))))


def remove_ticks(mask, straight_run, keep_leftover_area):
    """
    Remove small protrusions (ticks, '+' marks, jagged bumps) from a
    wall-like mask. Straight horizontal/vertical runs are kept, and so are
    big leftover pieces such as diagonal walls.
    """
    if not mask.any():
        return mask

    # A run that is straight but thick (a wall block) is also covered by a
    # square opening, so blocks never get eaten.
    block = max(3, int(round(0.8 * estimate_thickness(mask))))
    solid = ndi.binary_opening(
        mask, structure=np.ones((block, block), dtype=bool))

    kept = _line_open(mask, straight_run) | solid

    leftover = mask & ~kept
    leftover = _remove_small(leftover, keep_leftover_area)

    return kept | leftover


def _touches(mask, support, distance):
    """For each component of `mask`, True if it is within `distance` of
    `support`. Returns a boolean image of the kept components."""
    if not mask.any():
        return mask
    near = ndi.binary_dilation(
        support, structure=S8, iterations=max(1, distance))
    lab, n = ndi.label(mask, structure=S8)
    hit = ndi.maximum(near, lab, index=np.arange(1, n + 1)) > 0
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = hit
    return keep[lab]


def _component_length(mask):
    """Longest bounding-box side of every component (array indexed by id)."""
    lab, n = ndi.label(mask, structure=S8)
    lengths = np.zeros(n + 1)
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        lengths[i] = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start)
    return lab, lengths


def _drop_low_confidence(mask, class_probs, threshold):
    if not mask.any():
        return mask
    lab, n = ndi.label(mask, structure=S8)
    mean_p = ndi.mean(class_probs, lab, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = np.asarray(mean_p) >= threshold
    return keep[lab]


def smooth_labels(labels, sigma, classes):
    """
    Round off jagged edges: blur each class's binary mask and keep a pixel
    for the class with the highest blurred score (at least 0.5).
    A straight line keeps its width; only staircase edges are smoothed.
    """
    if sigma <= 0:
        return labels

    best_score = np.full(labels.shape, 0.5, dtype=np.float32)
    out = np.zeros_like(labels)

    # Classes that are not smoothed are copied through unchanged first.
    for class_id in range(1, NUM_CLASSES):
        if class_id not in classes:
            out[labels == class_id] = class_id
            best_score[labels == class_id] = 2.0   # cannot be overwritten

    for class_id in classes:
        mask = labels == class_id
        if not mask.any():
            continue
        score = ndi.gaussian_filter(mask.astype(np.float32), sigma)
        better = score > best_score
        out[better] = class_id
        best_score[better] = score[better]

    return out


def _fit_runs(runs, orient):
    """
    Fit one rectangle to every connected straight run.

    Returns (rects, fallback) where rects is a list of
    [y0, y1, x0, x1] (inclusive, image coordinates) and fallback is a mask
    with the runs that were too irregular to be turned into a rectangle.
    """
    rects = []
    fallback = np.zeros(runs.shape, dtype=bool)

    lab, n = ndi.label(runs)
    for index, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        comp = lab[sl] == index
        if orient == "v":
            comp = comp.T            # work in the horizontal frame

        counts = comp.sum(axis=0)
        valid = counts > 0
        rows = np.arange(comp.shape[0])[:, None]
        centres = (comp * rows).sum(axis=0)[valid] / counts[valid]
        counts = counts[valid]

        # 75th percentile: a wall that tapers or has a few thin columns is
        # still treated as a full-thickness wall.
        thick = max(1, int(round(np.percentile(counts, 75))))
        centre = float(np.median(centres))

        irregular = (
            np.mean(counts < 0.4 * thick) > 0.4
            or np.max(np.abs(centres - centre)) > max(2.0, 0.8 * thick)
        )

        if irregular:
            region = fallback[sl]
            region |= (lab[sl] == index)
            continue

        top = int(round(centre - (thick - 1) / 2.0))
        length = comp.shape[1]
        if orient == "h":
            y0, y1 = sl[0].start + top, sl[0].start + top + thick - 1
            x0, x1 = sl[1].start, sl[1].start + length - 1
        else:
            x0, x1 = sl[1].start + top, sl[1].start + top + thick - 1
            y0, y1 = sl[0].start, sl[0].start + length - 1
        rects.append([y0, y1, x0, x1])

    return rects, fallback


def _align_rects(rects, orient, tolerance):
    """Move parallel rectangles with nearly equal centre lines onto one."""
    if len(rects) < 2:
        return
    axis = (0, 1) if orient == "h" else (2, 3)
    centres = np.array([(r[axis[0]] + r[axis[1]]) / 2.0 for r in rects])
    order = np.argsort(centres)

    start = 0
    for k in range(1, len(order) + 1):
        if k == len(order) or centres[order[k]] - centres[order[k - 1]] > tolerance:
            group = order[start:k]
            if len(group) > 1:
                target = centres[group].mean()
                for g in group:
                    shift = int(round(target - centres[g]))
                    rects[g][axis[0]] += shift
                    rects[g][axis[1]] += shift
            start = k


def _fit_openings_to_walls(all_rects):
    """
    Make every door / window as thick as the wall it sits in, so there are
    no thin wall slivers beside it. Rectangles are [y0, y1, x0, x1].
    """
    walls = {"h": [], "v": []}
    for class_id, orient, rect in all_rects:
        if class_id == WALL:
            walls[orient].append(rect)

    for class_id, orient, rect in all_rects:
        if class_id not in (DOOR, WINDOW):
            continue
        cross = (0, 1) if orient == "h" else (2, 3)
        along = (2, 3) if orient == "h" else (0, 1)
        length = rect[along[1]] - rect[along[0]] + 1
        centre = (rect[cross[0]] + rect[cross[1]]) / 2.0

        best, best_overlap = None, 0
        for wall in walls[orient]:
            overlap = (min(rect[along[1]], wall[along[1]])
                       - max(rect[along[0]], wall[along[0]]) + 1)
            thick = wall[cross[1]] - wall[cross[0]] + 1
            if (overlap >= 0.5 * length
                    and abs((wall[cross[0]] + wall[cross[1]]) / 2.0 - centre)
                    <= 0.5 * thick + 1
                    and overlap > best_overlap):
                best, best_overlap = wall, overlap

        if best is not None:
            rect[cross[0]], rect[cross[1]] = best[cross[0]], best[cross[1]]


def straighten_layers(layers, classes, run_lengths, align_tol, shape):
    """
    Replace the straight horizontal / vertical runs of the given classes by
    exact rectangles, and align parallel ones. Anything that is not a
    straight run (diagonal walls, odd shapes) is kept as it was.
    Returns (new_layers, rect_records).
    """
    h, w = shape
    all_rects = []          # (class_id, orient, rect)
    handled = {}

    block_fill = {}
    wall_only = {}
    layers = dict(layers)
    for class_id in classes:
        mask = layers[class_id]
        if not mask.any():
            continue

        # Thick solid patches (columns, wall blocks) are handled as
        # rectangles of their own so they do not distort the line fit.
        if class_id == WALL:
            size = int(np.ceil(1.6 * estimate_thickness(mask)))
            if size >= 4:
                blocks = ndi.binary_opening(
                    mask, structure=np.ones((size, size), dtype=bool))
                if blocks.any():
                    fill = rectangularize(blocks, 0.85)
                    block_fill[class_id] = fill
                    layers[class_id] = mask & ~fill

        mask = layers[class_id]
        run_length = run_lengths[class_id]

        # Walls are measured together with the doors and windows set into
        # them, so a short wall pier between two openings stays part of one
        # straight line. Doors/windows are painted on top afterwards.
        source = mask
        if class_id == WALL:
            source = mask | layers[DOOR] | layers[WINDOW]
        wall_only[class_id] = mask

        horizontal = ndi.binary_opening(
            source, structure=np.ones((1, run_length), dtype=bool))
        vertical = ndi.binary_opening(
            source, structure=np.ones((run_length, 1), dtype=bool))
        for orient, runs in (("h", horizontal), ("v", vertical)):
            rects, fallback = _fit_runs(runs, orient)
            for r in rects:
                all_rects.append((class_id, orient, r))
            handled.setdefault(class_id, np.zeros(shape, dtype=bool))
            handled[class_id] |= fallback
        handled[class_id] |= False
        handled[class_id + 1000] = horizontal | vertical

    # align collinear rectangles across all classes at once
    for orient in ("h", "v"):
        group = [item[2] for item in all_rects if item[1] == orient]
        _align_rects(group, orient, align_tol)

    _fit_openings_to_walls(all_rects)

    out = {c: np.zeros(shape, dtype=bool) for c in layers}
    for class_id, orient, (y0, y1, x0, x1) in all_rects:
        out[class_id][max(y0, 0):y1 + 1, max(x0, 0):x1 + 1] = True

    for class_id in classes:
        mask = layers[class_id]
        if class_id in block_fill:
            out[class_id] |= block_fill[class_id]
        if class_id not in handled:
            out[class_id] |= mask
            continue
        # irregular runs stay as they were
        out[class_id] |= handled[class_id]
        # pixels that are not near any straight run (e.g. diagonal walls)
        near = ndi.binary_dilation(
            handled[class_id + 1000], structure=S8,
            iterations=max(2, int(round(3 * max(h, w) / 1024.0))))
        leftover = wall_only.get(class_id, mask) & ~near
        out[class_id] |= _remove_small(
            leftover, max(20, int(90 * (max(h, w) / 1024.0) ** 2)))

    for class_id in layers:
        if class_id not in classes:
            out[class_id] = layers[class_id]

    return out, all_rects


def rectangularize(mask, fill_ratio):
    """Turn near-rectangular blobs into their exact bounding rectangle."""
    if not mask.any():
        return mask
    out = mask.copy()
    lab, n = ndi.label(mask, structure=S8)
    for index, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        area = (lab[sl] == index).sum()
        box = (sl[0].stop - sl[0].start) * (sl[1].stop - sl[1].start)
        if area / box >= fill_ratio:
            out[sl] = True
    return out


# ----------------------------------------------------------------------
# main pipeline
# ----------------------------------------------------------------------

def clean_mask(labels, probs=None, config=None, report=None):
    """Return a cleaned copy of `labels`. `report` (dict) collects counts."""
    cfg = config or CleanConfig()
    labels = labels.astype(np.uint8)
    scale = _scale(labels)
    report = report if report is not None else {}

    def note(key, before, after):
        report[key] = int(before.sum() - after.sum())

    layers = {c: labels == c for c in range(1, NUM_CLASSES)}

    # 0. tiny blobs
    for class_id, area in cfg.min_area.items():
        layers[class_id] = _remove_small(
            layers[class_id], int(area * scale * scale))

    # 1. ticks / bumps on walls and railings
    base_run = _px(cfg.straight_run, scale, 3)
    for class_id in (WALL, RAILING):
        before = layers[class_id]
        layers[class_id] = remove_ticks(
            before,
            run_length_for(before, base_run),
            int(cfg.keep_leftover_area * scale * scale),
        )
        note(f"ticks_removed_{'wall' if class_id == WALL else 'railing'}",
             before, layers[class_id])

    # 2. hairline gaps inside wall lines (only fills background pixels)
    occupied = np.zeros(labels.shape, dtype=bool)
    for class_id in (DOOR, WINDOW, STAIRS, RAILING):
        occupied |= layers[class_id]
    closed = _line_close(layers[WALL], _px(cfg.close_gap, scale, 2))
    layers[WALL] = layers[WALL] | (closed & ~occupied)

    # 3. doors and windows must be attached to a wall (or railing)
    wall_like = layers[WALL] | layers[RAILING]
    for class_id in (DOOR, WINDOW):
        before = layers[class_id]
        kept = _touches(
            before, wall_like, _px(cfg.attach_distance, scale))
        kept = _remove_small(kept, 1)
        # shortest allowed length
        lab, lengths = _component_length(kept)
        long_enough = lengths >= _px(cfg.min_opening_length, scale)
        long_enough[0] = False
        layers[class_id] = long_enough[lab]
        note(f"detached_removed_{'door' if class_id == DOOR else 'window'}",
             before, layers[class_id])

    # 4. railings: connected to the building, or genuinely long
    before = layers[RAILING]
    if before.any():
        support = (layers[WALL] | layers[DOOR] | layers[WINDOW]
                   | layers[STAIRS])
        attached = _touches(
            before, support, _px(cfg.railing_attach_distance, scale))
        lab, lengths = _component_length(before)
        long_enough = lengths >= _px(cfg.railing_isolated_min_length, scale)
        long_enough[0] = False
        layers[RAILING] = attached | long_enough[lab]
        layers[RAILING] &= before
    note("railing_isolated_removed", before, layers[RAILING])

    # 5. model confidence
    if probs is not None:
        for class_id, threshold in cfg.min_confidence.items():
            before = layers[class_id]
            layers[class_id] = _drop_low_confidence(
                before, probs[class_id], threshold)
            note(f"low_confidence_removed_{class_id}",
                 before, layers[class_id])

    # 6. smooth jagged boundaries (per class, before straightening)
    sigma = cfg.smooth_sigma * max(scale, 1.0) if cfg.smooth_sigma else 0
    if sigma > 0:
        tmp = np.zeros_like(labels)
        for class_id in (WALL, RAILING, WINDOW, DOOR, STAIRS):
            tmp[layers[class_id]] = class_id
        tmp = smooth_labels(tmp, sigma, cfg.smooth_classes)
        layers = {c: tmp == c for c in range(1, NUM_CLASSES)}

    # 7. exact rectangles for straight runs; rectangular stairs
    rects = []
    if cfg.straighten:
        run_lengths = {
            c: run_length_for(layers[c], base_run)
            for c in cfg.straighten_classes
        }
        report["run_lengths"] = run_lengths
        layers, rects = straighten_layers(
            layers,
            cfg.straighten_classes,
            run_lengths,
            cfg.align_tolerance * max(scale, 1.0),
            labels.shape,
        )
        layers[STAIRS] = rectangularize(layers[STAIRS], cfg.stairs_fill_ratio)

    report["rects"] = rects

    # compose the final label image (later classes win on overlap)
    out = np.zeros_like(labels)
    for class_id in (WALL, RAILING, WINDOW, DOOR, STAIRS):
        out[layers[class_id]] = class_id

    return out