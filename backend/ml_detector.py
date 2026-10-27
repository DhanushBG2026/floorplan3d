"""Optional six-class U-Net detector; deliberately independent of detect.py.

Pipeline: image -> U-Net probabilities -> ml/postprocess.clean_mask -> clean label mask
-> non-overlapping axis-aligned boxes (same dict format detect.py returns, in metres).
"""
from functools import lru_cache
from pathlib import Path
import importlib.util
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from scipy import ndimage as ndi

NUM_CLASSES = 6
WALL, DOOR, WINDOW, STAIRS, RAILING = 1, 2, 3, 4, 5
IMAGE_SIZE = 1024
MAX_WORK_SIDE = 2048          # same cap predict.py uses; bigger plans are cleaned at this size
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
S8 = np.ones((3, 3), dtype=bool)

# Door sizing (metres): EVERY detected door is made exactly 3.5 ft wide (centred where it was
# detected). Door HEIGHT (7 ft) is set in the frontend.
NORMALISE_DOORS = True
FT = 0.3048
DOOR_STD = 3.5 * FT
DOOR_NOISE = 0.15            # components narrower than this (m) are noise, left alone
# Wall rescue: the model sometimes misses walls (especially on plans drawn differently from its
# training data). Thick dark strokes in the drawing that are connected to walls it DID find are
# added as walls. Set RESCUE_WALLS = False to use the raw model output only.
RESCUE_WALLS = True
RESCUE_REACH_M = 3.0         # a rescued wall may extend at most this far from a detected wall
DIAGONAL_MIN_DEG = 8.0        # a wall this far from horizontal/vertical is treated as diagonal


# ---------------------------------------------------------------- loading
@lru_cache(maxsize=1)
def _load_postprocessor():
    here = Path(__file__).resolve()
    candidates = [here.parent.parent / "ml" / "postprocess.py",
                  here.parent / "ml" / "postprocess.py"]
    path = next((c for c in candidates if c.is_file()), None)
    if path is None:
        raise FileNotFoundError("Missing ml/postprocess.py; expected project-root ml/postprocess.py")
    spec = importlib.util.spec_from_file_location("floorplan_ml_postprocess", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolves the module by name
    spec.loader.exec_module(module)
    return module


def _checkpoint_path():
    here = Path(__file__).resolve()
    for p in (here.parent.parent / "ml" / "output" / "best_unet_v5.pth",
              here.parent / "weights" / "best_unet_v5.pth"):
        if p.is_file():
            return p
    raise FileNotFoundError("Missing best_unet_v5.pth; expected ml/output/ or backend/weights/")


@lru_cache(maxsize=1)
def _load_model():
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3,
                     classes=NUM_CLASSES).to(DEVICE)
    checkpoint = torch.load(_checkpoint_path(), map_location=DEVICE)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# ---------------------------------------------------------------- inference
def _predict(gray):
    """Returns (cleaned label mask at work size, work_size_scale)."""
    h, w = gray.shape[:2]
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    scale = min(IMAGE_SIZE / w, IMAGE_SIZE / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    xoff, yoff = (IMAGE_SIZE - nw) // 2, (IMAGE_SIZE - nh) // 2
    canvas = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 255, dtype=np.uint8)
    canvas[yoff:yoff + nh, xoff:xoff + nw] = resized
    arr = (canvas.astype(np.float32) / 255.0 - MEAN) / STD
    tensor = torch.from_numpy(arr.transpose(2, 0, 1).copy()).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        probs = torch.softmax(_load_model()(tensor).float(), dim=1)
        probs = probs[:, :, yoff:yoff + nh, xoff:xoff + nw]
        # smooth (bilinear) upsampling of the PROBABILITIES, then argmax: much cleaner edges
        # than resizing the label image, and it is what predict.py does.
        shrink = min(1.0, MAX_WORK_SIDE / max(w, h))
        ww, wh = max(1, round(w * shrink)), max(1, round(h * shrink))
        probs = F.interpolate(probs, size=(wh, ww), mode="bilinear", align_corners=False)
        labels = probs.argmax(1)[0].byte().cpu().numpy()
        probs = probs[0].cpu().numpy()

    cleaned = _load_postprocessor().clean_mask(labels, probs=probs)
    return cleaned.astype(np.uint8), shrink


# ---------------------------------------------------------------- mask -> boxes
def _thickness(binary):
    return max(2.0, float(_load_postprocessor().estimate_thickness(binary)))


def _diagonal_from(component, min_len):
    """Rotated-rectangle fit of one leftover blob. Returns (cx, cy, length, thickness, angle_deg) or None."""
    ys, xs = np.nonzero(component)
    if len(xs) < 12:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float32)
    (cx, cy), (a, b), ang = cv2.minAreaRect(pts)
    length, thick = max(a, b), min(a, b)
    if thick < 2 or length < max(min_len, 2.0 * thick) or len(xs) / (a * b + 1e-6) < 0.6:
        return None
    theta = ang if a >= b else ang + 90.0           # direction of the long side
    theta = (theta + 180.0) % 180.0
    off_axis = min(theta % 90.0, 90.0 - theta % 90.0)
    if off_axis < DIAGONAL_MIN_DEG:
        return None
    return cx, cy, length, thick, theta


def _runs_to_rects(mask):
    """Exact cover of a binary mask by rectangles: horizontal runs per row, merged downwards when
    the next row has the identical run. Returns (x0, y0, x1, y1) with x1/y1 exclusive."""
    out, open_ = [], {}
    for y in range(mask.shape[0] + 1):
        row = mask[y] if y < mask.shape[0] else np.zeros(mask.shape[1], bool)
        d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
        runs = set(zip(np.nonzero(d == 1)[0].tolist(), np.nonzero(d == -1)[0].tolist()))
        for run in list(open_):
            if run not in runs:
                out.append((run[0], open_.pop(run), run[1], y))
        for run in runs:
            open_.setdefault(run, y)
    return out


def _fit_boxes(binary, min_len, min_area, allow_diagonal=False, ppm_hint=None):
    """Split a (mostly) rectilinear mask into NON-OVERLAPPING axis-aligned boxes.

    Rounds of straight-run detection (long horizontal runs first, then vertical runs in what is
    left, then again with shorter runs). The run length is always clearly longer than the wall
    thickness, so a thick wall is never mistaken for a run in the other direction. After the first
    round, elongated leftovers that are NOT axis-aligned are fitted as rotated rectangles
    (diagonal walls) instead of being turned into a fat bounding box. Whatever else is left is
    added as plain bounding boxes so no wall pixel is lost.
    Returns (boxes, diagonals): boxes = (x0, y0, x1, y1, horizontal) in pixels, x1/y1 exclusive;
    diagonals = (cx, cy, length, thickness, angle_deg) in pixels.
    """
    pp = _load_postprocessor()
    boxes, diagonals = [], []
    rest = binary.copy()
    t = _thickness(binary)
    for round_no, factor in enumerate((1.6, 1.0, 0.6)):
        k = max(int(min_len), int(np.ceil(factor * t)), 3)
        for orient in ("h", "v"):
            if not rest.any():
                break
            st = np.ones((1, k) if orient == "h" else (k, 1), dtype=bool)
            runs = ndi.binary_opening(rest, structure=st)
            if not runs.any():
                continue
            rects, _ = pp._fit_runs(runs, orient)
            for y0, y1, x0, x1 in rects:
                y0, x0 = max(0, y0), max(0, x0)
                # a box must actually be made of wall; otherwise leave those pixels for later rounds
                if rest[y0:y1 + 1, x0:x1 + 1].mean() < 0.65:
                    continue
                boxes.append((x0, y0, x1 + 1, y1 + 1, orient == "h"))
                rest[y0:y1 + 1, x0:x1 + 1] = False
            # pixels of irregular runs (not turned into a rectangle) stay in `rest`
        if round_no == 0 and allow_diagonal and rest.any():
            lab, n = ndi.label(rest, structure=S8)
            for i, sl in enumerate(ndi.find_objects(lab), start=1):
                if sl is None:
                    continue
                comp = lab[sl] == i
                if comp.sum() < min_area:
                    continue
                full = np.zeros_like(rest)
                full[sl] = comp
                fit = _diagonal_from(full, max(min_len, 2.5 * t))
                if fit:
                    diagonals.append(fit)
                    rest[sl] &= ~comp
    # Whatever is left: exact rectangles from merged row runs (never one big bounding box around
    # an L-shaped or ring-shaped leftover, which would turn a whole room into a solid slab).
    for x0, y0, x1, y1 in _runs_to_rects(rest):
        if (x1 - x0) * (y1 - y0) >= min_area and min(x1 - x0, y1 - y0) >= 2:
            boxes.append((x0, y0, x1, y1, (x1 - x0) >= (y1 - y0)))
    # Final safety net: a real wall is never a room-sized block.
    limit = max(1.0 * ppm_hint, 3.5 * t) if ppm_hint else 3.5 * t * 3
    boxes = [b for b in boxes if min(b[2] - b[0], b[3] - b[1]) <= limit]
    return boxes, diagonals


def _wall_dicts(boxes, ppm, diagonals=()):
    out = []
    for x0, y0, x1, y1, hz in boxes:
        length, thick = ((x1 - x0), (y1 - y0)) if hz else ((y1 - y0), (x1 - x0))
        out.append({"id": len(out), "x0": x0 / ppm, "y0": y0 / ppm, "x1": x1 / ppm, "y1": y1 / ppm,
                    "horizontal": bool(hz), "length": length / ppm, "thickness": thick / ppm})
    for cx, cy, length, thick, theta in diagonals:
        r = np.deg2rad(float(theta))
        cx, cy, length, thick = float(cx), float(cy), float(length), float(thick)
        dx, dy = np.cos(r) * length / 2, np.sin(r) * length / 2
        ex, ey = abs(dx) + abs(np.sin(r)) * thick / 2, abs(dy) + abs(np.cos(r)) * thick / 2
        out.append({"id": len(out), "diagonal": True,
                    "ax": (cx - dx) / ppm, "ay": (cy - dy) / ppm, "bx": (cx + dx) / ppm, "by": (cy + dy) / ppm,
                    "x0": (cx - ex) / ppm, "y0": (cy - ey) / ppm, "x1": (cx + ex) / ppm, "y1": (cy + ey) / ppm,
                    "horizontal": bool(abs(dx) >= abs(dy)), "length": length / ppm, "thickness": thick / ppm})
    return out


def _bboxes(binary, min_area):
    lab, n = ndi.label(binary, structure=S8)
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        area = int((lab[sl] == i).sum())
        if area >= min_area:
            out.append((sl[1].start, sl[0].start, sl[1].stop, sl[0].stop, area))
    return out


def _openings(binary, kind, ppm, min_len_px):
    out = []
    for x0, y0, x1, y1, area in _bboxes(binary, 4):
        w, h = x1 - x0, y1 - y0
        hz = w >= h
        length, thick = (w, h) if hz else (h, w)
        if length < min_len_px:
            continue
        out.append({"type": kind, "horizontal": bool(hz),
                    "start": (x0 if hz else y0) / ppm, "end": (x1 if hz else y1) / ppm,
                    "centre": ((y0 + y1) / 2 if hz else (x0 + x1) / 2) / ppm,
                    "thickness": max(2, thick) / ppm, "width": length / ppm,
                    "ink_density": min(1.0, area / max(1, w * h))})
    return out


def _stairs(binary, ppm, min_area):
    out = []
    for x0, y0, x1, y1, area in _bboxes(binary, min_area):
        w, h = x1 - x0, y1 - y0
        hz = w >= h
        length, width = (w, h) if hz else (h, w)
        out.append({"id": len(out), "x0": x0 / ppm, "y0": y0 / ppm, "x1": x1 / ppm, "y1": y1 / ppm,
                    "horizontal": bool(hz), "length": length / ppm, "width": width / ppm,
                    "steps": int(np.clip(round(length / max(width / 2.2, 1)), 4, 16))})
    return out


def _split_merged_doors(mask, ppm):
    """Split a wide connected door component only when its long-axis projection has a clear valley.

    Only DOOR pixels are changed; WINDOW pixels are never touched here. A short WALL divider
    breaks 8-connected contact so downstream opening extraction sees two door components.
    """
    out = mask.copy()
    lab, _ = ndi.label(mask == DOOR, structure=S8)
    target = max(3, int(round(DOOR_STD * ppm)))
    min_piece = max(3, int(round(0.55 * target)))

    for component_id, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        comp = lab[sl] == component_id
        h, w = comp.shape
        horizontal = w >= h
        axis_len = w if horizontal else h
        if axis_len < 1.55 * target:
            continue

        projection = comp.sum(axis=0 if horizontal else 1).astype(np.float32)
        smooth_width = max(3, int(round(0.08 * target)))
        if smooth_width % 2 == 0:
            smooth_width += 1
        if projection.size >= smooth_width:
            projection = np.convolve(
                projection, np.ones(smooth_width, dtype=np.float32) / smooth_width, mode="same"
            )

        lower = max(min_piece, int(0.55 * target))
        upper = min(axis_len - min_piece, int(1.45 * target))
        if upper <= lower:
            continue
        peak = float(projection.max())
        if peak <= 0:
            continue

        cut = min(range(lower, upper + 1),
                  key=lambda i: (float(projection[i]), abs(i - target)))
        if float(projection[cut]) > 0.55 * peak:
            continue

        left = comp[:, :cut] if horizontal else comp[:cut, :]
        right = comp[:, cut:] if horizontal else comp[cut:, :]
        left_len = left.shape[1] if horizontal else left.shape[0]
        right_len = right.shape[1] if horizontal else right.shape[0]
        if left_len < min_piece or right_len < min_piece:
            continue
        if left.sum() < min_piece * 2 or right.sum() < min_piece * 2:
            continue

        divider = max(3, int(round(0.04 * ppm)))
        half = divider // 2
        if horizontal:
            split_x = x0 + cut
            xa, xb = max(x0, split_x - half), min(x1, split_x + (divider - half))
            region = out[y0:y1, xa:xb]
            labels = lab[y0:y1, xa:xb]
            region[labels == component_id] = WALL
        else:
            split_y = y0 + cut
            ya, yb = max(y0, split_y - half), min(y1, split_y + (divider - half))
            region = out[ya:yb, x0:x1]
            labels = lab[ya:yb, x0:x1]
            region[labels == component_id] = WALL

    return out


def _normalise_doors(mask, ppm):
    """Make every door exactly 3.5 ft wide. Edits the mask so walls and doors stay consistent.
    First small undetected gaps between a door and the wall next to it are closed; then the door
    is widened (over wall / empty pixels) or narrowed (trimmed ends become wall), centred."""
    out = mask.copy()
    lab, n = ndi.label(mask == DOOR, structure=S8)
    for sl in ndi.find_objects(lab):
        if sl is None:
            continue
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        hz = (x1 - x0) >= (y1 - y0)
        v = out if hz else out.T                    # transposed view writes through to `out`
        a0, a1, c0, c1 = (x0, x1, y0, y1) if hz else (y0, y1, x0, x1)
        if (a1 - a0) / ppm < DOOR_NOISE:
            continue
        L = v.shape[1]
        band = v[c0:c1]

        reach = int(0.4 * ppm)
        for side in (-1, 1):                         # 1) undetected gap next to a wall
            n_gap = 0
            while n_gap < reach:
                i = (a0 - 1 - n_gap) if side < 0 else (a1 + n_gap)
                if i < 0 or i >= L or (band[:, i] == 0).mean() < 0.6:
                    break
                n_gap += 1
            j = (a0 - 1 - n_gap) if side < 0 else (a1 + n_gap)
            if n_gap and 0 <= j < L and (band[:, j] == WALL).mean() > 0.5:
                if side < 0:
                    band[:, a0 - n_gap:a0] = DOOR; a0 -= n_gap
                else:
                    band[:, a1:a1 + n_gap] = DOOR; a1 += n_gap

        tpx = max(3, int(round(DOOR_STD * ppm)))
        if abs((a1 - a0) - tpx) <= 1:
            continue
        mid = (a0 + a1) / 2.0
        n0 = int(round(mid - tpx / 2.0)); n1 = n0 + tpx
        n0, n1 = max(0, n0), min(L, n1)
        if n1 - n0 < a1 - a0:                        # too wide: the trimmed ends become wall
            band[:, a0:n0] = WALL
            band[:, n1:a1] = WALL
        else:                                        # too narrow: widen over wall / empty pixels
            for i in range(n0, n1):
                if i < a0 or i >= a1:
                    col = band[:, i]
                    band[:, i][(col == WALL) | (col == 0)] = DOOR
    return out

def _set_door_widths(walls, openings, tol=0.1):
    """Make every door exactly DOOR_STD (3.5 ft) wide, in the MODEL (not in the mask)."""
    for o in openings:
        if o["type"] != "door" or o["width"] < DOOR_NOISE:
            continue
        hz, c = o["horizontal"], o["centre"]
        mid = (o["start"] + o["end"]) / 2.0
        half = o["thickness"] / 2.0
        lo, hi = -1e9, 1e9
        # cross walls and other openings on the line limit how far the door can grow
        for w in walls:
            if w.get("diagonal") or w["horizontal"] == hz:
                continue
            p0, p1 = (w["y0"], w["y1"]) if hz else (w["x0"], w["x1"])
            a0, a1 = (w["x0"], w["x1"]) if hz else (w["y0"], w["y1"])
            if p0 - 0.02 <= c <= p1 + 0.02:
                if a1 <= mid:
                    lo = max(lo, a1)
                elif a0 >= mid:
                    hi = min(hi, a0)
        for q in openings:
            if q is o or q["horizontal"] != hz or abs(q["centre"] - c) > max(tol, half, q["thickness"] / 2):
                continue
            if q["end"] <= mid:
                lo = max(lo, q["end"])
            elif q["start"] >= mid:
                hi = min(hi, q["start"])
        lo = min(lo, o["start"]); hi = max(hi, o["end"])
        s, e = mid - DOOR_STD / 2, mid + DOOR_STD / 2

        # a door that sits against a cross wall stays against it (resizing around the centre would
        # leave a gap between the door frame and that wall)
        at_lo = lo > -1e8 and abs(o["start"] - lo) <= 0.1
        at_hi = hi < 1e8 and abs(o["end"] - hi) <= 0.1

        if at_lo and not at_hi:
            s, e = lo, lo + DOOR_STD
        elif at_hi and not at_lo:
            s, e = hi - DOOR_STD, hi
        if hi - lo < DOOR_STD:
            s, e = lo, hi
        else:
            if s < lo:
                s, e = lo, lo + DOOR_STD
            if e > hi:
                s, e = hi - DOOR_STD, hi
        o["start"], o["end"], o["width"] = float(s), float(e), float(e - s)

        # remove the wall that now lies under the door
        out = []
        for w in walls:
            if w.get("diagonal") or w["horizontal"] != hz:
                out.append(w)
                continue
            wc = (w["y0"] + w["y1"]) / 2 if hz else (w["x0"] + w["x1"]) / 2
            a0, a1 = (w["x0"], w["x1"]) if hz else (w["y0"], w["y1"])
            if abs(wc - c) > max(tol, half, w["thickness"] / 2) or a1 <= s + 0.005 or a0 >= e - 0.005:
                out.append(w)
                continue
            for pa, pb in ([(a0, s)] if s - a0 > 0.03 else []) + ([(e, a1)] if a1 - e > 0.03 else []):
                n = dict(w)
                if hz:
                    n["x0"], n["x1"], n["length"] = pa, pb, pb - pa
                else:
                    n["y0"], n["y1"], n["length"] = pa, pb, pb - pa
                out.append(n)
        walls[:] = out


def _snap_openings(walls, openings, reach=0.5, tol=0.1, thick_default=0.0):
    """Close the small gaps between a door/window and the wall pieces beside it: each neighbouring
    wall piece is extended to the opening's edge, and the opening takes the wall's centre line and
    thickness so the two line up exactly (this also lets groups.ts pair them)."""
    for o in openings:
        hz = o["horizontal"]
        best = {-1: None, 1: None}
        for w in walls:
            if w.get("diagonal") or w["horizontal"] != hz:
                continue
            centre = (w["y0"] + w["y1"]) / 2 if hz else (w["x0"] + w["x1"]) / 2
            if abs(centre - o["centre"]) > max(tol, o["thickness"] / 2, w["thickness"] / 2):
                continue
            a0, a1 = (w["x0"], w["x1"]) if hz else (w["y0"], w["y1"])
            if a1 <= o["start"] + tol and o["start"] - a1 <= reach:          # wall before
                if best[-1] is None or a1 > best[-1][0]:
                    best[-1] = (a1, w, centre)
            if a0 >= o["end"] - tol and a0 - o["end"] <= reach:              # wall after
                if best[1] is None or a0 < best[1][0]:
                    best[1] = (a0, w, centre)
        near = [b[1:] for b in best.values() if b]
        if not near and thick_default:      # a door with no wall beside it would be paper-thin
            o["thickness"] = max(o["thickness"], thick_default)
        if near:
            o["centre"] = float(np.mean([c for _, c in near]))
            o["thickness"] = float(max(w["thickness"] for w, _ in near))
        if best[-1]:
            w = best[-1][1]
            key = "x1" if hz else "y1"
            w[key] = o["start"]
            w["length"] = (w["x1"] - w["x0"]) if hz else (w["y1"] - w["y0"])
        if best[1]:
            w = best[1][1]
            key = "x0" if hz else "y0"
            w[key] = o["end"]
            w["length"] = (w["x1"] - w["x0"]) if hz else (w["y1"] - w["y0"])
            
        # No wall piece on this side of the opening, but a cross wall a little further on: fill the
        # gap with a short wall piece so the frame meets the cross wall.
        for side in (-1, 1):
            if best[side]:
                continue
            edge = o["start"] if side < 0 else o["end"]
            face = None
            for w in list(walls):
                if w.get("diagonal") or w["horizontal"] == hz:
                    continue
                p0, p1 = (w["y0"], w["y1"]) if hz else (w["x0"], w["x1"])        # across-range
                a0, a1 = (w["x0"], w["x1"]) if hz else (w["y0"], w["y1"])        # along-range
                if not (p0 - tol <= o["centre"] <= p1 + tol):
                    continue
                if side < 0 and a1 <= edge + 0.02 and edge - a1 <= reach:
                    face = a1 if face is None else max(face, a1)
                elif side > 0 and a0 >= edge - 0.02 and a0 - edge <= reach:
                    face = a0 if face is None else min(face, a0)
            if face is None or abs(face - edge) <= 0.015:
                continue
            lo_, hi_ = (face, edge) if side < 0 else (edge, face)
            th = o["thickness"]
            c = o["centre"]
            piece = {"id": len(walls), "horizontal": bool(hz), "length": hi_ - lo_, "thickness": th}
            if hz:
                piece.update(x0=lo_, x1=hi_, y0=c - th / 2, y1=c + th / 2)
            else:
                piece.update(y0=lo_, y1=hi_, x0=c - th / 2, x1=c + th / 2)
            walls.append(piece)


def mask_to_model(mask, ppm):
    """Cleaned label mask (work resolution) + px/m at that resolution -> PlanModel-style dict."""
    ppm = max(float(ppm), 1e-6)
    if NORMALISE_DOORS:
        if NORMALISE_DOORS:
            mask = _split_merged_doors(mask, ppm)
    wall_boxes, wall_diag = _fit_boxes(mask == WALL, max(4, int(ppm * .12)),
                                       max(6, int(ppm * .02) ** 2), allow_diagonal=True,
                                       ppm_hint=ppm)
    walls = _wall_dicts(wall_boxes, ppm, wall_diag)
    rail_boxes, _ = _fit_boxes(mask == RAILING, max(4, int(ppm * .10)), 6, ppm_hint=ppm)
    railings = _wall_dicts(rail_boxes, ppm)
        # Curved bay windows: window pixels that form an arc become curved walls with glass.
    # Everything else in the window mask stays an ordinary straight window.
    bays, bay_px = [], np.zeros(mask.shape, bool)
    try:
        from shapes import extra_walls
        win255 = ((mask == WINDOW) * 255).astype(np.uint8)
        bays = extra_walls(win255, [], ppm, first_id=0, window_mask=win255,
                           diagonal=False, consumed=bay_px)
    except Exception:
        bays = []                      # curved bays are optional; never break detection

    min_open = max(4, int(ppm * .25))
    openings = (_openings(mask == DOOR, "door", ppm, min_open)
                + _openings((mask == WINDOW) & ~bay_px, "window", ppm, min_open))
    if NORMALISE_DOORS:
        _set_door_widths(walls, openings)
    # specks left by jagged mask edges (much thinner than a wall and very short) are not walls
    th = sorted(w["thickness"] for w in walls if not w.get("diagonal") and w["length"] > 0.5)
    t_med = th[len(th) // 2] if th else 0.0
    if t_med:
        walls[:] = [w for w in walls if w.get("diagonal")
                    or not (w["thickness"] < 0.45 * t_med and w["length"] < 4 * t_med)]
    for i, w in enumerate(walls):
        w["id"] = i
    for i, o in enumerate(openings):
        o["id"] = i
    _snap_openings(walls, openings, thick_default=t_med)

    stairs = _stairs(mask == STAIRS, ppm, max(30, int((ppm * .3) ** 2)))

    # Floor slab / camera framing follow the building (walls + openings), not stray stairs/rails.
    xs = [v for w in walls for v in (w["x0"], w["x1"])]
    ys = [v for w in walls for v in (w["y0"], w["y1"])]
    if xs:
        bounds = {"x0": min(xs), "y0": min(ys), "x1": max(xs), "y1": max(ys)}
    else:
        h, w = mask.shape
        bounds = {"x0": 0., "y0": 0., "x1": w / ppm, "y1": h / ppm}
    return {"units": "m", "px_per_m": ppm, "bounds": bounds, "walls": walls, "openings": openings,
            "railings": railings, "stairs": stairs, "detector": "ml"}


def _rescue_walls(mask, gray, ppm):
    """Add wall pixels the model missed: thick, dark, ink strokes of the drawing that continue
    from a wall the model found. Thin lines (text, furniture, door arcs, dimension lines) and fat
    blobs (trees, solid fills) are ignored, and growth is limited to RESCUE_REACH_M."""
    pp = _load_postprocessor()
    walls = mask == WALL
    if not walls.any():
        return mask
    t = pp.estimate_thickness(walls)
    if t < 3:
        return mask
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = gray < min(float(otsu), 140.0)
    small = max(2, int(round(0.6 * t)))
    big = max(small + 2, int(round(2.2 * t)))
    thick = ndi.binary_opening(ink, structure=np.ones((small, small), bool))
    fat = ndi.binary_opening(ink, structure=np.ones((big, big), bool))
    fat = ndi.binary_dilation(fat, iterations=2)
    candidate = thick & ~fat & (mask == 0)

    seeds = ndi.binary_dilation(walls | (mask == DOOR) | (mask == WINDOW),
                                 iterations=max(3, int(round(0.4 * t))))
    reach = max(3, int(RESCUE_REACH_M * ppm))
    grown = ndi.binary_dilation(seeds, structure=ndi.generate_binary_structure(2, 1),
                                iterations=reach, mask=candidate | seeds)
    added = grown & candidate
    # restore the stroke's true width (the opening trimmed thin ends / corners)
    added = ndi.binary_dilation(added, iterations=1) & ink & (mask == 0)
    out = mask.copy()
    out[added] = WALL
    return out


def _debug_dump(mask):
    """Set env var ML_DEBUG_DIR to a folder to save the cleaned mask of every run."""
    import os, time
    folder = os.environ.get("ML_DEBUG_DIR")
    if not folder:
        return
    os.makedirs(folder, exist_ok=True)
    palette = np.array([[255, 255, 255], [220, 50, 50], [40, 180, 90], [50, 120, 230],
                        [255, 165, 0], [180, 70, 220]], dtype=np.uint8)
    cv2.imwrite(os.path.join(folder, f"mask_{int(time.time())}.png"),
                cv2.cvtColor(palette[mask], cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------- public entry
def analyse_ml(gray: np.ndarray, px_per_m: float):
    """Return (PlanModel-compatible dict in metres, BGR overlay) from the FINAL cleaned mask."""
    if gray.ndim != 2:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    mask, shrink = _predict(gray)
    if RESCUE_WALLS:
        work = gray if mask.shape == gray.shape else cv2.resize(
            gray, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA)
        mask = _rescue_walls(mask, work, float(px_per_m) * shrink)
    _debug_dump(mask)
    model = mask_to_model(mask, float(px_per_m) * shrink)
    model["px_per_m"] = float(px_per_m)

    h, w = gray.shape
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    colors = {1: (255, 150, 50), 2: (80, 200, 60), 3: (40, 170, 255), 4: (220, 80, 180), 5: (180, 60, 255)}
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for cls, color in colors.items():
        active = mask == cls
        overlay[active] = (0.45 * overlay[active] + 0.55 * np.array(color)).astype(np.uint8)
    return model, overlay