"""Find several floor plans drawn side by side (or one above the other) on one sheet.

    info = split_floors(model)        # model = the dict analyse()/analyse_ml() returns, in metres
    if info: model["floors"] = info

A sheet that shows "GROUND FLOOR PLAN" and "FIRST FLOOR PLAN" next to each other is detected as one
set of walls with a clear empty gap between two groups. Each group becomes a floor: its walls,
openings, stairs and railings are listed, together with the shift that lays it over the first one.
The viewer stacks the floors; the plan coordinates themselves are NOT changed, so every wall keeps
its id and its place on the sheet (selection, editing and snapping keep working).

Returned list (ground floor first, in the order they most likely are):
  [{"id": 0, "label": "Ground floor", "bbox": {x0,y0,x1,y1},
    "wallIds": [...], "openingIds": [...], "stairIds": [...], "railingIds": [...],
    "shiftTo": {"1": [sx, sz]}}, ...]
shiftTo[k] = metres to add to this floor's plan coordinates so that it lies over floor k.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
from scipy.signal import fftconvolve

LABELS = ["Ground floor", "First floor", "Second floor", "Third floor", "Fourth floor"]
MIN_SHARE = 0.12        # a floor must hold at least this share of the biggest plan's wall area
MAX_FLOORS = 5
RES = 0.05              # raster cell for alignment, metres
SEARCH = 1.0            # alignment may move a floor this far from "top-left corners together"


def _wall_box(w):
    return w["x0"], w["y0"], w["x1"], w["y1"]


def _dist_to_box(px, py, b):
    dx = max(b[0] - px, 0.0, px - b[2])
    dy = max(b[1] - py, 0.0, py - b[3])
    return float(np.hypot(dx, dy))


def _components(walls, gap):
    """Union-find over walls whose boxes, grown by `gap`, touch."""
    n = len(walls)
    b = np.array([_wall_box(w) for w in walls], dtype=float)
    b[:, 0] -= gap / 2; b[:, 1] -= gap / 2; b[:, 2] += gap / 2; b[:, 3] += gap / 2
    ov = ((b[:, None, 0] <= b[None, :, 2]) & (b[None, :, 0] <= b[:, None, 2]) &
          (b[:, None, 1] <= b[None, :, 3]) & (b[None, :, 1] <= b[:, None, 3]))
    # connected components of the overlap graph (BFS; n is a few hundred at most)
    seen = -np.ones(n, dtype=int)
    cur = 0
    for i in range(n):
        if seen[i] >= 0:
            continue
        stack = [i]
        seen[i] = cur
        while stack:
            a = stack.pop()
            for c in np.nonzero(ov[a] & (seen < 0))[0]:
                seen[c] = cur
                stack.append(int(c))
        cur += 1
    return seen, cur


def _raster(walls, box, pad=0.4):
    x0, y0 = box[0] - pad, box[1] - pad
    W = int(np.ceil((box[2] - box[0] + 2 * pad) / RES)) + 1
    H = int(np.ceil((box[3] - box[1] + 2 * pad) / RES)) + 1
    r = np.zeros((H, W), np.float32)
    for w in walls:
        xa = int(max(0, np.floor((w["x0"] - x0) / RES)))
        xb = int(min(W, np.ceil((w["x1"] - x0) / RES)))
        ya = int(max(0, np.floor((w["y0"] - y0) / RES)))
        yb = int(min(H, np.ceil((w["y1"] - y0) / RES)))
        if xb > xa and yb > ya:
            r[ya:yb, xa:xb] = 1.0
    r = ndi.binary_dilation(r > 0, iterations=max(1, int(round(0.10 / RES)))).astype(np.float32)
    return r, (x0, y0)


def _shift(base_walls, base_box, walls, box):
    """(sx, sz) to add to `walls` so that they lie best over `base_walls`.
    Start with the top-left corners together, then refine by wall overlap within SEARCH metres."""
    s0 = (base_box[0] - box[0], base_box[1] - box[1])
    try:
        ra, (ax, ay) = _raster(base_walls, base_box)
        rb, (bx, by) = _raster(walls, box)
        # correlation: c[dy, dx] = sum ra[y + dy, x + dx] * rb[y, x]  (rb moved by +dx, +dy over ra)
        c = fftconvolve(ra, rb[::-1, ::-1], mode="full")
        # lag (i, j) of the full output corresponds to moving rb by (j - (rb_w-1), i - (rb_h-1)) cells
        H, W = rb.shape
        ys, xs = np.mgrid[0:c.shape[0], 0:c.shape[1]]
        dx_m = (xs - (W - 1)) * RES + (ax - bx)
        dy_m = (ys - (H - 1)) * RES + (ay - by)
        near = (np.abs(dx_m - s0[0]) <= SEARCH) & (np.abs(dy_m - s0[1]) <= SEARCH)
        if not near.any():
            return s0
        score = np.where(near, c, -1.0)
        # a tiny pull towards the corner alignment so that ties keep the simple answer
        score = score - 0.02 * np.hypot(dx_m - s0[0], dy_m - s0[1]) * RES
        i, j = np.unravel_index(int(np.argmax(score)), score.shape)
        # the corner alignment itself: nearest lag to s0
        j0 = int(round((s0[0] - (ax - bx)) / RES + (W - 1)))
        i0 = int(round((s0[1] - (ay - by)) / RES + (H - 1)))
        base = float(c[i0, j0]) if 0 <= i0 < c.shape[0] and 0 <= j0 < c.shape[1] else 0.0
        # only move away from "corners together" when the walls clearly fit better there
        if c[i, j] < 1.25 * base + 0.02 * min(ra.sum(), rb.sum()):
            return s0
        return float(dx_m[i, j]), float(dy_m[i, j])
    except Exception:
        return s0



# ------------------------------------------------------------------------------------------
# Finding the plans on a sheet.
#
# A plan is a set of walls that are all connected in projection: an empty vertical (or horizontal)
# band that no wall of the drawing crosses separates two plans, while a courtyard or an L-shape is
# always crossed by some wall. Thin strokes (dimension lines, borders, titles) are left out of the
# search, tiny groups (north arrow, a lone stair) are treated as debris, and each cut must leave
# two real plans.
# ------------------------------------------------------------------------------------------
CELL = 0.1              # metres per cell of the occupancy grid
MIN_GAP_M = 1.2         # narrowest empty band that separates two plans
MIN_GAP_FRAC = 0.04     # ... and at least this share of the region's length
MIN_SIDE_M = 2.5        # a plan is at least this big both ways
DEBRIS_SHARE = 0.12     # a group with less wall than this share of the sheet is not a plan


def _grid(model):
    walls = [w for w in model.get("walls", []) if "x0" in w]
    if not walls:
        return None
    th = sorted(w["thickness"] for w in walls if w.get("length", 0) > 0.5)
    med = th[len(th) // 2] if th else 0.15
    solid = [w for w in walls if w["thickness"] >= 0.55 * med]          # drop hairlines
    if len(solid) < 6:
        solid = walls
    x0 = min(w["x0"] for w in solid) - 0.3
    y0 = min(w["y0"] for w in solid) - 0.3
    x1 = max(w["x1"] for w in solid) + 0.3
    y1 = max(w["y1"] for w in solid) + 0.3
    W = int(np.ceil((x1 - x0) / CELL)) + 1
    H = int(np.ceil((y1 - y0) / CELL)) + 1
    occ = np.zeros((H, W), bool)
    for w in solid:
        a = max(0, int(np.floor((w["x0"] - x0) / CELL))); b = min(W, int(np.ceil((w["x1"] - x0) / CELL)) + 1)
        c = max(0, int(np.floor((w["y0"] - y0) / CELL))); d = min(H, int(np.ceil((w["y1"] - y0) / CELL)) + 1)
        occ[c:d, a:b] = True
    return occ, x0, y0


def _best_gap(occ, reg, min_gap_m, min_frac):
    """Widest empty band inside region (r0, r1, c0, c1) that has wall on both sides.
    Returns (width_m, axis, cut_cell, (r0, r1, c0, c1) of side A, ... of side B) or None."""
    r0, r1, c0, c1 = reg
    sub = occ[r0:r1, c0:c1]
    best = None
    for axis in (0, 1):
        prof = sub.any(axis=0) if axis == 0 else sub.any(axis=1)      # axis 0: columns (x), 1: rows (y)
        idx = np.nonzero(prof)[0]
        if len(idx) < 2:
            continue
        lo, hi = idx[0], idx[-1]
        span = (hi - lo + 1) * CELL
        need = max(min_gap_m, min_frac * span)
        i = lo
        while i <= hi:
            if not prof[i]:
                j = i
                while j + 1 <= hi and not prof[j + 1]:
                    j += 1
                width = (j - i + 1) * CELL
                if width >= need and (best is None or width > best[0]):
                    cut = (i + j) // 2
                    if axis == 0:
                        A = (r0, r1, c0 + lo, c0 + cut); B = (r0, r1, c0 + cut + 1, c0 + hi + 1)
                    else:
                        A = (r0 + lo, r0 + cut, c0, c1); B = (r0 + cut + 1, r0 + hi + 1, c0, c1)
                    best = (width, axis, cut, A, B)
                i = j + 1
            else:
                i += 1
    return best


def _tight(occ, reg):
    r0, r1, c0, c1 = reg
    sub = occ[r0:r1, c0:c1]
    rows = np.nonzero(sub.any(axis=1))[0]; cols = np.nonzero(sub.any(axis=0))[0]
    if not len(rows) or not len(cols):
        return None
    return (r0 + rows[0], r0 + rows[-1] + 1, c0 + cols[0], c0 + cols[-1] + 1)


def _regions(model, count=0):
    g = _grid(model)
    if g is None:
        return None
    occ, ox, oy = g
    total = int(occ.sum())
    regs = [(0, occ.shape[0], 0, occ.shape[1])]
    forced = count >= 2
    while len(regs) < (count if forced else MAX_FLOORS):
        cands = []
        for k, r in enumerate(regs):
            b = _best_gap(occ, r, 0.4 if forced else MIN_GAP_M, 0.0 if forced else MIN_GAP_FRAC)
            if b:
                cands.append((b[0], k, b))
        if not cands:
            break
        _, k, (width, axis, cut, A, B) = max(cands, key=lambda t: t[0])
        a = int(occ[A[0]:A[1], A[2]:A[3]].sum()); b = int(occ[B[0]:B[1], B[2]:B[3]].sum())
        if not forced:
            small = lambda n, reg: n < DEBRIS_SHARE * total or min(reg[1] - reg[0], reg[3] - reg[2]) * CELL < MIN_SIDE_M
            if small(a, A) and small(b, B):
                break
            if small(a, A) or small(b, B):                      # debris: keep the big side, drop the cut
                keep = B if small(a, A) else A
                regs[k] = _tight(occ, keep) or regs[k]
                continue
        regs[k:k + 1] = [t for t in (_tight(occ, A), _tight(occ, B)) if t]
    if len(regs) < 2:
        return None
    boxes = [(ox + c0 * CELL, oy + r0 * CELL, ox + c1 * CELL, oy + r1 * CELL) for r0, r1, c0, c1 in regs]
    return boxes


def _order(boxes):
    """ground floor first: left to right when the plans sit side by side, top to bottom when stacked"""
    cx = [(b[0] + b[2]) / 2 for b in boxes]; cy = [(b[1] + b[3]) / 2 for b in boxes]
    key = (lambda i: cx[i]) if (max(cx) - min(cx)) >= (max(cy) - min(cy)) else (lambda i: cy[i])
    return sorted(range(len(boxes)), key=key)


def _assign(model, boxes):
    """Give every wall, opening, stair and railing to the box that holds its centre (else the nearest)."""
    def pick(x, y):
        best, bd = 0, 1e18
        for k, (a, b, c, d) in enumerate(boxes):
            dx = max(a - x, 0, x - c); dy = max(b - y, 0, y - d)
            dd = dx * dx + dy * dy
            if dd < bd:
                best, bd = k, dd
        return best
    mem = [{"wallIds": [], "openingIds": [], "stairIds": [], "railingIds": []} for _ in boxes]
    for w in model.get("walls", []):
        if "x0" in w:
            mem[pick((w["x0"] + w["x1"]) / 2, (w["y0"] + w["y1"]) / 2)]["wallIds"].append(w["id"])
    for o in model.get("openings", []) or []:
        a, c = (o["start"] + o["end"]) / 2, o["centre"]
        x, y = (a, c) if o["horizontal"] else (c, a)
        mem[pick(x, y)]["openingIds"].append(o["id"])
    for key, src in (("stairIds", "stairs"), ("railingIds", "railings")):
        for o in model.get(src, []) or []:
            mem[pick((o["x0"] + o["x1"]) / 2, (o["y0"] + o["y1"]) / 2)][key].append(o["id"])
    return mem


def _finish(model, boxes, source, labels=None):
    boxes = [tuple(map(float, b)) for b in boxes]
    order = _order(boxes) if source == "auto" else list(range(len(boxes)))
    boxes = [boxes[i] for i in order]
    mem = _assign(model, boxes)
    by_id = {w["id"]: w for w in model.get("walls", []) if "x0" in w}
    th = sorted(w["thickness"] for w in by_id.values() if w.get("length", 0) > 0.5)
    med = th[len(th) // 2] if th else 0.15
    infos = []
    for k, m in enumerate(mem):
        ws = [by_id[i] for i in m["wallIds"] if i in by_id]
        if not ws:
            continue
        core = [w for w in ws if w["thickness"] >= 0.55 * med] or ws      # hairlines do not size a plan
        bb = {"x0": min(w["x0"] for w in core), "y0": min(w["y0"] for w in core),
              "x1": max(w["x1"] for w in core), "y1": max(w["y1"] for w in core)}
        infos.append({"bbox": bb, "walls": core, **m})
    if len(infos) < 2:
        return None
    out = []
    for k, i in enumerate(infos):
        label = (labels[k] if labels and k < len(labels) and labels[k] else None) or (LABELS[k] if k < len(LABELS) else f"Floor {k + 1}")
        out.append({"id": k, "label": label, "bbox": i["bbox"], "wallIds": i["wallIds"], "openingIds": i["openingIds"],
                    "stairIds": i["stairIds"], "railingIds": i["railingIds"],
                    "similar": True, "source": source, "shiftTo": {}})
    areas = [(i["bbox"]["x1"] - i["bbox"]["x0"]) * (i["bbox"]["y1"] - i["bbox"]["y0"]) for i in infos]
    if source == "auto":
        for o in out:
            o["similar"] = bool(min(areas) / max(areas) >= 0.30)
    for a, ia in enumerate(infos):
        for b, ib in enumerate(infos):
            if a == b:
                continue
            bbx = (ib["bbox"]["x0"], ib["bbox"]["y0"], ib["bbox"]["x1"], ib["bbox"]["y1"])
            abx = (ia["bbox"]["x0"], ia["bbox"]["y0"], ia["bbox"]["x1"], ia["bbox"]["y1"])
            sx, sz = _shift(ib["walls"], bbx, ia["walls"], abx)
            out[a]["shiftTo"][str(b)] = [round(sx, 3), round(sz, 3)]
    return out


def split_floors(model, count=0):
    """Automatic: count = 0 decides by the empty bands between plans, count >= 2 forces that many plans
    (the widest bands are used), count = 1 means one plan only."""
    if count == 1 or len([w for w in model.get("walls", []) if "x0" in w]) < 8:
        return None
    boxes = _regions(model, count)
    return _finish(model, boxes, "auto") if boxes else None


def floors_from_boxes(model, boxes, labels=None):
    """Manual: boxes = [[x0, y0, x1, y1], ...] in plan metres, ground floor first, as drawn by the user."""
    if len(boxes) < 2:
        return None
    return _finish(model, [(b[0], b[1], b[2], b[3]) for b in boxes], "manual", labels)