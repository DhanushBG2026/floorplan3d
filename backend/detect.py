"""Deterministic floor-plan analysis (OpenCV only, no ML).

Assumptions (MVP): residential plan, solid or hollow (two-line) walls, axis-aligned.
Output units are metres; x = right, y = down in the source image.
"""
import base64
import cv2
import numpy as np


def _binary_ink(gray):
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return ink


def _wall_mask(ink, ppm):
    """Keep only 'thick' strokes: removes text, dimension lines, door arcs, hatching."""
    k = max(3, int(round(0.04 * ppm)))
    return cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))


def _outline_bands(ink, ppm, horizontal):
    """Walls drawn as two thin parallel lines (the wall faces) with an empty middle.
    Closing across the wall joins the two lines into one band; bands of plausible wall thickness
    are kept. Done at several scales so one nearby object cannot swallow a wall. Runs that contain
    3+ separate lines are window glazing, not wall, and are removed."""
    tmin = max(3, int(round(0.04 * ppm)))
    out = np.zeros_like(ink)
    for g_m in (0.12, 0.25, 0.45):
        g = max(tmin + 2, int(round(g_m * ppm)))
        gk = (g + 1) | 1                      # odd size keeps the closing symmetric (no 1px shift)
        k_close = (1, gk) if horizontal else (gk, 1)
        closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, k_close))
        k_min = (1, tmin) if horizontal else (tmin, 1)
        k_max = (1, g + 4) if horizontal else (g + 4, 1)
        ge_min = cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k_min))
        ge_max = cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k_max))
        out = cv2.bitwise_or(out, cv2.bitwise_and(ge_min, cv2.bitwise_not(ge_max)))
    return _drop_glazing_runs(out, ink, horizontal)


def _drop_glazing_runs(band, ink, horizontal):
    """Remove every run (across the wall) of `band` that contains 3 or more separate ink lines."""
    b = band > 0
    i = (ink > 0) & b
    if not horizontal:
        b, i = b.T, i.T                       # now runs go along axis 0 (down a column)
    prev_b = np.zeros_like(b); prev_b[1:] = b[:-1]
    prev_i = np.zeros_like(i); prev_i[1:] = i[:-1]
    bf = b.ravel(order="F")                   # column-major: each run is contiguous
    run_id = np.cumsum((b & ~prev_b).ravel(order="F")) * bf
    ink_start = (i & ~prev_i).ravel(order="F")
    counts = np.bincount(run_id[ink_start], minlength=int(run_id.max()) + 1)
    drop = (counts[run_id] >= 3) & bf
    keep = (bf & ~drop).reshape(b.shape, order="F")
    if not horizontal:
        keep = keep.T
    return (keep * 255).astype(np.uint8)


def _opening_boxes(openings, ppm, shape, margin=4):
    """Paint the area of already-found openings (door/window gaps) as a mask."""
    m = np.zeros(shape, np.uint8)
    for o in openings:
        t = o["thickness"] * ppm / 2 + margin
        a0, a1 = o["start"] * ppm - margin, o["end"] * ppm + margin
        c = o["centre"] * ppm
        if o["horizontal"]:
            x0, x1, y0, y1 = a0, a1, c - t, c + t
        else:
            x0, x1, y0, y1 = c - t, c + t, a0, a1
        m[max(0, int(y0)):int(y1) + 1, max(0, int(x0)):int(x1) + 1] = 255
    return m


def _drop_isolated(h_rects, v_rects, ppm):
    """Group wall pieces that touch, or that line up across a door/window gap (up to ~2 m), and
    drop tiny groups: leftovers such as dimension-line boxes or fixtures are not walls."""
    items = [(r, True) for r in h_rects] + [(r, False) for r in v_rects]
    n = len(items)
    across, along = 0.15 * ppm, 1.0 * ppm
    boxes = []
    for r, hz in items:
        if hz:
            boxes.append((r[0] - along, r[1] - across, r[2] + along, r[3] + across))
        else:
            boxes.append((r[0] - across, r[1] - along, r[2] + across, r[3] + along))
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        a = boxes[i]
        for j in range(i + 1, n):
            b = boxes[j]
            if a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]:
                parent[find(i)] = find(j)
    total = {}
    for i, (r, hz) in enumerate(items):
        length = (r[2] - r[0]) if hz else (r[3] - r[1])
        total[find(i)] = total.get(find(i), 0) + length
    if not total:
        return h_rects, v_rects
    need = max(1.5 * ppm, 0.1 * max(total.values()))
    keep = [i for i in range(n) if total[find(i)] >= need]
    return ([items[i][0] for i in keep if items[i][1]],
            [items[i][0] for i in keep if not items[i][1]])


def _runs(mask, horizontal, min_len_px):
    shape = (min_len_px, 1) if horizontal else (1, min_len_px)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, shape))


def _components(mask, horizontal, min_len_px):
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, _a = stats[i]
        length, thick = (w, h) if horizontal else (h, w)
        if length >= min_len_px and thick >= 3:
            out.append((int(x), int(y), int(x + w), int(y + h)))
    return out


def _trim_vertical(v_rects, h_rects):
    """Butt vertical walls against horizontal ones so corners don't overlap."""
    out = []
    for x0, y0, x1, y1 in v_rects:
        for hx0, hy0, hx1, hy1 in h_rects:
            ox = min(x1, hx1) - max(x0, hx0)
            if ox < 0.5 * (x1 - x0):
                continue
            if hy0 - 4 <= y0 <= hy1 + 4:
                y0 = hy1
            if hy0 - 4 <= y1 <= hy1 + 4:
                y1 = hy0
        if y1 - y0 > 0:
            out.append((x0, y0, x1, y1))
    return out


def _arc_score(dist, horizontal, s, e, c, tol_px):
    """Look for a door-swing arc next to the gap [s, e] (along-axis pixels, wall centre-line c).
    A swing is a quarter circle centred on the hinge, radius = leaf length. The hinge can be at
    either end of the gap or in the middle (two doors sharing a gap). Sample points along every
    candidate quarter circle and count how many sit on thin ink (dist = distance to the nearest
    thin-line pixel).
    Returns (arc score 0..1, clutter 0..1) for the best candidate, where clutter is the fraction of
    the swept area that holds ink. A real door swing is empty inside; plants / furniture are not."""
    H, W = dist.shape
    gap = e - s
    thetas = np.deg2rad(np.linspace(12, 78, 28))
    best, best_cand = 0.0, None
    for h in (s, (s + e) / 2, e):                 # hinge position along the wall
        for d in (1, -1):                         # leaf direction along the wall when closed
            for frac in np.arange(0.30, 1.11, 0.02):  # leaf length relative to the gap
                L = gap * float(frac)
                far = h + d * L
                if far < s - 0.1 * gap or far > e + 0.1 * gap:
                    continue                      # closed leaf must stay inside the gap
                for side in (-1, 1):              # swing side
                    along = h + d * L * np.cos(thetas)
                    perp = c + side * L * np.sin(thetas)
                    xs, ys = (along, perp) if horizontal else (perp, along)
                    xi, yi = np.round(xs).astype(int), np.round(ys).astype(int)
                    ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
                    hit = np.zeros(len(thetas), bool)
                    hit[ok] = dist[yi[ok], xi[ok]] <= tol_px
                    sc = float(hit.mean())
                    if sc > best:
                        best, best_cand = sc, (h, d, L, side)
    clutter = 0.0
    if best_cand:
        h, d, L, side = best_cand
        rr, tt = np.meshgrid(np.linspace(0.25, 0.8, 8) * L, np.deg2rad(np.linspace(12, 78, 14)))
        along = h + d * rr * np.cos(tt)
        perp = c + side * rr * np.sin(tt)
        xs, ys = (along, perp) if horizontal else (perp, along)
        xi, yi = np.round(xs).astype(int), np.round(ys).astype(int)
        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        v = np.ones(xi.shape, bool)
        v[ok] = dist[yi[ok], xi[ok]] <= 1.5
        clutter = float(v.mean())
    return best, clutter


def _glazing_score(band, horizontal):
    """Windows draw 2-3 parallel thin lines running the full length of the gap.
    Returns the fraction of positions along the gap that cross >= 2 separate ink lines."""
    b = (band > 0)
    if b.size == 0:
        return 0.0
    if not horizontal:
        b = b.T                                   # rows = across the wall, cols = along it
    runs = (b[1:, :] & ~b[:-1, :]).sum(axis=0) + b[0, :]
    return float((runs >= 2).mean())


def _classify(horizontal, s, e, c, t, ink, dist, ppm, need_evidence):
    """Decide what a gap [s, e] in a wall (centre-line c, thickness t, pixels) is.
    Returns (kind, density, arc, glaze) or None when need_evidence and nothing convincing is seen."""
    gap_m = (e - s) / ppm
    m = int(0.1 * (e - s))  # ignore the very ends
    if horizontal:
        band = ink[c - t // 2: c + t // 2, s + m: e - m]
    else:
        band = ink[s + m: e - m, c - t // 2: c + t // 2]
    density = float(band.mean() / 255) if band.size else 0.0
    glaze = _glazing_score(band, horizontal)
    arc, clutter = _arc_score(dist, horizontal, s, e, c, max(2.0, 0.025 * ppm))
    if glaze >= 0.8:
        kind = "window"                     # full-length parallel glazing lines -> window
    elif need_evidence:
        # gap ends at a perpendicular wall: accept only a clean swing arc (empty swept area)
        if arc >= 0.7 and clutter <= 0.2:
            kind = "door"
        else:
            return None
    elif arc >= 0.6 and clutter <= 0.3 and (gap_m <= 1.5 or arc >= 0.85):
        kind = "door"                       # swing arc present -> door
    elif density > 0.10:
        kind = "window"
    elif gap_m > 1.5:
        kind = "opening"                    # wide cased opening / archway
    else:
        kind = "door"                       # empty doorway
    return kind, density, arc, glaze


def _find_openings(rects, perp_rects, horizontal, ink, dist, ppm, min_gap_m=0.45, max_gap_m=3.0):
    """Openings are gaps in a wall line. Two kinds of gap are searched:
    (a) between two collinear wall pieces, (b) between the end of a wall piece and the face of a
    perpendicular wall (e.g. a door at the end of a partition). (b) is only accepted when a swing
    arc or glazing lines are seen, otherwise every free-standing partition end would look like one."""
    # work in (along, perp) space so horizontal and vertical walls share one code path
    to_ax = (lambda r: r) if horizontal else (lambda r: (r[1], r[0], r[3], r[2]))
    R = [to_ax(r) for r in rects]          # (a0, p0, a1, p1)
    P = [to_ax(r) for r in perp_rects]
    min_px, max_px = min_gap_m * ppm, max_gap_m * ppm

    groups = []
    for r in sorted(R, key=lambda r: r[1] + r[3]):
        c = (r[1] + r[3]) / 2
        t = r[3] - r[1]
        for g in groups:
            if abs(g["c"] - c) <= max(3, 0.5 * min(g["t"], t)):
                g["rects"].append(r)
                break
        else:
            groups.append({"c": c, "t": t, "rects": [r]})

    openings = []
    for g in groups:
        rs = sorted(g["rects"], key=lambda r: r[0])
        c, t = int(g["c"]), int(max(g["t"], 3))
        cands = []                                           # (s, e, need_evidence)
        for a, b in zip(rs, rs[1:]):                         # (a) collinear neighbours
            if min_px <= b[0] - a[2] <= max_px:
                cands.append((a[2], b[0], False))
        for i, piece in enumerate(rs):                       # (b) piece end -> perpendicular wall
            for direction in (1, -1):
                end = piece[2] if direction == 1 else piece[0]
                nbr = rs[i + 1][0] - end if direction == 1 and i + 1 < len(rs) else \
                      end - rs[i - 1][2] if direction == -1 and i > 0 else 1e9
                best = None
                for q in P:
                    if not (q[1] - 3 <= c <= q[3] + 3):      # blocker must cross this wall line
                        continue
                    face = q[0] if direction == 1 else q[2]
                    d = (face - end) * direction
                    if d >= -3 and (best is None or d < best[0]):
                        best = (d, face)
                if best and best[0] < nbr and min_px <= best[0] <= max_px:
                    s_, e_ = sorted((end, best[1]))
                    cands.append((int(s_), int(e_), True))
        seen = set()
        for s_, e_, need in cands:
            s_, e_ = int(s_), int(e_)
            if (s_, e_) in seen:
                continue
            seen.add((s_, e_))
            res = _classify(horizontal, s_, e_, c, t, ink, dist, ppm, need)
            if res is None:
                continue
            kind, density, arc, glaze = res
            openings.append({
                "type": kind,
                "horizontal": horizontal,
                "start": s_ / ppm, "end": e_ / ppm,      # along-axis range (m)
                "centre": c / ppm,                        # perpendicular coordinate (m)
                "thickness": g["t"] / ppm,
                "width": (e_ - s_) / ppm,
                "ink_density": round(density, 3),
                "arc_score": round(arc, 2),
                "glazing": round(glaze, 2),
            })
    # a gap that contains other accepted gaps on the same line is a duplicate of them
    keep = []
    for o in openings:
        inner = [q for q in openings if q is not o and q["horizontal"] == o["horizontal"]
                 and abs(q["centre"] - o["centre"]) < 1e-6
                 and q["start"] >= o["start"] - 1e-6 and q["end"] <= o["end"] + 1e-6]
        if len(inner) < 1:
            keep.append(o)
    return keep


def analyse(gray, ppm):
    """gray: uint8 image, ppm: pixels per metre. Returns (model dict, overlay BGR image)."""
    ink = _binary_ink(gray)
    thick = _wall_mask(ink, ppm)
    # thin strokes only (door arcs, glazing lines, furniture): ink minus (dilated) thick walls
    k = max(3, int(0.03 * ppm)) | 1
    thin = cv2.bitwise_and(ink, cv2.bitwise_not(cv2.dilate(thick, np.ones((k, k), np.uint8))))
    dist = cv2.distanceTransform(cv2.bitwise_not(thin), cv2.DIST_L2, 3)
    min_len = int(0.5 * ppm)
    # stage 1: solid (filled) walls
    h_s = _runs(thick, True, min_len)
    v_s = _runs(thick, False, min_len)
    h_rects = _components(h_s, True, min_len)
    v_rects = _trim_vertical(_components(v_s, False, min_len), h_rects)
    # stage 2: hollow walls (two thin lines). Not inside gaps between solid wall pieces, because
    # those gaps are windows/doors whose frame lines would otherwise look like a hollow wall.
    solid_gaps = (_find_openings(h_rects, v_rects, True, ink, dist, ppm)
                  + _find_openings(v_rects, h_rects, False, ink, dist, ppm))
    gaps_out = cv2.bitwise_not(_opening_boxes(solid_gaps, ppm, ink.shape))
    # a hollow band that sits right next to a solid wall is usually that wall merged with a nearby
    # line (grid line, dimension, fixture): leave such walls to the solid detector, which measures
    # their thickness correctly
    d = max(3, int(round(0.25 * ppm)))
    near_h = cv2.dilate(thick, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2 * d + 1)))
    near_v = cv2.dilate(thick, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * d + 1, 1)))
    o_h = cv2.bitwise_and(_outline_bands(ink, ppm, True), cv2.bitwise_and(gaps_out, cv2.bitwise_not(near_h)))
    o_v = cv2.bitwise_and(_outline_bands(ink, ppm, False), cv2.bitwise_and(gaps_out, cv2.bitwise_not(near_v)))
    hmask = _runs(cv2.bitwise_or(h_s, o_h), True, min_len)
    vmask = _runs(cv2.bitwise_or(v_s, o_v), False, min_len)
    h_rects = _components(hmask, True, min_len)
    v_rects = _trim_vertical(_components(vmask, False, min_len), h_rects)
    # drop pieces much thinner than the plan's typical wall (furniture / fixtures)
    th = [(r[3] - r[1]) for r in h_rects if (r[2] - r[0]) >= 1.5 * ppm] + \
         [(r[2] - r[0]) for r in v_rects if (r[3] - r[1]) >= 1.5 * ppm]
    if th:
        tmin_keep = 0.4 * float(np.median(th))
        h_rects = [r for r in h_rects if (r[3] - r[1]) >= tmin_keep]
        v_rects = [r for r in v_rects if (r[2] - r[0]) >= tmin_keep]
    h_rects, v_rects = _drop_isolated(h_rects, v_rects, ppm)

    walls = []
    for horizontal, rects in ((True, h_rects), (False, v_rects)):
        for x0, y0, x1, y1 in rects:
            walls.append({
                "id": len(walls),
                "x0": x0 / ppm, "y0": y0 / ppm, "x1": x1 / ppm, "y1": y1 / ppm,
                "horizontal": horizontal,
                "length": ((x1 - x0) if horizontal else (y1 - y0)) / ppm,
                "thickness": ((y1 - y0) if horizontal else (x1 - x0)) / ppm,
            })
    
    openings = _find_openings(h_rects, v_rects, True, ink, dist, ppm) + _find_openings(v_rects, h_rects, False, ink, dist, ppm)
    for i, o in enumerate(openings):
        o["id"] = i

    # overall bounds
    if walls:
        bx0 = min(w["x0"] for w in walls); by0 = min(w["y0"] for w in walls)
        bx1 = max(w["x1"] for w in walls); by1 = max(w["y1"] for w in walls)
    else:
        bx0 = by0 = bx1 = by1 = 0.0
    model = {
        "units": "m",
        "px_per_m": ppm,
        "bounds": {"x0": bx0, "y0": by0, "x1": bx1, "y1": by1},
        "walls": walls,
        "openings": openings,
    }

    # debug overlay
    ov = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    ov = cv2.addWeighted(ov, 0.5, np.full_like(ov, 255), 0.5, 0)
    for w in walls:
        cv2.rectangle(ov, (int(w["x0"] * ppm), int(w["y0"] * ppm)), (int(w["x1"] * ppm), int(w["y1"] * ppm)), (200, 120, 30), 2)
    col = {"door": (60, 170, 60), "window": (40, 140, 240), "opening": (160, 60, 160)}
    for o in openings:
        t = o["thickness"] / 2
        if o["horizontal"]:
            p0, p1 = (o["start"], o["centre"] - t), (o["end"], o["centre"] + t)
        else:
            p0, p1 = (o["centre"] - t, o["start"]), (o["centre"] + t, o["end"])
        cv2.rectangle(ov, (int(p0[0] * ppm), int(p0[1] * ppm)), (int(p1[0] * ppm), int(p1[1] * ppm)), col[o["type"]], -1)
    return model, ov


def overlay_b64(ov, max_side=1400):
    h, w = ov.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        ov = cv2.resize(ov, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", ov)
    return "data:image/png;base64," + base64.b64encode(buf).decode()