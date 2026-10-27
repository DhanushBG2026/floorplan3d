"""Diagonal and curved walls: the shapes that axis-aligned rectangles cannot describe.

Call after the straight walls are found. It looks at the wall pixels the rectangles do not cover and
turns each leftover blob into either a diagonal wall (straight, tilted) or a curved wall (an arc),
e.g. a bay window. Results use the same units as the other walls (metres, x right, y down)."""
import cv2
import numpy as np


def _circle_fit(pts):
    x, y = pts[:, 0], pts[:, 1]
    a = np.c_[2 * x, 2 * y, np.ones(len(x))]
    c, *_ = np.linalg.lstsq(a, x * x + y * y, rcond=None)
    cx, cy = c[0], c[1]
    return cx, cy, float(np.sqrt(max(c[2] + cx * cx + cy * cy, 0.0)))


def extra_walls(wall_mask, rects, ppm, first_id, window_mask=None, diagonal=True, consumed=None):
    """wall_mask: uint8 (0/255) wall pixels. rects: [(x0, y0, x1, y1)] pixel boxes already used as
    straight walls. window_mask: optional ML window pixels, used to flag curved bay windows.diagonal=False reports only curved shapes. consumed: optional bool array, set True on the pixels
    of every shape returned (so the caller can remove them from its own masks)."""
    covered = np.zeros_like(wall_mask)
    for x0, y0, x1, y1 in rects:
        cv2.rectangle(covered, (int(x0), int(y0)), (int(x1), int(y1)), 255, -1)
    pad = max(2, int(0.03 * ppm))
    covered = cv2.dilate(covered, np.ones((2 * pad + 1, 2 * pad + 1), np.uint8))
    left = cv2.bitwise_and(wall_mask, cv2.bitwise_not(covered))
    k = max(3, int(0.04 * ppm))
    left = cv2.morphologyEx(left, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))

    n, lab, stats, _ = cv2.connectedComponentsWithStats(left, connectivity=8)
    out, wid = [], first_id
    for i in range(1, n):
        ys, xs = np.where(lab == i)
        area = len(xs)
        pts = np.c_[xs, ys].astype(np.float64)
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(pts.astype(np.float32))
        length, thick = max(rw, rh), min(rw, rh)
        if length < 0.6 * ppm:
            continue

        # 1) straight but tilted
        deg = ang if rw >= rh else ang + 90
        off_axis = abs(((deg + 45) % 90) - 45)            # 0 = along an axis, 45 = diagonal
        if 0.04 * ppm <= thick <= 0.5 * ppm and area / (length * thick) >= 0.75 and off_axis >= 7:
            if not diagonal:
                continue
            if consumed is not None:
                consumed[ys, xs] = True
            dx, dy = np.cos(np.radians(deg)), np.sin(np.radians(deg))
            ax, ay = cx - dx * length / 2, cy - dy * length / 2
            bx, by = cx + dx * length / 2, cy + dy * length / 2
            out.append({
                "id": wid, "horizontal": False, "diagonal": True,
                "ax": ax / ppm, "ay": ay / ppm, "bx": bx / ppm, "by": by / ppm,
                "x0": min(ax, bx) / ppm, "x1": max(ax, bx) / ppm, "y0": min(ay, by) / ppm, "y1": max(ay, by) / ppm,
                "length": length / ppm, "thickness": thick / ppm,
            })
            wid += 1
            continue

        # 2) curved: fit a circle and read off the arc
        if len(pts) < 30:
            continue
        ccx, ccy, r = _circle_fit(pts)
        if not (0.3 * ppm <= r <= 25 * ppm):
            continue
        ang_all = np.arctan2(pts[:, 1] - ccy, pts[:, 0] - ccx)
        order = np.sort(ang_all)
        gaps = np.diff(np.r_[order, order[0] + 2 * np.pi])
        j = int(np.argmax(gaps))
        span = 2 * np.pi - gaps[j]
        a0 = float(order[(j + 1) % len(order)])
        if not (np.radians(20) <= span <= np.radians(340)):
            continue
        t_est = area / (r * span)
        resid = float(np.std(np.hypot(pts[:, 0] - ccx, pts[:, 1] - ccy) - r))
        if not (0.04 * ppm <= t_est <= 0.5 * ppm) or resid > max(1.5, 0.45 * t_est):
            continue
        a1 = a0 + float(span)
        if consumed is not None:
            consumed[ys, xs] = True
        bay = False
        if window_mask is not None:
            bay = float((window_mask[ys, xs] > 0).mean()) >= 0.25
        out.append({
            "id": wid, "horizontal": False, "diagonal": False, "bay": bay,
            "arc": {"cx": ccx / ppm, "cy": ccy / ppm, "r": r / ppm, "a0": a0, "a1": a1},
            "x0": float(xs.min()) / ppm, "x1": float(xs.max()) / ppm,
            "y0": float(ys.min()) / ppm, "y1": float(ys.max()) / ppm,
            "length": r * span / ppm, "thickness": t_est / ppm,
        })
        wid += 1
    return out