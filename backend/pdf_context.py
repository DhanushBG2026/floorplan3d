"""Read design information from a multi-page architectural PDF (floor plan + elevations).

    ctx = analyse_pdf(pdf_bytes)

Everything is taken from the PDF's own text and vector drawing (nothing is guessed from pixels):
  * which page is the floor plan and which pages are elevations,
  * levels printed on the elevations (FFL, door top, ceiling, wall top, roof ridge),
  * roof type and the length of the ridge (from which side the roof is seen as a flat band),
  * verandah roof, and the deck (size from the "DECK 11380 mm x 2000 mm" label, side from where
    that label sits relative to the room labels).
Needs PyMuPDF:  pip install pymupdf
"""
from __future__ import annotations
import math
import re

try:
    import pymupdf as fitz
except ImportError:                      # older PyMuPDF
    import fitz                          # type: ignore


def _lines(page, tol=3.0):
    """Words grouped into text lines: [(y, x0, x1, 'text of the line'), ...]."""
    words = sorted(page.get_text("words"), key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    rows, cur, cy = [], [], None
    for w in words:
        y = (w[1] + w[3]) / 2
        if cy is None or abs(y - cy) <= tol:
            cur.append(w)
            cy = y if cy is None else (cy + y) / 2
        else:
            rows.append(cur)
            cur, cy = [w], y
    if cur:
        rows.append(cur)
    out = []
    for r in rows:
        r.sort(key=lambda w: w[0])
        out.append((sum((w[1] + w[3]) / 2 for w in r) / len(r), r[0][0], r[-1][2], " ".join(w[4] for w in r)))
    return out


def _level(lines, name):
    """Value (m) printed next to a label like 'TRUSS ROOF LVL +3.290 m'."""
    pat = re.compile(name + r"\s*(?:LVL|LEVEL)?\s*([+-]?\s*\d+(?:\.\d+)?)", re.I)
    for _, _, _, text in lines:
        m = pat.search(text)
        if m:
            try:
                return float(m.group(1).replace(" ", ""))
            except ValueError:
                pass
    return None


def _slanted(page, y0, y1):
    """Number of long, gently sloping line segments inside a horizontal band: a gable end view."""
    n = 0
    for d in page.get_drawings():
        for it in d["items"]:
            if it[0] != "l":
                continue
            a, b = it[1], it[2]
            if not (y0 <= a.y <= y1 and y0 <= b.y <= y1):
                continue
            dx, dy = abs(b.x - a.x), abs(b.y - a.y)
            if dx < 40 or dy < 2:
                continue
            ang = math.degrees(math.atan2(dy, dx))
            if 4.0 <= ang <= 45.0:
                n += 1
    return n


def _numbers(page, y0, y1, lo=1000, hi=40000):
    vals = []
    for w in page.get_text("words"):
        yc = (w[1] + w[3]) / 2
        if y0 <= yc <= y1 and re.fullmatch(r"\d{4,5}", w[4]):
            v = int(w[4])
            if lo <= v <= hi:
                vals.append(v)
    return vals


def analyse_pdf(data: bytes) -> dict:
    doc = fitz.open(stream=data, filetype="pdf")
    ctx: dict = {"ok": False, "pages": len(doc), "floorPlanPage": None, "elevationPages": [],
                 "levels": {}, "roof": None, "deck": None, "warnings": []}

    # ---- classify pages
    texts = [p.get_text().upper() for p in doc]
    for i, t in enumerate(texts):
        if re.search(r"ELEVATION\s*-\s*[A-D]", t):
            ctx["elevationPages"].append(i)
        elif "FLOOR PLAN" in t and ctx["floorPlanPage"] is None:
            ctx["floorPlanPage"] = i
    if ctx["floorPlanPage"] is None:
        ctx["warnings"].append("No page titled FLOOR PLAN was found.")

    # ---- levels from the elevations
    lv = {"ffl": 0.0}
    for i in ctx["elevationPages"]:
        ln = _lines(doc[i])
        for key, name in (("ridge", r"TRUSS\s+ROOF"), ("ceiling", r"INNER\s+CEILING"),
                          ("doorTop", r"DOOR\s+TOP")):
            v = _level(ln, name)
            if v is not None and key not in lv:
                lv[key] = v
    if "ridge" in lv and "ceiling" in lv:
        # printed overall wall height (e.g. 2560): a 4-digit dimension between ceiling and ridge
        cands = []
        for i in ctx["elevationPages"]:
            cands += [v for v in _numbers(doc[i], 0, 2000, 2000, 4000)
                      if lv["ceiling"] * 1000 < v < lv["ridge"] * 1000]
        lv["wallTop"] = (max(set(cands), key=cands.count) / 1000.0) if cands else lv["ceiling"]
    ctx["levels"] = lv

    # ---- roof: type, ridge direction, verandah
    all_text = " ".join(texts)
    roof_type = ("gable" if "GABLE" in all_text else "hip" if re.search(r"\bHIP", all_text)
                 else "skillion" if re.search(r"SKILLION|MONO[- ]?PITCH", all_text)
                 else "flat" if "FLAT ROOF" in all_text else None)
    ridge_len = None
    side_views = flat_views = 0
    for i in ctx["elevationPages"]:
        page = doc[i]
        titles = [w for w in page.get_text("words") if w[4].upper() == "ELEVATION"]
        for t in titles:
            band = (t[1] - 460, t[1])                      # the drawing sits above its title
            s = _slanted(page, *band)
            if s >= 5:
                side_views += 1
            else:
                flat_views += 1
                nums = _numbers(page, *band)
                if nums:
                    ridge_len = max(ridge_len or 0, max(nums))
    if roof_type is None and "ridge" in lv and "wallTop" in lv:
        roof_type = "gable" if side_views else "flat"
    if roof_type and "ridge" in lv and "wallTop" in lv:
        ctx["roof"] = {
            "type": roof_type,
            "riseM": round(max(0.0, lv["ridge"] - lv["wallTop"]), 3),
            "ridgeLengthM": round(ridge_len / 1000.0, 3) if ridge_len else None,
            "verandah": "VERANDAH" in all_text,
        }
    elif roof_type:
        ctx["warnings"].append("Roof type found but no ridge level.")

    # ---- deck (floor plan page)
    fp = ctx["floorPlanPage"]
    if fp is not None:
        page = doc[fp]
        words = page.get_text("words")
        dims = []
        for m in re.finditer(r"(\d{3,5})\s*mm\s*x\s*(\d{3,5})\s*mm", page.get_text()):
            dims.append(m.group(0))
        decks = [w for w in words if w[4].upper() == "DECK"]
        # the plan label is the DECK word that has a "NNNN mm x NNNN mm" line right under it
        chosen = None
        for w in decks:
            below = [x for x in words if 0 < x[1] - w[3] < 25 and abs((x[0] + x[2]) / 2 - (w[0] + w[2]) / 2) < 80]
            line = " ".join(x[4] for x in sorted(below, key=lambda x: (round(x[1]), x[0])))
            m = re.search(r"(\d{3,5})\s*mm\s*x\s*(\d{3,5})\s*mm", line)
            if m:
                chosen = (w, int(m.group(1)), int(m.group(2)))
                break
        # room labels (inside the building) to find which side of it the deck lies on
        rooms = []
        for w in words:
            if not w[4].upper().startswith(("BEDROOM", "KITCHEN", "LOUNGE", "BATHROOM", "LAUNDRY",
                                            "LIVING", "DINING", "STUDY", "OFFICE", "GARAGE", "HALL")):
                continue
            below = [x for x in words if -2 < x[1] - w[3] < 22 and abs((x[0] + x[2]) / 2 - (w[0] + w[2]) / 2) < 90]
            line = " ".join(x[4] for x in sorted(below, key=lambda x: (round(x[1]), x[0])))
            if re.search(r"\d{3,5}\s*mm\s*x", line, re.I):          # a room label carries its size
                rooms.append(((w[0] + w[2]) / 2, (w[1] + w[3]) / 2))
        if chosen and rooms:
            w, a, b = chosen
            dx, dy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
            xs, ys = [r[0] for r in rooms], [r[1] for r in rooms]
            over = {"left": min(xs) - dx, "right": dx - max(xs), "top": min(ys) - dy, "bottom": dy - max(ys)}
            side = max(over, key=over.get)
            if over[side] > 0:
                ctx["deck"] = {"side": side, "dimsMm": [a, b],
                               "roofed": bool(ctx["roof"] and ctx["roof"]["verandah"])}
            else:
                ctx["warnings"].append("DECK label sits inside the building; side unknown.")
        elif decks:
            ctx["warnings"].append("DECK label found without a size line.")
    ctx["ok"] = bool(ctx["roof"] or ctx["deck"])
    return ctx


if __name__ == "__main__":
    import json, sys
    print(json.dumps(analyse_pdf(open(sys.argv[1], "rb").read()), indent=2))