import os
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Body
from fastapi.staticfiles import StaticFiles
import base64
from pydantic import BaseModel
from chat import ask

from detect import analyse, overlay_b64
from ml_detector import analyse_ml

import hashlib
from pdf_context import analyse_pdf
from floors import split_floors, floors_from_boxes

app = FastAPI(title="Floor plan -> 3D (MVP)")
PDF_DPI = 200

_design_cache: dict = {}

def design_of(data: bytes) -> dict | None:
    """PDF roof/deck/levels, cached by file hash. None for images or on failure."""
    if data[:4] != b"%PDF":
        return None
    key = hashlib.md5(data).hexdigest()
    if key not in _design_cache:
        try:
            _design_cache[key] = analyse_pdf(data)
        except Exception as e:           # never break plan detection because of this
            _design_cache[key] = {"ok": False, "floorPlanPage": None, "warnings": [str(e)]}
    return _design_cache[key]

def plan_page(data: bytes) -> int:
    d = design_of(data)
    p = d.get("floorPlanPage") if d else None
    return p if p is not None else 0


def pick_page(data: bytes, page: int, n: int) -> int:
    """The page the user chose, or the automatically found floor-plan page."""
    return page if 0 <= page < n else min(plan_page(data), n - 1)


def load_gray(data: bytes, filename: str, page: int = -1):
    """Returns (gray image, is_pdf)."""
    if filename.lower().endswith(".pdf") or data[:4] == b"%PDF":
        import fitz  # PyMuPDF
        doc = fitz.open(stream=data, filetype="pdf")
        pix = doc[pick_page(data, page, len(doc))].get_pixmap(dpi=PDF_DPI, colorspace=fitz.csGRAY)
        return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy(), True
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(400, "Could not decode image")
    return img, False

def apply_crop(gray, crop: str):
    """crop = 'x0,y0,x1,y1' as fractions (0..1) of the page. Empty = whole page."""
    if not crop:
        return gray
    try:
        x0, y0, x1, y1 = [float(v) for v in crop.split(",")]
    except ValueError:
        raise HTTPException(400, "Bad crop value")
    H, W = gray.shape
    clamp = lambda v: min(max(v, 0.0), 1.0)
    x0, x1 = sorted((clamp(x0), clamp(x1)))
    y0, y1 = sorted((clamp(y0), clamp(y1)))
    px0, px1, py0, py1 = int(x0 * W), int(x1 * W), int(y0 * H), int(y1 * H)
    if px1 - px0 < 50 or py1 - py0 < 50:
        raise HTTPException(400, "Selected area is too small")
    return gray[py0:py1, px0:px1].copy()

def apply_adjustments(
    gray: np.ndarray,
    brightness: float = 0,
    contrast: float = 0,
    highlights: float = 0,
    shadows: float = 0,
    whites: float = 0,
    blacks: float = 0,
    sharpness: float = 0,
    invert: bool = False,
) -> np.ndarray:
    """Apply tonal adjustments to an 8-bit grayscale floor-plan image."""
    brightness = float(np.clip(brightness, -100, 100))
    contrast = float(np.clip(contrast, -100, 100))
    highlights = float(np.clip(highlights, -100, 100))
    shadows = float(np.clip(shadows, -100, 100))
    whites = float(np.clip(whites, -100, 100))
    blacks = float(np.clip(blacks, -100, 100))
    sharpness = float(np.clip(sharpness, 0, 100))

    x = gray.astype(np.float32) / 255.0

    # Brightness and contrast.
    x += brightness / 200.0
    factor = 2.0 ** (contrast / 50.0)
    x = (x - 0.5) * factor + 0.5
    x = np.clip(x, 0.0, 1.0)

    # Shadows affect dark pixels more; highlights affect bright pixels more.
    shadow_mask = (1.0 - x) ** 2
    highlight_mask = x ** 2

    x += (shadows / 200.0) * shadow_mask
    x += (highlights / 200.0) * highlight_mask

    # Whites and blacks target the upper and lower tonal ranges.
    white_mask = np.clip((x - 0.55) / 0.45, 0.0, 1.0) ** 2
    black_mask = np.clip((0.45 - x) / 0.45, 0.0, 1.0) ** 2

    x += (whites / 200.0) * white_mask
    x += (blacks / 200.0) * black_mask
    x = np.clip(x, 0.0, 1.0)

    result = np.round(x * 255).astype(np.uint8)

    # Unsharp masking to enhance thin wall edges.
    if sharpness > 0:
        blurred = cv2.GaussianBlur(result, (0, 0), sigmaX=1.2)
        amount = sharpness / 50.0
        result = cv2.addWeighted(result, 1.0 + amount, blurred, -amount, 0)

    if invert:
        result = cv2.bitwise_not(result)

    return result


@app.post("/api/convert")
async def convert(
    file: UploadFile = File(...),
    drawing_scale: float = Form(100),      # 1:100 -> 100 (PDF only)
    px_per_m: float = Form(0),             # manual override (images)
    known_width_m: float = Form(0),
    crop: str = Form(""),
    brightness: float = Form(0),
    contrast: float = Form(0),
    highlights: float = Form(0),
    shadows: float = Form(0),
    whites: float = Form(0),
    blacks: float = Form(0),
    sharpness: float = Form(0),
    invert: bool = Form(False),  
    detector: str = Form("opencv"),      # real overall width of the plan -> auto-calibrate
    page: int = Form(-1),                # PDF page chosen by the user; -1 = found automatically
    ):
    data = await file.read()
    gray, is_pdf = load_gray(data, file.filename or "", page)
    gray = apply_crop(gray, crop)

    gray = apply_adjustments(
        gray,
        brightness=brightness,
        contrast=contrast,
        highlights=highlights,
        shadows=shadows,
        whites=whites,
        blacks=blacks,
        sharpness=sharpness,
        invert=invert,
    )
    # long side cap keeps processing fast
    s = 3000 / max(gray.shape)
    if s < 1:
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    else:
        s = 1.0

    if px_per_m > 0:
        ppm = px_per_m * s
    elif is_pdf:
        ppm = (PDF_DPI / 25.4) * 1000 / drawing_scale * s
    else:
        ppm = 60.0  # arbitrary fallback; user should calibrate

    if detector.lower() == "ml":
       model, ov = analyse_ml(gray, ppm)
    elif detector.lower() == "opencv":
       model, ov = analyse(gray, ppm)
    else:
       raise HTTPException(400, "detector must be 'opencv' or 'ml'")
    b = model["bounds"]
    width = b["x1"] - b["x0"]
    if known_width_m > 0 and width > 0:
        # re-run with the corrected scale so every metre-based threshold (min wall length,
        # door sizes, tolerances) is applied in real metres, not in the guessed scale
        ppm = ppm * width / known_width_m
        if detector.lower() == "ml":
            model, ov = analyse_ml(gray, ppm)
        else:
            model, ov = analyse(gray, ppm)

    if detector.lower() == "ml":
       import base64
       ok, buf = cv2.imencode(".png", ov)
       if not ok:
           raise HTTPException(500, "Could not encode ML overlay")
       model["overlay"] = "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
    else:
       model["overlay"] = overlay_b64(ov)
    d = design_of(data)
    if d and d.get("ok"):
        model["design"] = d
    
    ph, pw = gray.shape[:2]
    tex = gray
    if max(ph, pw) > 2400:                       # texture only; the physical size below is unchanged
        f = 2400 / max(ph, pw)
        tex = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", tex)
    if ok:
        model["plan"] = {
            "src": "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii"),
            "wM": pw / ppm, "hM": ph / ppm,      # size of the image in metres (final, calibrated ppm)
        }
    fl = split_floors(model)
    if fl:
        model["floors"] = fl

    return model


def _page_list(doc, data: bytes):
    """Small thumbnails of every page with the kind found by pdf_context (site, plan, elevation...)."""
    import fitz
    d = design_of(data) or {}
    info = {p["index"]: p for p in d.get("pageInfo", [])}
    out = []
    for i in range(min(len(doc), 16)):
        pg = doc[i]
        z = 300 / max(pg.rect.width, pg.rect.height)
        pix = pg.get_pixmap(matrix=fitz.Matrix(z, z), alpha=False)
        out.append({
            "index": i,
            "thumb": "data:image/jpeg;base64," + base64.b64encode(pix.tobytes("jpeg")).decode(),
            "kind": info.get(i, {}).get("kind", "other"),
            "label": info.get(i, {}).get("label", f"Page {i + 1}"),
        })
    return out


@app.post("/api/preview")
async def preview(file: UploadFile = File(...), page: int = Form(-1)):
    """A downscaled picture of one page so the user can choose the plan area.
    page = -1: the floor-plan page found automatically, plus thumbnails of all pages (multipage PDFs)."""
    data = await file.read()
    name = file.filename or ""
    page_list, used, suggest = [], 0, None
    if name.lower().endswith(".pdf") or data[:4] == b"%PDF":
        import fitz
        doc = fitz.open(stream=data, filetype="pdf")
        used = pick_page(data, page, len(doc))
        pg = doc[used]
        long_pt = max(pg.rect.width, pg.rect.height)
        dpi = int(min(150, 1800 * 72 / long_pt))      # int: PyMuPDF rejects a float dpi
        pix = pg.get_pixmap(dpi=dpi)
        png = pix.tobytes("png")
        w, h, pages = pix.width, pix.height, len(doc)
        text = "\n".join(p.get_text("text") for p in doc).strip()[:100_000]
        if pages > 1 and page < 0:
            page_list = _page_list(doc, data)
        d = design_of(data) or {}
        s = d.get("suggest")
        if s and s.get("page") == used:
            suggest = s
    else:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "Could not decode image")
        h, w = img.shape[:2]
        s = min(1.0, 1800 / max(h, w))
        if s < 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
        png = cv2.imencode(".png", img)[1].tobytes()
        pages = 1
        text = ""
    return {
        "image": "data:image/png;base64," + base64.b64encode(png).decode(),
        "width": w, "height": h, "pages": pages, "page": used,
        "pageList": page_list, "suggest": suggest,
        "text": text,
    }

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    document: str
    messages: list[ChatMessage]


@app.post("/api/chat")
def chat(req: ChatRequest):
    if not req.document.strip():
        raise HTTPException(400, "This file has no readable text, so there is nothing to ask about.")
    try:
        reply = ask(req.document, [{"role": m.role, "content": m.content} for m in req.messages])
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"The language model call failed: {e}")
    return {"reply": reply}

@app.post("/api/floors")
async def api_floors(payload: dict = Body(...)):
    model = payload["model"]                      # walls, openings, stairs, railings
    boxes = payload.get("boxes") or []
    if boxes:
        info = floors_from_boxes(model, boxes, payload.get("labels"))
    else:
        info = split_floors(model, count=int(payload.get("count") or 0))
    return {"floors": info}

app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "frontend"), html=True), name="ui")