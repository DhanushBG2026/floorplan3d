import os
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
import base64
from pydantic import BaseModel
from chat import ask

from detect import analyse, overlay_b64

app = FastAPI(title="Floor plan -> 3D (MVP)")
PDF_DPI = 200


def load_gray(data: bytes, filename: str):
    """Returns (gray image, is_pdf)."""
    if filename.lower().endswith(".pdf") or data[:4] == b"%PDF":
        import fitz  # PyMuPDF
        doc = fitz.open(stream=data, filetype="pdf")
        pix = doc[0].get_pixmap(dpi=PDF_DPI, colorspace=fitz.csGRAY)
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


@app.post("/api/convert")
async def convert(
    file: UploadFile = File(...),
    drawing_scale: float = Form(100),      # 1:100 -> 100 (PDF only)
    px_per_m: float = Form(0),             # manual override (images)
    known_width_m: float = Form(0),
    crop: str = Form(""),        # real overall width of the plan -> auto-calibrate
):
    gray, is_pdf = load_gray(await file.read(), file.filename or "")
    gray = apply_crop(gray, crop)
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

    model, ov = analyse(gray, ppm)
    b = model["bounds"]
    width = b["x1"] - b["x0"]
    if known_width_m > 0 and width > 0:
        # re-run with the corrected scale so every metre-based threshold (min wall length,
        # door sizes, tolerances) is applied in real metres, not in the guessed scale
        ppm = ppm * width / known_width_m
        model, ov = analyse(gray, ppm)

    model["overlay"] = overlay_b64(ov)
    return model


@app.post("/api/preview")
async def preview(file: UploadFile = File(...)):
    """Returns a downscaled picture of the first page so the user can choose the plan area."""
    data = await file.read()
    name = file.filename or ""
    if name.lower().endswith(".pdf") or data[:4] == b"%PDF":
        import fitz
        doc = fitz.open(stream=data, filetype="pdf")
        page = doc[0]
        long_pt = max(page.rect.width, page.rect.height)
        dpi = min(150, 1800 * 72 / long_pt)          # keep the preview around 1800 px
        pix = page.get_pixmap(dpi=dpi)
        png = pix.tobytes("png")
        w, h, pages = pix.width, pix.height, len(doc)
        text = "\n".join(p.get_text("text") for p in doc).strip()[:100_000]   # <- new
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
        text = ""                                                              # <- new
    return {
        "image": "data:image/png;base64," + base64.b64encode(png).decode(),
        "width": w, "height": h, "pages": pages,
        "text": text,                                                          # <- new
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

app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "frontend"), html=True), name="ui")