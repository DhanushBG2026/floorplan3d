# Floor plan → 3D (MVP)

    cd backend
    python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    uvicorn app:app --reload
    # open http://localhost:8000

Upload a PDF/PNG/JPG, set the scale (PDF: 1:N; image: real plan width in metres), click Generate.
`/api/convert` returns JSON (walls, openings, bounds, debug overlay) that the Three.js viewer renders.
