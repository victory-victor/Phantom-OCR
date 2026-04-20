from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
import pytesseract
import cv2
import numpy as np
import os

app = FastAPI()

if os.name == "nt":  # Windows
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


@app.get("/", response_class=HTMLResponse)
async def home():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("<h2>index.html not found</h2>", status_code=404)


@app.post("/extract-text/")
async def extract_text(file: UploadFile = File(...)):
    try:
        # ✅ Validate file type
        if not file.content_type.startswith("image/"):
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "File must be an image"}
            )

        contents = await file.read()

        # ✅ File size limit (5MB)
        if len(contents) > 5 * 1024 * 1024:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "File too large (max 5MB)"}
            )

        # ✅ Convert bytes → OpenCV image
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Invalid image file"}
            )

        # ✅ Preprocessing for better OCR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Reduce noise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive threshold (better than fixed threshold)
        gray = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )

        # ✅ OCR configuration
        custom_config = r'--oem 3 --psm 6'

        text = pytesseract.image_to_string(
            gray,
            config=custom_config,
            lang="eng"  # 👉 Add more like "eng+hin" if needed
        )

        return {
            "success": True,
            "text": text.strip()
        }

    except Exception:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "Internal server error"}
        )