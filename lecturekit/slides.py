"""Slide text: PDF text layer first, Tesseract OCR on page images otherwise."""
import json
import time
from pathlib import Path


def extract(pdf: Path, out: Path, force_ocr: bool = False, dpi: int = 150) -> dict:
    import pdfplumber
    t0 = time.time()
    slides = []
    with pdfplumber.open(pdf) as doc:
        for n, page in enumerate(doc.pages, start=1):
            text, source = "", "pdf"
            if not force_ocr:
                text = (page.extract_text() or "").strip()
            if force_ocr or len(text) < 15:         # image-only slide -> OCR
                import pytesseract
                img = page.to_image(resolution=dpi).original
                text, source = pytesseract.image_to_string(img).strip(), "ocr"
            slides.append({"slide": n, "text": text, "source": source})
    result = {"engine": "tesseract" if force_ocr else "pdf+tesseract",
              "seconds": round(time.time() - t0, 1), "slides": slides}
    out.write_text(json.dumps(result, indent=1))
    return result
