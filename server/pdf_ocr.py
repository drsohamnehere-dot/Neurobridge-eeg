"""Optional OCR helper for scanned PDF report pages."""
import shutil
from io import BytesIO


def extract_scanned_page_text(page, page_index: int, enabled: bool = True) -> tuple[str, str]:
    """Return (text, status). Only call for pages lacking embedded text."""
    if not enabled:
        return "", "disabled"
    if page_index >= 5:
        return "", "page_limit"
    if not shutil.which("tesseract"):
        return "", "tesseract_unavailable"
    try:
        import pytesseract
        from PIL import Image
        import fitz
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        image = Image.open(BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(image, config="--psm 6", timeout=3).strip()
        return text, "completed" if text else "no_text"
    except Exception as exc:
        return "", "error"
