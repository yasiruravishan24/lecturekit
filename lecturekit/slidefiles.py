"""Turn the user's slide files (one or several; PDF, PowerPoint, Word, ...) into the single slides.pdf the pipeline reads.

PDF is used as is. Office formats are converted with LibreOffice (`brew install --cask libreoffice`).
Several files are merged in the order given, so give them in lecture order.
"""
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Union

from .errors import ExternalToolError, MissingInputError, ValidationError

logger = logging.getLogger(__name__)

OFFICE_TYPES = {".pptx", ".ppt", ".pps", ".ppsx", ".odp", ".docx", ".doc", ".odt", ".rtf"}
SLIDE_TYPES = {".pdf"} | OFFICE_TYPES
_SOFFICE_NAMES = ("soffice", "libreoffice", "/Applications/LibreOffice.app/Contents/MacOS/soffice")


def as_list(slides: Union[None, str, Path, Iterable]) -> list:
    """None -> [], one path -> [path], several -> list of Paths."""
    if slides is None or slides == "" or slides == []:
        return []
    if isinstance(slides, (str, Path)):
        return [Path(slides)]
    return [Path(s) for s in slides]


def validate(files: list) -> None:
    for f in files:
        if not f.exists():
            raise MissingInputError(f"slides not found: {f}")
        if f.suffix.lower() not in SLIDE_TYPES:
            raise ValidationError(f"unsupported slides type {f.suffix!r}; use one of {sorted(SLIDE_TYPES)}")


def _soffice() -> str:
    for name in _SOFFICE_NAMES:
        found = shutil.which(name)
        if found:
            return found
    raise ExternalToolError("LibreOffice is needed to read PowerPoint/Word files "
                            "(brew install --cask libreoffice), or export the slides to PDF yourself")


def _office_to_pdf(src: Path, work: Path) -> Path:
    cmd = [_soffice(), "--headless", "--convert-to", "pdf", "--outdir", str(work), str(src)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        raise ExternalToolError(f"LibreOffice timed out converting {src.name}")
    out = work / (src.stem + ".pdf")
    if r.returncode != 0 or not out.exists():
        tail = (r.stderr or r.stdout or "").strip()[-300:]
        raise ExternalToolError(f"LibreOffice could not convert {src.name}: {tail}")
    return out


def build_pdf(files: list, out_pdf: Path) -> Path:
    """Write out_pdf from the given slide files (already validated). Returns out_pdf."""
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        pdfs = []
        for f in files:
            if f.suffix.lower() == ".pdf":
                pdfs.append(f)
            else:
                logger.info("converting %s to PDF", f.name)
                pdfs.append(_office_to_pdf(f, Path(tmp)))
        if len(pdfs) == 1:
            if pdfs[0].resolve() != out_pdf.resolve():
                shutil.copy2(pdfs[0], out_pdf)
            return out_pdf
        logger.info("merging %d slide files into %s", len(pdfs), out_pdf.name)
        try:
            import pypdfium2 as pdfium       # installed with pdfplumber
            merged = pdfium.PdfDocument.new()
            for p in pdfs:
                src = pdfium.PdfDocument(str(p))
                merged.import_pages(src)
            merged.save(str(out_pdf))
        except Exception as e:               # corrupt/encrypted PDF: report which problem, don't crash
            raise ExternalToolError(f"could not merge the slide files: {type(e).__name__}: {e}")
    return out_pdf
