import pytest
import pypdfium2 as pdfium

from lecturekit import slidefiles
from lecturekit.errors import ExternalToolError, MissingInputError, ValidationError


def make_pdf(path, pages):
    doc = pdfium.PdfDocument.new()
    for _ in range(pages):
        doc.new_page(200, 200)
    doc.save(str(path))
    return path


def test_as_list():
    assert slidefiles.as_list(None) == []
    assert [p.name for p in slidefiles.as_list("a.pdf")] == ["a.pdf"]
    assert [p.name for p in slidefiles.as_list(["a.pdf", "b.pdf"])] == ["a.pdf", "b.pdf"]


def test_validate_errors(tmp_path):
    (tmp_path / "x.txt").write_text("hi")
    with pytest.raises(MissingInputError):
        slidefiles.validate(slidefiles.as_list(tmp_path / "nope.pdf"))
    with pytest.raises(ValidationError):
        slidefiles.validate(slidefiles.as_list(tmp_path / "x.txt"))


def test_single_pdf_is_copied(tmp_path):
    src = make_pdf(tmp_path / "a.pdf", 2)
    out = slidefiles.build_pdf([src], tmp_path / "w" / "slides.pdf")
    assert len(pdfium.PdfDocument(str(out))) == 2


def test_several_pdfs_merge_in_order(tmp_path):
    a, b = make_pdf(tmp_path / "a.pdf", 2), make_pdf(tmp_path / "b.pdf", 3)
    out = slidefiles.build_pdf([a, b], tmp_path / "slides.pdf")
    assert len(pdfium.PdfDocument(str(out))) == 5


def test_office_file_without_libreoffice(tmp_path, monkeypatch):
    f = tmp_path / "deck.pptx"
    f.write_bytes(b"x")
    monkeypatch.setattr(slidefiles.shutil, "which", lambda name: None)
    with pytest.raises(ExternalToolError, match="LibreOffice"):
        slidefiles.build_pdf([f], tmp_path / "slides.pdf")
