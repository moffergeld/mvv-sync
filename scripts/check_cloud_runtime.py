"""Check cloud dependencies and FORMS contracts without accessing secrets or data."""

import ast
import importlib
import io
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def check_forms_contracts() -> None:
    """Keep the login, player and tablet forms present during dependency repairs."""
    expected_markers = {
        "app.py": ("login_form", "form_submit_button"),
        "roles.py": ("extra_streamlit_components", "CookieManager", "mvv_cookie_mgr"),
        "pages/Subscripts/player_tab_forms.py": (
            "render_forms_tab",
            "asrm_form",
            "rpe_form",
            "ASRM opslaan",
            "RPE opslaan",
        ),
        "pages/07_Player_Page_Beta.py": ("render_forms_tab", "forms_status"),
        "tablet_app/app.py": (
            "tablet_cookie_mgr",
            "tablet_asrm_form_",
            "tablet_rpe_form_",
            "forms_status",
        ),
    }
    for relative_path, markers in expected_markers.items():
        path = REPOSITORY_ROOT / relative_path
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path))
        missing = [marker for marker in markers if marker not in source]
        if missing:
            raise RuntimeError(f"FORMS contract missing from {relative_path}: {', '.join(missing)}")
        print(f"FORMS contract passed: {relative_path}")


def check_reportlab_pdf() -> None:
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.drawString(72, 800, "MVV PDF fallback check")
    pdf.save()
    pdf_bytes = buffer.getvalue()
    if not pdf_bytes.startswith(b"%PDF-") or len(pdf_bytes) < 1000:
        raise RuntimeError("ReportLab did not produce a usable fallback PDF.")
    print(f"ReportLab PDF fallback passed ({len(pdf_bytes)} bytes).")


def check_optional_weasyprint() -> None:
    try:
        from weasyprint import HTML

        pdf = HTML(string="<html><body><h1>MVV PDF check</h1></body></html>").write_pdf()
        if not pdf.startswith(b"%PDF-") or len(pdf) < 1000:
            raise RuntimeError("WeasyPrint did not produce a usable PDF.")
        print(f"WeasyPrint PDF rendering passed ({len(pdf)} bytes).")
    except (ImportError, OSError) as exc:
        print(f"WeasyPrint native runtime unavailable; ReportLab fallback will be used: {exc}")


def main() -> None:
    print(f"Checking cloud runtime: {sys.executable} ({sys.version.split()[0]})")
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError("The current Streamlit app uses Python 3.13.")

    modules = (
        "streamlit",
        "extra_streamlit_components",
        "pandas",
        "requests",
        "dateutil",
        "openpyxl",
        "xlrd",
        "supabase",
        "numpy",
        "plotly",
        "streamlit_calendar",
        "reportlab",
        "jinja2",
    )
    for name in modules:
        module = importlib.import_module(name)
        print(f"Imported {name}: {module.__file__}")

    check_reportlab_pdf()
    check_optional_weasyprint()
    check_forms_contracts()


if __name__ == "__main__":
    main()
