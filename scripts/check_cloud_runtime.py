"""Check installed cloud dependencies without accessing app secrets or data."""

import importlib
import sys


def main() -> None:
    print(f"Checking cloud runtime: {sys.executable} ({sys.version.split()[0]})")
    if sys.version_info[:2] != (3, 14):
        raise RuntimeError("The cloud dependency check requires Python 3.14.")

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
        "weasyprint",
    )
    for name in modules:
        module = importlib.import_module(name)
        print(f"Imported {name}: {module.__file__}")

    from weasyprint import HTML

    pdf = HTML(string="<html><body><h1>MVV PDF check</h1></body></html>").write_pdf()
    if not pdf.startswith(b"%PDF-") or len(pdf) < 1000:
        raise RuntimeError("WeasyPrint did not produce a usable PDF.")
    print(f"PDF rendering passed ({len(pdf)} bytes).")


if __name__ == "__main__":
    main()
