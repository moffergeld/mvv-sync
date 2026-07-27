from __future__ import annotations

import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

from html_report_generator import build_player_report_html_pdf_bytes, build_week_report_html_pdf_bytes

REPORT_STYLE_LABELS: dict[str, str] = {"html": "Rapport"}
REPORT_STYLE_OPTIONS: tuple[str, ...] = tuple(REPORT_STYLE_LABELS.keys())

ReportBuilder = Callable[..., bytes]


def normalize_report_style(report_style: str | None = None) -> str:
    style = (report_style or "html").strip().lower()
    if style not in REPORT_STYLE_LABELS:
        allowed = ", ".join(REPORT_STYLE_OPTIONS)
        raise ValueError(f"Onbekende report_style '{report_style}'. Toegestane waarden: {allowed}.")
    return style


def _call_builder(builder: ReportBuilder, payload: Mapping[str, Any]) -> bytes:
    signature = inspect.signature(builder)
    compatible_kwargs = {key: value for key, value in payload.items() if key in signature.parameters}
    return builder(**compatible_kwargs)


def _resolve_builder(report_kind: str, report_style: str) -> ReportBuilder:
    builders: dict[tuple[str, str], ReportBuilder] = {
        ("week", "html"): build_week_report_html_pdf_bytes,
        ("player", "html"): build_player_report_html_pdf_bytes,
    }
    key_style = normalize_report_style(report_style)
    key = (report_kind.strip().lower(), key_style)
    if key not in builders:
        supported = sorted({kind for kind, _ in builders})
        raise ValueError(f"Onbekend rapporttype '{report_kind}'. Beschikbare types: {', '.join(supported)}.")
    return builders[key]


def generate_report(
    report_kind: str,
    data: Mapping[str, Any] | None = None,
    output_path: str | Path | None = None,
    report_style: str = "html",
    **kwargs: Any,
) -> bytes:
    payload: dict[str, Any] = {}
    if data:
        payload.update(dict(data))
    payload.update(kwargs)
    style = normalize_report_style(report_style)
    builder = _resolve_builder(report_kind, style)
    pdf_bytes = _call_builder(builder, payload)
    if output_path is not None:
        Path(output_path).expanduser().resolve().write_bytes(pdf_bytes)
    return pdf_bytes


def generate_week_report(
    data: Mapping[str, Any] | None = None,
    output_path: str | Path | None = None,
    report_style: str = "html",
    **kwargs: Any,
) -> bytes:
    return generate_report(
        "week",
        data=data,
        output_path=output_path,
        report_style=report_style,
        **kwargs,
    )


def generate_player_report(
    data: Mapping[str, Any] | None = None,
    output_path: str | Path | None = None,
    report_style: str = "html",
    **kwargs: Any,
) -> bytes:
    return generate_report(
        "player",
        data=data,
        output_path=output_path,
        report_style=report_style,
        **kwargs,
    )
