from __future__ import annotations

import math
from html import escape
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import pandas as pd

from pages.Subscripts.mvv_branding import TEAM_LOGO

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
CSS_PATH = BASE_DIR / "static" / "css" / "report.css"
LOGO_SRC = TEAM_LOGO.relative_to(BASE_DIR).as_posix() if TEAM_LOGO.exists() else ""

SVG_COLORS = ["#C8102E", "#6E1222", "#EA3351", "#F59E0B", "#2563EB", "#0F766E"]

HTML_RUNTIME_UNAVAILABLE_MESSAGE = (
    "Nieuw vormgegeven rapport is op deze server nog niet beschikbaar. "
    "De WeasyPrint runtime mist nog Linux-systeembibliotheken. "
    "Na redeploy met de packages uit packages.txt moet deze stijl werken. "
    "Kies tijdelijk 'Klassiek rapport'."
)


def _require_jinja2() -> Any:
    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
    except ImportError as exc:
        raise RuntimeError(
            "HTML rapportstijl vereist Jinja2. Voeg jinja2 toe aan requirements.txt en installeer de dependency."
        ) from exc
    return Environment, FileSystemLoader, select_autoescape


def _require_weasyprint() -> Any:
    try:
        from weasyprint import CSS, HTML
    except (ImportError, OSError) as exc:
        raise RuntimeError(HTML_RUNTIME_UNAVAILABLE_MESSAGE) from exc
    return HTML, CSS


def _template_environment() -> Any:
    Environment, FileSystemLoader, select_autoescape = _require_jinja2()
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _render_html_pdf(template_name: str, context: dict[str, Any]) -> bytes:
    html_cls, css_cls = _require_weasyprint()
    env = _template_environment()
    template = env.get_template(template_name)
    html = template.render(**context)
    return html_cls(string=html, base_url=str(BASE_DIR)).write_pdf(
        stylesheets=[css_cls(filename=str(CSS_PATH), base_url=str(BASE_DIR))]
    )


def _fmt_int(value: object) -> str:
    if pd.isna(value):
        return "--"
    return f"{int(round(float(value))):,}".replace(",", ".")


def _fmt_dec(value: object, decimals: int = 1) -> str:
    if pd.isna(value):
        return "--"
    formatted = f"{float(value):,.{decimals}f}"
    return formatted.replace(",", "X").replace(".", ",").replace("X", ".")


def _fmt_distance(value: object) -> str:
    base = _fmt_int(value)
    return "--" if base == "--" else f"{base} m"


def _fmt_speed(value: object) -> str:
    base = _fmt_dec(value, 1)
    return "--" if base == "--" else f"{base} km/h"


def _fmt_minutes(value: object) -> str:
    base = _fmt_int(value)
    return "--" if base == "--" else f"{base} min"


def _fmt_text(value: object) -> str:
    if value is None or pd.isna(value):
        return "--"
    return str(value)


def _fmt_axis(value: float) -> str:
    if abs(value) >= 1000:
        return _fmt_int(value)
    if value.is_integer():
        return str(int(value))
    return _fmt_dec(value, 1)


def _clean_series(values: Sequence[object]) -> list[float]:
    cleaned: list[float] = []
    for value in values:
        numeric = pd.to_numeric(value, errors="coerce")
        cleaned.append(float(numeric) if pd.notna(numeric) else 0.0)
    return cleaned


def _nice_max(value: float) -> float:
    if value <= 0:
        return 1.0
    padded = value * 1.15
    magnitude = 10 ** math.floor(math.log10(padded))
    normalized = padded / magnitude
    if normalized <= 1:
        nice = 1
    elif normalized <= 2:
        nice = 2
    elif normalized <= 5:
        nice = 5
    else:
        nice = 10
    return nice * magnitude


def _empty_svg(title: str, message: str, *, width: int = 860, height: int = 250) -> str:
    return f"""
    <svg class="chart-svg" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{escape(title)}">
      <rect x="0" y="0" width="{width}" height="{height}" rx="16" fill="#F8FAFC" stroke="#D7DEE8" />
      <text x="24" y="36" font-size="22" font-weight="700" fill="#0B1020">{escape(title)}</text>
      <text x="{width/2:.0f}" y="{height/2:.0f}" text-anchor="middle" font-size="15" fill="#64748B">{escape(message)}</text>
    </svg>
    """.strip()


def _build_vertical_bar_chart_svg(
    title: str,
    labels: Sequence[object],
    values: Sequence[object],
    *,
    color: str = "#C8102E",
    width: int = 860,
    height: int = 270,
    y_max: float | None = None,
    formatter: Callable[[object], str] = _fmt_int,
) -> str:
    clean_labels = [_fmt_text(label) for label in labels]
    clean_values = _clean_series(values)
    if not clean_labels or not clean_values or max(clean_values, default=0) <= 0:
        return _empty_svg(title, "Geen data beschikbaar.", width=width, height=height)

    chart_max = y_max if y_max is not None else _nice_max(max(clean_values))
    margin_left, margin_right, margin_top, margin_bottom = 64, 24, 46, 90
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    slot_width = plot_width / max(1, len(clean_values))
    bar_width = max(16, min(44, slot_width * 0.58))
    label_font = 9 if len(clean_labels) > 8 else 10
    grid_lines = 4

    parts: list[str] = [
        f'<svg class="chart-svg" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{escape(title)}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="16" fill="#F8FAFC" stroke="#D7DEE8" />',
        f'<text x="24" y="36" font-size="22" font-weight="700" fill="#0B1020">{escape(title)}</text>',
    ]

    for step in range(grid_lines + 1):
        ratio = step / grid_lines
        y = margin_top + plot_height - ratio * plot_height
        axis_value = chart_max * ratio
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#E2E8F0" stroke-dasharray="4 6" />')
        parts.append(f'<text x="{margin_left - 12}" y="{y + 4:.1f}" text-anchor="end" font-size="10" fill="#64748B">{escape(_fmt_axis(axis_value))}</text>')

    for index, (label, value) in enumerate(zip(clean_labels, clean_values)):
        x_center = margin_left + slot_width * index + slot_width / 2
        bar_height = 0 if chart_max <= 0 else (value / chart_max) * plot_height
        x = x_center - bar_width / 2
        y = margin_top + plot_height - bar_height
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="5" fill="{color}" />')
        if value > 0:
            parts.append(f'<text x="{x_center:.1f}" y="{max(y - 8, margin_top + 12):.1f}" text-anchor="middle" font-size="10" font-weight="700" fill="#0F172A">{escape(formatter(value))}</text>')
        label_y = height - 24
        parts.append(
            f'<text x="{x_center:.1f}" y="{label_y}" font-size="{label_font}" fill="#475569" text-anchor="end" transform="rotate(-42 {x_center:.1f} {label_y})">{escape(label)}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def _build_grouped_bar_chart_svg(
    title: str,
    labels: Sequence[object],
    series: Sequence[dict[str, Any]],
    *,
    width: int = 860,
    height: int = 280,
    y_max: float | None = None,
) -> str:
    clean_labels = [_fmt_text(label) for label in labels]
    if not clean_labels or not series:
        return _empty_svg(title, "Geen data beschikbaar.", width=width, height=height)

    clean_series = [
        {
            "label": str(item.get("label") or "Serie"),
            "color": str(item.get("color") or SVG_COLORS[index % len(SVG_COLORS)]),
            "values": _clean_series(item.get("values") or []),
        }
        for index, item in enumerate(series)
    ]
    flat_values = [value for item in clean_series for value in item["values"]]
    if max(flat_values, default=0) <= 0:
        return _empty_svg(title, "Geen data beschikbaar.", width=width, height=height)

    chart_max = y_max if y_max is not None else _nice_max(max(flat_values))
    margin_left, margin_right, margin_top, margin_bottom = 64, 24, 62, 90
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    slot_width = plot_width / max(1, len(clean_labels))
    series_width = slot_width * 0.72
    bar_width = max(9, min(22, series_width / max(1, len(clean_series))))
    grid_lines = 4
    label_font = 9 if len(clean_labels) > 8 else 10

    parts: list[str] = [
        f'<svg class="chart-svg" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{escape(title)}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="16" fill="#F8FAFC" stroke="#D7DEE8" />',
        f'<text x="24" y="36" font-size="22" font-weight="700" fill="#0B1020">{escape(title)}</text>',
    ]

    legend_x = 24
    for item in clean_series:
        parts.append(f'<rect x="{legend_x}" y="44" width="12" height="12" rx="3" fill="{item["color"]}" />')
        parts.append(f'<text x="{legend_x + 18}" y="54" font-size="10" fill="#475569">{escape(item["label"])}</text>')
        legend_x += max(100, len(item["label"]) * 7 + 34)

    for step in range(grid_lines + 1):
        ratio = step / grid_lines
        y = margin_top + plot_height - ratio * plot_height
        axis_value = chart_max * ratio
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#E2E8F0" stroke-dasharray="4 6" />')
        parts.append(f'<text x="{margin_left - 12}" y="{y + 4:.1f}" text-anchor="end" font-size="10" fill="#64748B">{escape(_fmt_axis(axis_value))}</text>')

    for label_index, label in enumerate(clean_labels):
        x_slot = margin_left + slot_width * label_index
        x_start = x_slot + (slot_width - series_width) / 2
        for series_index, item in enumerate(clean_series):
            value = item["values"][label_index] if label_index < len(item["values"]) else 0.0
            bar_height = 0 if chart_max <= 0 else (value / chart_max) * plot_height
            x = x_start + series_index * bar_width
            y = margin_top + plot_height - bar_height
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width - 2:.1f}" height="{bar_height:.1f}" rx="4" fill="{item["color"]}" />')
        label_x = x_slot + slot_width / 2
        label_y = height - 24
        parts.append(
            f'<text x="{label_x:.1f}" y="{label_y}" font-size="{label_font}" fill="#475569" text-anchor="end" transform="rotate(-42 {label_x:.1f} {label_y})">{escape(label)}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def _build_error_bar_chart_svg(
    title: str,
    labels: Sequence[object],
    mean_values: Sequence[object],
    error_values: Sequence[object],
    *,
    color: str = "#C8102E",
    width: int = 860,
    height: int = 280,
    y_max: float | None = None,
    formatter: Callable[[object], str] = _fmt_int,
) -> str:
    clean_labels = [_fmt_text(label) for label in labels]
    clean_means = _clean_series(mean_values)
    clean_errors = _clean_series(error_values)
    if not clean_labels or not clean_means or max(clean_means, default=0) <= 0:
        return _empty_svg(title, "Geen data beschikbaar.", width=width, height=height)

    maxima = [mean + error for mean, error in zip(clean_means, clean_errors, strict=False)]
    chart_max = y_max if y_max is not None else _nice_max(max(maxima, default=max(clean_means, default=0)))
    margin_left, margin_right, margin_top, margin_bottom = 64, 24, 46, 90
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    slot_width = plot_width / max(1, len(clean_means))
    bar_width = max(16, min(42, slot_width * 0.54))
    label_font = 9 if len(clean_labels) > 8 else 10
    grid_lines = 4

    parts: list[str] = [
        f'<svg class="chart-svg" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{escape(title)}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="16" fill="#F8FAFC" stroke="#D7DEE8" />',
        f'<text x="24" y="36" font-size="22" font-weight="700" fill="#0B1020">{escape(title)}</text>',
    ]

    for step in range(grid_lines + 1):
        ratio = step / grid_lines
        y = margin_top + plot_height - ratio * plot_height
        axis_value = chart_max * ratio
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#E2E8F0" stroke-dasharray="4 6" />')
        parts.append(f'<text x="{margin_left - 12}" y="{y + 4:.1f}" text-anchor="end" font-size="10" fill="#64748B">{escape(_fmt_axis(axis_value))}</text>')

    for index, (label, mean_value, error_value) in enumerate(zip(clean_labels, clean_means, clean_errors, strict=False)):
        x_center = margin_left + slot_width * index + slot_width / 2
        bar_height = 0 if chart_max <= 0 else (mean_value / chart_max) * plot_height
        x = x_center - bar_width / 2
        y = margin_top + plot_height - bar_height
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="5" fill="{color}" />')

        error_top_value = min(chart_max, mean_value + error_value)
        error_top_y = margin_top + plot_height - ((error_top_value / chart_max) * plot_height if chart_max > 0 else 0)
        cap_half = max(6, bar_width * 0.28)
        parts.append(f'<line x1="{x_center:.1f}" y1="{error_top_y:.1f}" x2="{x_center:.1f}" y2="{y:.1f}" stroke="#475569" stroke-width="1.4" />')
        parts.append(f'<line x1="{x_center - cap_half:.1f}" y1="{error_top_y:.1f}" x2="{x_center + cap_half:.1f}" y2="{error_top_y:.1f}" stroke="#475569" stroke-width="1.4" />')

        if mean_value > 0:
            parts.append(f'<text x="{x_center:.1f}" y="{max(y - 8, margin_top + 12):.1f}" text-anchor="middle" font-size="10" font-weight="700" fill="#0F172A">{escape(formatter(mean_value))}</text>')
        label_y = height - 24
        parts.append(
            f'<text x="{x_center:.1f}" y="{label_y}" font-size="{label_font}" fill="#475569" text-anchor="end" transform="rotate(-42 {x_center:.1f} {label_y})">{escape(label)}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def _build_grouped_error_bar_chart_svg(
    title: str,
    labels: Sequence[object],
    series: Sequence[dict[str, Any]],
    *,
    width: int = 860,
    height: int = 290,
    y_max: float | None = None,
) -> str:
    clean_labels = [_fmt_text(label) for label in labels]
    if not clean_labels or not series:
        return _empty_svg(title, "Geen data beschikbaar.", width=width, height=height)

    clean_series = [
        {
            "label": str(item.get("label") or "Serie"),
            "color": str(item.get("color") or SVG_COLORS[index % len(SVG_COLORS)]),
            "values": _clean_series(item.get("values") or []),
            "errors": _clean_series(item.get("errors") or []),
        }
        for index, item in enumerate(series)
    ]
    flat_maxima = [
        value + (item["errors"][idx] if idx < len(item["errors"]) else 0.0)
        for item in clean_series
        for idx, value in enumerate(item["values"])
    ]
    if max(flat_maxima, default=0) <= 0:
        return _empty_svg(title, "Geen data beschikbaar.", width=width, height=height)

    chart_max = y_max if y_max is not None else _nice_max(max(flat_maxima))
    margin_left, margin_right, margin_top, margin_bottom = 64, 24, 62, 90
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    slot_width = plot_width / max(1, len(clean_labels))
    series_width = slot_width * 0.72
    bar_width = max(9, min(20, series_width / max(1, len(clean_series))))
    grid_lines = 4
    label_font = 9 if len(clean_labels) > 8 else 10

    parts: list[str] = [
        f'<svg class="chart-svg" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{escape(title)}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="16" fill="#F8FAFC" stroke="#D7DEE8" />',
        f'<text x="24" y="36" font-size="22" font-weight="700" fill="#0B1020">{escape(title)}</text>',
    ]

    legend_x = 24
    for item in clean_series:
        parts.append(f'<rect x="{legend_x}" y="44" width="12" height="12" rx="3" fill="{item["color"]}" />')
        parts.append(f'<text x="{legend_x + 18}" y="54" font-size="10" fill="#475569">{escape(item["label"])}</text>')
        legend_x += max(100, len(item["label"]) * 7 + 34)

    for step in range(grid_lines + 1):
        ratio = step / grid_lines
        y = margin_top + plot_height - ratio * plot_height
        axis_value = chart_max * ratio
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#E2E8F0" stroke-dasharray="4 6" />')
        parts.append(f'<text x="{margin_left - 12}" y="{y + 4:.1f}" text-anchor="end" font-size="10" fill="#64748B">{escape(_fmt_axis(axis_value))}</text>')

    for label_index, label in enumerate(clean_labels):
        x_slot = margin_left + slot_width * label_index
        x_start = x_slot + (slot_width - series_width) / 2
        for series_index, item in enumerate(clean_series):
            value = item["values"][label_index] if label_index < len(item["values"]) else 0.0
            error = item["errors"][label_index] if label_index < len(item["errors"]) else 0.0
            bar_height = 0 if chart_max <= 0 else (value / chart_max) * plot_height
            x = x_start + series_index * bar_width
            y = margin_top + plot_height - bar_height
            draw_width = max(6, bar_width - 2)
            x_center = x + draw_width / 2
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{draw_width:.1f}" height="{bar_height:.1f}" rx="4" fill="{item["color"]}" />')
            error_top_value = min(chart_max, value + error)
            error_top_y = margin_top + plot_height - ((error_top_value / chart_max) * plot_height if chart_max > 0 else 0)
            cap_half = max(4, draw_width * 0.28)
            parts.append(f'<line x1="{x_center:.1f}" y1="{error_top_y:.1f}" x2="{x_center:.1f}" y2="{y:.1f}" stroke="#475569" stroke-width="1.2" />')
            parts.append(f'<line x1="{x_center - cap_half:.1f}" y1="{error_top_y:.1f}" x2="{x_center + cap_half:.1f}" y2="{error_top_y:.1f}" stroke="#475569" stroke-width="1.2" />')
        label_x = x_slot + slot_width / 2
        label_y = height - 24
        parts.append(
            f'<text x="{label_x:.1f}" y="{label_y}" font-size="{label_font}" fill="#475569" text-anchor="end" transform="rotate(-42 {label_x:.1f} {label_y})">{escape(label)}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def _build_horizontal_bar_chart_svg(
    title: str,
    labels: Sequence[object],
    values: Sequence[object],
    *,
    color: str = "#6E1222",
    width: int = 860,
    height: int = 340,
    formatter: Callable[[object], str] = _fmt_int,
) -> str:
    clean_labels = [_fmt_text(label) for label in labels]
    clean_values = _clean_series(values)
    if not clean_labels or not clean_values or max(clean_values, default=0) <= 0:
        return _empty_svg(title, "Geen data beschikbaar.", width=width, height=height)

    chart_max = _nice_max(max(clean_values))
    margin_left, margin_right, margin_top, margin_bottom = 170, 40, 46, 28
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    row_height = plot_height / max(1, len(clean_labels))
    bar_height = max(14, min(22, row_height * 0.62))

    parts: list[str] = [
        f'<svg class="chart-svg" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{escape(title)}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="16" fill="#F8FAFC" stroke="#D7DEE8" />',
        f'<text x="24" y="36" font-size="22" font-weight="700" fill="#0B1020">{escape(title)}</text>',
    ]

    for index, (label, value) in enumerate(zip(clean_labels, clean_values)):
        y = margin_top + row_height * index + (row_height - bar_height) / 2
        width_value = 0 if chart_max <= 0 else (value / chart_max) * plot_width
        parts.append(f'<text x="{margin_left - 14}" y="{y + bar_height * 0.72:.1f}" text-anchor="end" font-size="10" fill="#334155">{escape(label)}</text>')
        parts.append(f'<rect x="{margin_left}" y="{y:.1f}" width="{plot_width:.1f}" height="{bar_height:.1f}" rx="5" fill="#EFF3F9" />')
        parts.append(f'<rect x="{margin_left}" y="{y:.1f}" width="{width_value:.1f}" height="{bar_height:.1f}" rx="5" fill="{color}" />')
        parts.append(f'<text x="{margin_left + width_value + 8:.1f}" y="{y + bar_height * 0.72:.1f}" font-size="10" fill="#0F172A">{escape(formatter(value))}</text>')

    parts.append("</svg>")
    return "".join(parts)


def _build_share_chart_svg(
    title: str,
    labels: Sequence[object],
    values: Sequence[object],
    *,
    width: int = 860,
    height: int = 220,
) -> str:
    clean_labels = [_fmt_text(label) for label in labels]
    clean_values = _clean_series(values)
    total = sum(clean_values)
    if not clean_labels or total <= 0:
        return _empty_svg(title, "Geen data beschikbaar.", width=width, height=height)

    parts: list[str] = [
        f'<svg class="chart-svg" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{escape(title)}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="16" fill="#F8FAFC" stroke="#D7DEE8" />',
        f'<text x="24" y="36" font-size="22" font-weight="700" fill="#0B1020">{escape(title)}</text>',
    ]

    bar_x, bar_y, bar_width, bar_height = 28, 66, width - 56, 28
    current_x = bar_x
    for index, (label, value) in enumerate(zip(clean_labels, clean_values)):
        color = SVG_COLORS[index % len(SVG_COLORS)]
        segment_width = (value / total) * bar_width
        parts.append(f'<rect x="{current_x:.1f}" y="{bar_y}" width="{segment_width:.1f}" height="{bar_height}" fill="{color}" rx="8" />')
        current_x += segment_width

    row_y = 124
    for index, (label, value) in enumerate(zip(clean_labels, clean_values)):
        color = SVG_COLORS[index % len(SVG_COLORS)]
        percentage = (value / total) * 100
        col = index % 2
        row = index // 2
        x = 30 + col * ((width - 60) / 2)
        y = row_y + row * 28
        parts.append(f'<rect x="{x}" y="{y - 10}" width="12" height="12" rx="3" fill="{color}" />')
        parts.append(
            f'<text x="{x + 20}" y="{y}" font-size="10" fill="#334155">{escape(label)}: {escape(_fmt_distance(value))} ({escape(_fmt_dec(percentage, 1))}%)</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def _table_payload(
    title: str,
    subtitle: str,
    dataframe: pd.DataFrame | None,
    columns: Sequence[tuple[str, str, Callable[[object], str] | None]],
    *,
    empty_message: str,
) -> dict[str, Any]:
    headers = [label for _, label, _ in columns]
    if dataframe is None or dataframe.empty:
        return {
            "title": title,
            "subtitle": subtitle,
            "headers": headers,
            "rows": [],
            "empty_message": empty_message,
        }

    rows: list[list[str]] = []
    for _, row in dataframe.iterrows():
        rows.append(
            [
                formatter(row.get(column)) if formatter else _fmt_text(row.get(column))
                for column, _, formatter in columns
            ]
        )
    return {
        "title": title,
        "subtitle": subtitle,
        "headers": headers,
        "rows": rows,
        "empty_message": empty_message,
    }


def _week_badges(summary: dict[str, object], monitoring_summary: dict[str, object]) -> list[str]:
    badges = [
        f"{_fmt_int(summary.get('active_days'))} actieve dagen",
        f"{_fmt_int(summary.get('training_sessions'))} trainingen",
        f"{_fmt_int(summary.get('match_sessions'))} matchsessies",
        f"{_fmt_int(summary.get('player_sessions'))} player sessions",
    ]
    if pd.notna(summary.get("td_vs_prev")):
        badges.append(f"TD vs vorige 4 weken: {_fmt_dec(summary.get('td_vs_prev'), 1)}%")
    if pd.notna(summary.get("hsr_vs_prev")):
        badges.append(f"HSR vs vorige 4 weken: {_fmt_dec(summary.get('hsr_vs_prev'), 1)}%")
    if pd.notna(monitoring_summary.get("avg_rpe")):
        badges.append(f"Gem. RPE: {_fmt_dec(monitoring_summary.get('avg_rpe'), 1)}")
    return badges


def _fmt_percent(value: object, decimals: int = 1, *, signed: bool = False) -> str:
    if pd.isna(value):
        return "--"
    numeric = float(value)
    prefix = "+" if signed and numeric > 0 else ""
    return f"{prefix}{_fmt_dec(numeric, decimals)}%"


def _weekday_label(value: object) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return _fmt_text(value)
    weekdays = ["Ma", "Di", "Wo", "Do", "Vr", "Za", "Zo"]
    return f"{weekdays[int(ts.weekday())]} {ts:%d/%m}"


def _tone_from_change(value: object) -> str:
    if pd.isna(value):
        return "neutral"
    magnitude = abs(float(value))
    if magnitude >= 20:
        return "alert"
    if magnitude >= 8:
        return "warning"
    return "positive"


def _tone_from_coverage(covered: object, total: object) -> str:
    if pd.isna(covered) or pd.isna(total) or float(total) <= 0:
        return "neutral"
    ratio = float(covered) / float(total)
    if ratio < 0.6:
        return "alert"
    if ratio < 0.8:
        return "warning"
    return "positive"


def _build_week_focus_cards(
    summary: dict[str, object],
    monitoring_summary: dict[str, object],
    day_table: pd.DataFrame,
) -> list[dict[str, str]]:
    cards: list[dict[str, str]] = []
    td_change = summary.get("td_vs_prev")
    hsr_change = summary.get("hsr_vs_prev")
    load_title_parts: list[str] = []
    valid_changes = [value for value in [td_change, hsr_change] if pd.notna(value)]
    if pd.notna(td_change):
        load_title_parts.append(f"TD {_fmt_percent(td_change, signed=True)}")
    if pd.notna(hsr_change):
        load_title_parts.append(f"HSR {_fmt_percent(hsr_change, signed=True)}")
    cards.append(
        {
            "eyebrow": "Load status",
            "title": " | ".join(load_title_parts) if load_title_parts else _fmt_distance(summary.get("total_distance")),
            "body": "Vergeleken met de rolling 4-week referentie voor externe load."
            if load_title_parts
            else "Totale teambelasting binnen de geselecteerde week.",
            "tone": _tone_from_change(max(valid_changes, key=lambda item: abs(float(item)))) if valid_changes else "neutral",
        }
    )

    if isinstance(day_table, pd.DataFrame) and not day_table.empty:
        peak_day = day_table.sort_values("total_distance", ascending=False).iloc[0]
        cards.append(
            {
                "eyebrow": "Peak day",
                "title": f"{_weekday_label(peak_day.get('datum'))} | {_fmt_distance(peak_day.get('total_distance'))}",
                "body": (
                    f"{_fmt_int(peak_day.get('active_players'))} spelers | "
                    f"{_fmt_int(peak_day.get('player_sessions'))} sessies | "
                    f"max {_fmt_speed(peak_day.get('max_speed'))}"
                ),
                "tone": "accent",
            }
        )
    else:
        cards.append(
            {
                "eyebrow": "Peak day",
                "title": "--",
                "body": "Geen dagniveau-data beschikbaar voor deze week.",
                "tone": "neutral",
            }
        )

    cards.append(
        {
            "eyebrow": "Speed activation",
            "title": f"{_fmt_int(summary.get('speed_exposures'))} exposures",
            "body": f"{_fmt_int(summary.get('sprints'))} sprints | top speed {_fmt_speed(summary.get('top_speed'))}",
            "tone": "positive" if pd.notna(summary.get("speed_exposures")) and float(summary.get("speed_exposures")) > 0 else "neutral",
        }
    )

    cards.append(
        {
            "eyebrow": "Monitoring coverage",
            "title": (
                f"{_fmt_int(monitoring_summary.get('wellness_players'))}/{_fmt_int(summary.get('active_players'))} wellness | "
                f"{_fmt_int(monitoring_summary.get('rpe_players'))}/{_fmt_int(summary.get('active_players'))} RPE"
            ),
            "body": (
                f"Readiness {_fmt_dec(monitoring_summary.get('readiness_avg'), 1)} | "
                f"Avg RPE {_fmt_dec(monitoring_summary.get('avg_rpe'), 1)}"
            ),
            "tone": _tone_from_coverage(
                min(
                    float(monitoring_summary.get("wellness_players") or 0),
                    float(monitoring_summary.get("rpe_players") or 0),
                ),
                summary.get("active_players"),
            ),
        }
    )
    return cards


def _build_week_day_cards(day_table: pd.DataFrame) -> list[dict[str, object]]:
    if not isinstance(day_table, pd.DataFrame) or day_table.empty:
        return []
    peak_distance = pd.to_numeric(day_table.get("total_distance"), errors="coerce").max()
    cards: list[dict[str, object]] = []
    for _, row in day_table.sort_values("datum").iterrows():
        total_distance = row.get("total_distance")
        tone = "accent" if pd.notna(total_distance) and pd.notna(peak_distance) and float(total_distance) == float(peak_distance) else "neutral"
        cards.append(
            {
                "label": _weekday_label(row.get("datum")),
                "value": _fmt_distance(total_distance),
                "subvalue": f"{_fmt_distance(row.get('distance_per_player'))} per speler",
                "meta": f"{_fmt_int(row.get('active_players'))} spelers | {_fmt_int(row.get('player_sessions'))} sessies",
                "stats": [
                    {"label": "HSR / HSD", "value": _fmt_distance(row.get("hsr_hsd"))},
                    {"label": "Sprints", "value": _fmt_int(row.get("sprints"))},
                    {"label": "Exposures", "value": _fmt_int(row.get("speed_exposures"))},
                    {"label": "Max speed", "value": _fmt_speed(row.get("max_speed"))},
                ],
                "tone": tone,
            }
        )
    return cards


def _build_week_leader_cards(player_table: pd.DataFrame) -> list[dict[str, str]]:
    if not isinstance(player_table, pd.DataFrame) or player_table.empty:
        return []

    leader_specs: list[tuple[str, str, Callable[[object], str], Callable[[pd.Series], str]]] = [
        ("TD leader", "total_distance", _fmt_distance, lambda row: f"{_fmt_int(row.get('sessions'))} sessies"),
        ("HSR leader", "hsr_hsd", _fmt_distance, lambda row: f"{_fmt_int(row.get('sessions'))} sessies"),
        ("Sprint leader", "sprints", _fmt_int, lambda row: _fmt_distance(row.get("total_distance"))),
        ("Top speed", "max_speed", _fmt_speed, lambda row: _fmt_distance(row.get("hsr_hsd"))),
        ("Intensity", "distance_per_min", lambda value: f"{_fmt_dec(value, 1)} m/min", lambda row: _fmt_distance(row.get("total_distance"))),
    ]
    cards: list[dict[str, str]] = []
    for label, column, formatter, foot_factory in leader_specs:
        ranked = player_table.dropna(subset=[column]).sort_values(column, ascending=False)
        if ranked.empty:
            continue
        top_row = ranked.iloc[0]
        cards.append(
            {
                "label": label,
                "player": _fmt_text(top_row.get("player_name")),
                "value": formatter(top_row.get(column)),
                "foot": foot_factory(top_row),
            }
        )
    return cards


def build_week_report_html_pdf_bytes(
    *,
    week_label: str,
    iso_label: str,
    summary: dict[str, object],
    monitoring_summary: dict[str, object],
    day_table: pd.DataFrame,
    type_table: pd.DataFrame,
    player_table: pd.DataFrame,
    monitoring_day_table: pd.DataFrame,
    notes: Iterable[str],
    day_stats: pd.DataFrame | None = None,
    zone_df: pd.DataFrame | None = None,
    rpe_session_day_table: pd.DataFrame | None = None,
    monitoring_player_table: pd.DataFrame | None = None,
) -> bytes:
    top_players = (
        player_table.sort_values("total_distance", ascending=False)
        .head(12)
        .assign(distance_per_min=lambda frame: pd.to_numeric(frame.get("distance_per_min"), errors="coerce"))
        if isinstance(player_table, pd.DataFrame) and not player_table.empty
        else pd.DataFrame()
    )
    hsr_leaders = (
        player_table.sort_values("hsr_hsd", ascending=False)
        .head(12)
        if isinstance(player_table, pd.DataFrame) and not player_table.empty
        else pd.DataFrame()
    )
    sprint_leaders = (
        player_table.sort_values("sprints", ascending=False)
        .head(12)
        if isinstance(player_table, pd.DataFrame) and not player_table.empty
        else pd.DataFrame()
    )

    monitoring_timeline = monitoring_day_table.copy() if isinstance(monitoring_day_table, pd.DataFrame) else pd.DataFrame()
    if not monitoring_timeline.empty and "readiness_score" not in monitoring_timeline.columns:
        monitoring_timeline["readiness_score"] = pd.NA
    rpe_session_timeline = rpe_session_day_table.copy() if isinstance(rpe_session_day_table, pd.DataFrame) else pd.DataFrame()
    if not rpe_session_timeline.empty:
        rpe_session_timeline["axis_label"] = rpe_session_timeline.apply(
            lambda row: f"{_fmt_text(row.get('label'))} S{_fmt_int(row.get('session_index'))}",
            axis=1,
        )
    squad_spread = day_stats.copy() if isinstance(day_stats, pd.DataFrame) else pd.DataFrame()
    monitoring_watchlist = monitoring_player_table.copy() if isinstance(monitoring_player_table, pd.DataFrame) else pd.DataFrame()
    if not monitoring_watchlist.empty:
        monitoring_watchlist = monitoring_watchlist.sort_values(
            ["readiness_score", "avg_rpe", "player_name"],
            ascending=[True, False, True],
            na_position="last",
        ).head(12)

    header_meta = [
        {"label": "Week", "value": week_label, "foot": iso_label},
        {
            "label": "Activity",
            "value": f"{_fmt_int(summary.get('active_days'))} dagen",
            "foot": f"{_fmt_int(summary.get('player_sessions'))} player sessions",
        },
        {
            "label": "Monitoring",
            "value": f"{_fmt_int(monitoring_summary.get('wellness_entries'))} / {_fmt_int(monitoring_summary.get('rpe_entries'))}",
            "foot": "Wellness / RPE entries",
        },
    ]
    monitoring_cards = [
        {"label": "Readiness Avg", "value": _fmt_dec(monitoring_summary.get("readiness_avg"), 1), "foot": "Teamgemiddelde over alle monitoringdagen"},
        {"label": "Avg RPE", "value": _fmt_dec(monitoring_summary.get("avg_rpe"), 1), "foot": "Gemiddelde interne load binnen de week"},
        {"label": "Wellness Entries", "value": _fmt_int(monitoring_summary.get("wellness_entries")), "foot": f"{_fmt_int(monitoring_summary.get('wellness_players'))} spelers met input"},
        {"label": "RPE Entries", "value": _fmt_int(monitoring_summary.get("rpe_entries")), "foot": f"{_fmt_int(monitoring_summary.get('rpe_players'))} spelers met input"},
    ]

    context = {
        "document_title": f"Week Report | {week_label}",
        "report_title": "Week Report",
        "report_kicker": "MVV Maastricht | Reports | Team Week Overview",
        "report_subtitle": f"{week_label} | {iso_label}",
        "report_description": "Compact staffoverzicht voor trainingssturing, wedstrijdbelasting en monitoring follow-up binnen dezelfde weekselectie.",
        "logo_src": LOGO_SRC,
        "report_header_meta": header_meta,
        "badges": _week_badges(summary, monitoring_summary),
        "cards": [
            {"label": "Total Distance", "value": _fmt_distance(summary.get("total_distance")), "foot": "Opgetelde teamload in de week"},
            {"label": "HSR / HSD", "value": _fmt_distance(summary.get("hsr_hsd")), "foot": "Sprint plus high sprint distance"},
            {"label": "Dist / Player", "value": _fmt_distance(summary.get("dist_per_player")), "foot": "Teamload gedeeld door actieve spelers"},
            {"label": "Sprints", "value": _fmt_int(summary.get("sprints")), "foot": "Totale sprintacties in deze week"},
            {"label": "Top Speed", "value": _fmt_speed(summary.get("top_speed")), "foot": "Hoogste gemeten snelheid"},
            {"label": "Speed Exposures", "value": _fmt_int(summary.get("speed_exposures")), "foot": "Sessies op 90% van seizoenstop"},
        ],
        "focus_cards": _build_week_focus_cards(summary, monitoring_summary, day_table),
        "day_cards": _build_week_day_cards(day_table),
        "leader_cards": _build_week_leader_cards(player_table),
        "monitoring_cards": monitoring_cards,
        "chart_sections": [
            {
                "eyebrow": "Load profile",
                "title": "Weekly load rhythm",
                "subtitle": "Dagelijkse teambelasting en high-speed output binnen de geselecteerde microcycle.",
                "panels": [
                    {
                        "svg": _build_vertical_bar_chart_svg(
                            "Daily Team Distance",
                            day_table.get("label", pd.Series(dtype=str)).tolist(),
                            day_table.get("total_distance", pd.Series(dtype=float)).tolist(),
                            color="#6E1222",
                            formatter=_fmt_distance,
                        )
                    },
                    {
                        "svg": _build_vertical_bar_chart_svg(
                            "Daily Team HSR / HSD",
                            day_table.get("label", pd.Series(dtype=str)).tolist(),
                            day_table.get("hsr_hsd", pd.Series(dtype=float)).tolist(),
                            color="#EA3351",
                            formatter=_fmt_distance,
                        )
                    },
                ],
            },
            {
                "eyebrow": "Squad spread",
                "title": "Average player load +/- SD",
                "subtitle": "Dagelijkse gemiddelde spelerbelasting met spreiding binnen de selectie.",
                "page_break": True,
                "panels": [
                    {
                        "svg": _build_error_bar_chart_svg(
                            "Player Avg Total Distance +/- SD",
                            squad_spread.get("label", pd.Series(dtype=str)).tolist(),
                            squad_spread.get("total_distance_mean", pd.Series(dtype=float)).tolist(),
                            squad_spread.get("total_distance_std", pd.Series(dtype=float)).tolist(),
                            color="#6E1222",
                            formatter=_fmt_distance,
                        )
                    },
                    {
                        "svg": _build_error_bar_chart_svg(
                            "Player Avg HSR / HSD +/- SD",
                            squad_spread.get("label", pd.Series(dtype=str)).tolist(),
                            squad_spread.get("hsr_hsd_mean", pd.Series(dtype=float)).tolist(),
                            squad_spread.get("hsr_hsd_std", pd.Series(dtype=float)).tolist(),
                            color="#EA3351",
                            formatter=_fmt_distance,
                        )
                    },
                ],
            },
            {
                "eyebrow": "Squad spread",
                "title": "Explosive outputs +/- SD",
                "subtitle": "Acceleraties, deceleraties en sprintgemiddelden per speler per dag.",
                "panels": [
                    {
                        "svg": _build_grouped_error_bar_chart_svg(
                            "Player Avg Accel / Decel +/- SD",
                            squad_spread.get("label", pd.Series(dtype=str)).tolist(),
                            [
                                {
                                    "label": "Accelerations",
                                    "color": "#6E1222",
                                    "values": squad_spread.get("total_accelerations_mean", pd.Series(dtype=float)).tolist(),
                                    "errors": squad_spread.get("total_accelerations_std", pd.Series(dtype=float)).tolist(),
                                },
                                {
                                    "label": "Decelerations",
                                    "color": "#EA3351",
                                    "values": squad_spread.get("total_decelerations_mean", pd.Series(dtype=float)).tolist(),
                                    "errors": squad_spread.get("total_decelerations_std", pd.Series(dtype=float)).tolist(),
                                },
                            ],
                        )
                    },
                    {
                        "svg": _build_error_bar_chart_svg(
                            "Player Avg Sprints +/- SD",
                            squad_spread.get("label", pd.Series(dtype=str)).tolist(),
                            squad_spread.get("sprints_mean", pd.Series(dtype=float)).tolist(),
                            squad_spread.get("sprints_std", pd.Series(dtype=float)).tolist(),
                            color="#C8102E",
                            formatter=lambda value: _fmt_dec(value, 1),
                        )
                    },
                ],
            },
            {
                "eyebrow": "Speed profile",
                "title": "Activation and zone distribution",
                "subtitle": "Speed exposures per dag en verdeling van de weekafstand over de locomotion zones.",
                "panels": [
                    {
                        "svg": _build_vertical_bar_chart_svg(
                            "Daily Speed Exposures",
                            day_table.get("label", pd.Series(dtype=str)).tolist(),
                            day_table.get("speed_exposures", pd.Series(dtype=float)).tolist(),
                            color="#C8102E",
                            formatter=_fmt_int,
                        )
                    },
                    {
                        "svg": _build_share_chart_svg(
                            "Distance Zone Share",
                            zone_df.get("zone", pd.Series(dtype=str)).tolist() if isinstance(zone_df, pd.DataFrame) else [],
                            zone_df.get("value", pd.Series(dtype=float)).tolist() if isinstance(zone_df, pd.DataFrame) else [],
                        )
                    },
                ],
            },
            {
                "eyebrow": "Monitoring",
                "title": "Daily wellness profile +/- SD",
                "subtitle": "Fysieke en mentale welzijnsindicatoren per dag met standaarddeviatie.",
                "page_break": True,
                "panels": [
                    {
                        "svg": _build_grouped_error_bar_chart_svg(
                            "Physical Wellness +/- SD",
                            monitoring_timeline.get("label", pd.Series(dtype=str)).tolist(),
                            [
                                {
                                    "label": "Muscle",
                                    "color": "#6E1222",
                                    "values": monitoring_timeline.get("muscle_soreness", pd.Series(dtype=float)).tolist(),
                                    "errors": monitoring_timeline.get("muscle_soreness_std", pd.Series(dtype=float)).tolist(),
                                },
                                {
                                    "label": "Fatigue",
                                    "color": "#EA3351",
                                    "values": monitoring_timeline.get("fatigue", pd.Series(dtype=float)).tolist(),
                                    "errors": monitoring_timeline.get("fatigue_std", pd.Series(dtype=float)).tolist(),
                                },
                            ],
                            y_max=10,
                        )
                    },
                    {
                        "svg": _build_grouped_error_bar_chart_svg(
                            "Mental Wellness +/- SD",
                            monitoring_timeline.get("label", pd.Series(dtype=str)).tolist(),
                            [
                                {
                                    "label": "Sleep",
                                    "color": "#2563EB",
                                    "values": monitoring_timeline.get("sleep_quality", pd.Series(dtype=float)).tolist(),
                                    "errors": monitoring_timeline.get("sleep_quality_std", pd.Series(dtype=float)).tolist(),
                                },
                                {
                                    "label": "Stress",
                                    "color": "#0F766E",
                                    "values": monitoring_timeline.get("stress", pd.Series(dtype=float)).tolist(),
                                    "errors": monitoring_timeline.get("stress_std", pd.Series(dtype=float)).tolist(),
                                },
                                {
                                    "label": "Mood",
                                    "color": "#F59E0B",
                                    "values": monitoring_timeline.get("mood", pd.Series(dtype=float)).tolist(),
                                    "errors": monitoring_timeline.get("mood_std", pd.Series(dtype=float)).tolist(),
                                },
                            ],
                            y_max=10,
                        )
                    },
                ],
            },
            {
                "eyebrow": "Player leaders",
                "title": "Top outputs within the squad",
                "subtitle": "Snel overzicht van volume- en high-speed leiders voor de weekevaluatie.",
                "panels": [
                    {
                        "svg": _build_horizontal_bar_chart_svg(
                            "Top Players by Total Distance",
                            top_players.get("player_name", pd.Series(dtype=str)).tolist(),
                            top_players.get("total_distance", pd.Series(dtype=float)).tolist(),
                            color="#6E1222",
                            formatter=_fmt_distance,
                        )
                    },
                    {
                        "svg": _build_horizontal_bar_chart_svg(
                            "Top Players by HSR / HSD",
                            hsr_leaders.get("player_name", pd.Series(dtype=str)).tolist(),
                            hsr_leaders.get("hsr_hsd", pd.Series(dtype=float)).tolist(),
                            color="#C8102E",
                            formatter=_fmt_distance,
                        )
                    },
                ],
            },
            {
                "eyebrow": "Leaders",
                "title": "Sprint and internal load activation",
                "subtitle": "Sprintleiders en sessie-RPE overzicht voor dagen met enkele of dubbele sessies.",
                "panels": [
                    {
                        "svg": _build_horizontal_bar_chart_svg(
                            "Top Players by Sprints",
                            sprint_leaders.get("player_name", pd.Series(dtype=str)).tolist(),
                            sprint_leaders.get("sprints", pd.Series(dtype=float)).tolist(),
                            color="#EA3351",
                            formatter=_fmt_int,
                        )
                    },
                    {
                        "svg": _build_error_bar_chart_svg(
                            "Session RPE +/- SD",
                            rpe_session_timeline.get("axis_label", pd.Series(dtype=str)).tolist(),
                            rpe_session_timeline.get("avg_rpe", pd.Series(dtype=float)).tolist(),
                            rpe_session_timeline.get("avg_rpe_std", pd.Series(dtype=float)).tolist(),
                            color="#EA3351",
                            formatter=lambda value: _fmt_dec(value, 1),
                            y_max=10,
                        )
                    },
                ],
            },
        ],
        "table_sections": [
            {
                "eyebrow": "Week flow",
                "title": "Week at a glance",
                "subtitle": "Dagritme en sessieverdeling voor de volledige teamweek.",
                "tables": [
                    _table_payload(
                        "Week at a glance",
                        "Dagoverzicht met spelers, sessies en externe load.",
                        day_table,
                        [
                            ("label", "Dag", None),
                            ("active_players", "Players", _fmt_int),
                            ("player_sessions", "Sessions", _fmt_int),
                            ("total_distance", "TD", _fmt_distance),
                            ("distance_per_player", "Dist / Player", _fmt_distance),
                            ("hsr_hsd", "HSR / HSD", _fmt_distance),
                            ("sprints", "Sprints", _fmt_int),
                            ("speed_exposures", "Exposures", _fmt_int),
                            ("max_speed", "Top Speed", _fmt_speed),
                        ],
                        empty_message="Geen weekoverzicht beschikbaar.",
                    ),
                    _table_payload(
                        "Training vs Match",
                        "Verdeling van de weekbelasting per sessiecategorie.",
                        type_table,
                        [
                            ("session_category", "Type", None),
                            ("active_players", "Players", _fmt_int),
                            ("player_sessions", "Sessions", _fmt_int),
                            ("total_distance", "TD", _fmt_distance),
                            ("hsr_hsd", "HSR / HSD", _fmt_distance),
                            ("sprints", "Sprints", _fmt_int),
                            ("max_speed", "Top Speed", _fmt_speed),
                        ],
                        empty_message="Geen sessie-indeling beschikbaar.",
                    ),
                ],
            },
            {
                "eyebrow": "Squad output",
                "title": "Leaders and spread",
                "subtitle": "Volumeleiders en spreiding per dag als basis voor staffbespreking.",
                "tables": [
                    _table_payload(
                        "Top Players",
                        "Spelers met de hoogste totale afstand in deze week.",
                        top_players,
                        [
                            ("player_name", "Speler", None),
                            ("sessions", "Sessies", _fmt_int),
                            ("total_distance", "TD", _fmt_distance),
                            ("hsr_hsd", "HSR / HSD", _fmt_distance),
                            ("sprints", "Sprints", _fmt_int),
                            ("distance_per_min", "m/min", lambda value: _fmt_dec(value, 1)),
                            ("max_speed", "Top Speed", _fmt_speed),
                        ],
                        empty_message="Geen spelerssamenvatting beschikbaar.",
                    ),
                    _table_payload(
                        "Squad Spread by Day",
                        "Gemiddelde en spreiding per speler op dagniveau.",
                        squad_spread,
                        [
                            ("label", "Dag", None),
                            ("player_count", "Players", _fmt_int),
                            ("total_distance_mean", "TD mean", _fmt_distance),
                            ("total_distance_std", "TD SD", _fmt_distance),
                            ("hsr_hsd_mean", "HSR mean", _fmt_distance),
                            ("hsr_hsd_std", "HSR SD", _fmt_distance),
                            ("sprints_mean", "Sprint mean", lambda value: _fmt_dec(value, 1)),
                            ("sprints_std", "Sprint SD", lambda value: _fmt_dec(value, 1)),
                            ("distance_per_min_mean", "m/min", lambda value: _fmt_dec(value, 1)),
                        ],
                        empty_message="Geen squad-spread beschikbaar.",
                    ),
                ],
            },
            {
                "eyebrow": "Monitoring detail",
                "title": "Readiness, wellness and RPE",
                "subtitle": "Dag- en spelersniveau voor interne load en welzijnsopvolging.",
                "tables": [
                    _table_payload(
                        "Monitoring Timeline",
                        "Wellness, readiness en RPE door de week heen.",
                        monitoring_timeline,
                        [
                            ("label", "Periode", None),
                            ("muscle_soreness", "Muscle", lambda value: _fmt_dec(value, 1)),
                            ("fatigue", "Fatigue", lambda value: _fmt_dec(value, 1)),
                            ("sleep_quality", "Sleep", lambda value: _fmt_dec(value, 1)),
                            ("stress", "Stress", lambda value: _fmt_dec(value, 1)),
                            ("mood", "Mood", lambda value: _fmt_dec(value, 1)),
                            ("readiness_score", "Readiness", lambda value: _fmt_dec(value, 1)),
                            ("avg_rpe", "Avg RPE", lambda value: _fmt_dec(value, 1)),
                        ],
                        empty_message="Geen monitoringdata beschikbaar.",
                    ),
                    _table_payload(
                        "Monitoring Watchlist",
                        "Spelers met de laagste readiness en hoogste interne load binnen de week.",
                        monitoring_watchlist,
                        [
                            ("player_name", "Speler", None),
                            ("wellness_days", "Wellness Days", _fmt_int),
                            ("rpe_days", "RPE Days", _fmt_int),
                            ("muscle_soreness", "Muscle", lambda value: _fmt_dec(value, 1)),
                            ("fatigue", "Fatigue", lambda value: _fmt_dec(value, 1)),
                            ("sleep_quality", "Sleep", lambda value: _fmt_dec(value, 1)),
                            ("stress", "Stress", lambda value: _fmt_dec(value, 1)),
                            ("mood", "Mood", lambda value: _fmt_dec(value, 1)),
                            ("readiness_score", "Readiness", lambda value: _fmt_dec(value, 1)),
                            ("avg_rpe", "Avg RPE", lambda value: _fmt_dec(value, 1)),
                        ],
                        empty_message="Geen monitoringwatchlist beschikbaar.",
                    ),
                ],
            },
            {
                "eyebrow": "Session detail",
                "title": "Session RPE detail",
                "subtitle": "Interne load uitgesplitst per sessie-index op dagen met meerdere trainingen.",
                "tables": [
                    _table_payload(
                        "RPE Sessions",
                        "Sessie-voor-sessie team-RPE inclusief spreiding en aantal spelers.",
                        rpe_session_timeline,
                        [
                            ("label", "Dag", None),
                            ("session_label", "Sessie", None),
                            ("avg_rpe", "Avg RPE", lambda value: _fmt_dec(value, 1)),
                            ("avg_rpe_std", "RPE SD", lambda value: _fmt_dec(value, 1)),
                            ("rpe_players", "Players", _fmt_int),
                        ],
                        empty_message="Geen sessie-RPE detail beschikbaar.",
                    ),
                ],
            },
        ],
        "notes": list(notes),
    }
    return _render_html_pdf("week_report.html", context)


def _player_badges(player_name: str, scope_label: str, period_label: str, summary: dict[str, object]) -> list[str]:
    badges = [
        f"Speler: {player_name}",
        f"Scope: {scope_label}",
        f"Periode: {period_label}",
        f"{_fmt_int(summary.get('sessions'))} sessies",
        f"{_fmt_int(summary.get('active_days'))} actieve dagen",
    ]
    if pd.notna(summary.get("distance_per_min")):
        badges.append(f"Intensiteit: {_fmt_dec(summary.get('distance_per_min'), 1)} m/min")
    if pd.notna(summary.get("speed_exposures")):
        badges.append(f"Speed exposures: {_fmt_int(summary.get('speed_exposures'))}")
    return badges


def build_player_report_html_pdf_bytes(
    *,
    player_name: str,
    scope_label: str,
    period_label: str,
    summary: dict[str, object],
    monitoring_summary: dict[str, object],
    sessions_df: pd.DataFrame,
    monitoring_group_df: pd.DataFrame,
    notes: Iterable[str],
    period_df: pd.DataFrame | None = None,
    type_table: pd.DataFrame | None = None,
    zone_df: pd.DataFrame | None = None,
    recent_sessions_subtitle: str | None = None,
) -> bytes:
    period_table = period_df.copy() if isinstance(period_df, pd.DataFrame) else pd.DataFrame()
    type_summary = type_table.copy() if isinstance(type_table, pd.DataFrame) else pd.DataFrame()
    zones = zone_df.copy() if isinstance(zone_df, pd.DataFrame) else pd.DataFrame()
    monitoring_rows = monitoring_group_df.copy() if isinstance(monitoring_group_df, pd.DataFrame) else pd.DataFrame()
    recent_sessions = sessions_df.copy() if isinstance(sessions_df, pd.DataFrame) else pd.DataFrame()
    recent_sessions = recent_sessions.head(12).copy() if not recent_sessions.empty else recent_sessions

    workload_values = period_table.get("total_distance", pd.Series(dtype=float)).tolist()
    intensity_values = period_table.get("distance_per_min", pd.Series(dtype=float)).tolist()

    context = {
        "document_title": f"Player Report | {player_name}",
        "report_title": "Player Report",
        "report_kicker": "MVV Maastricht | Reports | HTML report",
        "report_subtitle": f"{player_name} | {scope_label} | {period_label}",
        "report_description": "Nieuwe HTML/CSS-rapportstijl voor individuele speleranalyse op basis van dezelfde GPS-, wellness- en RPE-selectie als de bestaande export.",
        "logo_src": LOGO_SRC,
        "badges": _player_badges(player_name, scope_label, period_label, summary),
        "cards": [
            {"label": "Sessions", "value": _fmt_int(summary.get("sessions")), "foot": "Summary-sessies binnen de huidige scope"},
            {"label": "Active Days", "value": _fmt_int(summary.get("active_days")), "foot": "Dagen met GPS-activiteit"},
            {"label": "Total Distance", "value": _fmt_distance(summary.get("total_distance")), "foot": "Totale afstand in de huidige scope"},
            {"label": "HSR / HSD", "value": _fmt_distance(summary.get("hsr_hsd")), "foot": "Sprint plus high sprint distance"},
            {"label": "Sprints", "value": _fmt_int(summary.get("sprints")), "foot": "Totaal aantal sprintacties"},
            {"label": "Top Speed", "value": _fmt_speed(summary.get("top_speed")), "foot": "Hoogste gemeten snelheid"},
        ],
        "chart_rows": [
            [
                {
                    "svg": _build_vertical_bar_chart_svg(
                        "Workload Trend",
                        period_table.get("label", pd.Series(dtype=str)).tolist(),
                        workload_values,
                        color="#6E1222",
                        formatter=_fmt_distance,
                    )
                },
                {
                    "svg": _build_vertical_bar_chart_svg(
                        "Intensity Trend",
                        period_table.get("label", pd.Series(dtype=str)).tolist(),
                        intensity_values,
                        color="#EA3351",
                        formatter=lambda value: f"{_fmt_dec(value, 1)} m/min",
                    )
                },
            ],
            [
                {
                    "svg": _build_share_chart_svg(
                        "Distance Zone Share",
                        zones.get("zone", pd.Series(dtype=str)).tolist(),
                        zones.get("value", pd.Series(dtype=float)).tolist(),
                    )
                },
                {
                    "svg": _build_horizontal_bar_chart_svg(
                        "Recent Sessions by Distance",
                        recent_sessions.get("datum_label", pd.Series(dtype=str)).tolist(),
                        recent_sessions.get("total_distance", pd.Series(dtype=float)).tolist(),
                        color="#C8102E",
                        formatter=_fmt_distance,
                    )
                },
            ],
            [
                {
                    "svg": _build_grouped_bar_chart_svg(
                        "Physical Wellness",
                        monitoring_rows.get("label", pd.Series(dtype=str)).tolist(),
                        [
                            {"label": "Muscle", "color": "#6E1222", "values": monitoring_rows.get("muscle_soreness", pd.Series(dtype=float)).tolist()},
                            {"label": "Fatigue", "color": "#EA3351", "values": monitoring_rows.get("fatigue", pd.Series(dtype=float)).tolist()},
                        ],
                        y_max=10,
                    )
                },
                {
                    "svg": _build_grouped_bar_chart_svg(
                        "Mental Wellness",
                        monitoring_rows.get("label", pd.Series(dtype=str)).tolist(),
                        [
                            {"label": "Sleep", "color": "#2563EB", "values": monitoring_rows.get("sleep_quality", pd.Series(dtype=float)).tolist()},
                            {"label": "Stress", "color": "#0F766E", "values": monitoring_rows.get("stress", pd.Series(dtype=float)).tolist()},
                            {"label": "Mood", "color": "#F59E0B", "values": monitoring_rows.get("mood", pd.Series(dtype=float)).tolist()},
                        ],
                        y_max=10,
                    )
                },
            ],
            [
                {
                    "svg": _build_vertical_bar_chart_svg(
                        "Average RPE",
                        monitoring_rows.get("label", pd.Series(dtype=str)).tolist(),
                        monitoring_rows.get("avg_rpe", pd.Series(dtype=float)).tolist(),
                        color="#EA3351",
                        formatter=lambda value: _fmt_dec(value, 1),
                        y_max=10,
                    )
                }
            ],
        ],
        "table_rows": [
            [
                _table_payload(
                    "Period Table",
                    "GPS-uitkomsten per dag of week binnen de gekozen scope.",
                    period_table,
                    [
                        ("label", "Periode", None),
                        ("sessions", "Sessies", _fmt_int),
                        ("total_distance", "Distance", _fmt_distance),
                        ("hsr_hsd", "HSR / HSD", _fmt_distance),
                        ("number_of_sprints", "Sprints", _fmt_int),
                        ("distance_per_min", "m/min", lambda value: _fmt_dec(value, 1)),
                        ("max_speed", "Top Speed", _fmt_speed),
                    ],
                    empty_message="Geen periodetabel beschikbaar.",
                ),
                _table_payload(
                    "Training vs Match",
                    "Vergelijking tussen trainings- en wedstrijdbelasting.",
                    type_summary,
                    [
                        ("session_category", "Type", None),
                        ("sessions", "Sessies", _fmt_int),
                        ("total_distance", "Distance", _fmt_distance),
                        ("hsr_hsd", "HSR / HSD", _fmt_distance),
                        ("sprints", "Sprints", _fmt_int),
                        ("distance_per_min", "m/min", lambda value: _fmt_dec(value, 1)),
                        ("max_speed", "Top Speed", _fmt_speed),
                    ],
                    empty_message="Geen trainings- versus matchdata beschikbaar.",
                ),
            ],
            [
                _table_payload(
                    "Recent Sessions",
                    recent_sessions_subtitle or "Laatste sessies binnen de huidige selectie.",
                    recent_sessions,
                    [
                        ("datum_label", "Datum", None),
                        ("type", "Type", None),
                        ("event", "Event", None),
                        ("total_distance", "Distance", _fmt_distance),
                        ("hsr_hsd", "HSR / HSD", _fmt_distance),
                        ("number_of_sprints", "Sprints", _fmt_int),
                        ("duration", "Duur", _fmt_minutes),
                        ("max_speed", "Top Speed", _fmt_speed),
                    ],
                    empty_message="Geen recente sessies beschikbaar.",
                ),
                _table_payload(
                    "Monitoring Timeline",
                    "Wellness, readiness en RPE over de gekozen spelerperiode.",
                    monitoring_rows,
                    [
                        ("label", "Periode", None),
                        ("muscle_soreness", "Muscle", lambda value: _fmt_dec(value, 1)),
                        ("fatigue", "Fatigue", lambda value: _fmt_dec(value, 1)),
                        ("sleep_quality", "Sleep", lambda value: _fmt_dec(value, 1)),
                        ("stress", "Stress", lambda value: _fmt_dec(value, 1)),
                        ("mood", "Mood", lambda value: _fmt_dec(value, 1)),
                        ("readiness_score", "Readiness", lambda value: _fmt_dec(value, 1)),
                        ("avg_rpe", "Avg RPE", lambda value: _fmt_dec(value, 1)),
                    ],
                    empty_message="Geen monitoringdata beschikbaar.",
                ),
            ],
        ],
        "notes": list(notes),
    }
    return _render_html_pdf("player_report.html", context)
