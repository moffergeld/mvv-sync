from __future__ import annotations

import math
from html import escape
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import pandas as pd

from pages.Subscripts.mvv_branding import TEAM_LOGO

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
CSS_PATH = BASE_DIR / "static" / "css" / "report.css"
LOGO_SRC = TEAM_LOGO.relative_to(BASE_DIR).as_posix() if TEAM_LOGO.exists() else ""

SVG_COLORS = ["#C8102E", "#6E1222", "#EA3351", "#F59E0B", "#2563EB", "#0F766E"]
ZONE_SPECS = [
    ("Walking", "walking", "#F5D2D8"),
    ("Jogging", "jogging", "#F1A4B5"),
    ("Running", "running", "#E97A93"),
    ("Sprint", "sprint", "#D92B4D"),
    ("High Sprint", "high_sprint", "#6E1222"),
]
ZONE_COLOR_LOOKUP = {label: color for label, _, color in ZONE_SPECS}

HTML_RUNTIME_UNAVAILABLE_MESSAGE = (
    "PDF-export voor dit rapport is op deze server nog niet beschikbaar. "
    "De WeasyPrint runtime mist nog Linux-systeembibliotheken. "
    "Na redeploy met de packages uit packages.txt moet deze export weer werken."
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
    buffer = BytesIO()
    html_cls(string=html, base_url=str(BASE_DIR)).write_pdf(
        target=buffer,
        stylesheets=[css_cls(filename=str(CSS_PATH), base_url=str(BASE_DIR))],
    )
    return buffer.getvalue()


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


def _fmt_distance_km(value: object) -> str:
    if pd.isna(value):
        return "--"
    return f"{_fmt_dec(float(value) / 1000.0, 1)} km"


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


def _chart_id(title: str, suffix: str) -> str:
    base = sum((index + 1) * ord(char) for index, char in enumerate(str(title)))
    return f"mvv-{suffix}-{base}"


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    normalized = str(value).strip().lstrip("#")
    if len(normalized) != 6:
        return (200, 16, 46)
    return tuple(int(normalized[index : index + 2], 16) for index in (0, 2, 4))


def _blend_hex(color: str, blend_with: str, ratio: float) -> str:
    ratio = max(0.0, min(1.0, ratio))
    r1, g1, b1 = _hex_to_rgb(color)
    r2, g2, b2 = _hex_to_rgb(blend_with)
    red = round(r1 + (r2 - r1) * ratio)
    green = round(g1 + (g2 - g1) * ratio)
    blue = round(b1 + (b2 - b1) * ratio)
    return f"#{red:02X}{green:02X}{blue:02X}"


def _alpha_hex(color: str, alpha: float) -> str:
    red, green, blue = _hex_to_rgb(color)
    return f"rgba({red}, {green}, {blue}, {max(0.0, min(1.0, alpha)):.2f})"


def _series_path(points: Sequence[tuple[float, float]]) -> str:
    if not points:
        return ""
    return " ".join(
        [f"M {points[0][0]:.1f} {points[0][1]:.1f}"]
        + [f"L {x:.1f} {y:.1f}" for x, y in points[1:]]
    )


def _empty_svg(title: str, message: str, *, width: int = 860, height: int = 190) -> str:
    return f"""
    <svg class="chart-svg" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{escape(title)}">
      <text x="8" y="20" font-size="14" font-weight="700" fill="#0B1020">{escape(title)}</text>
      <text x="{width/2:.0f}" y="{height/2:.0f}" text-anchor="middle" font-size="12" fill="#64748B">{escape(message)}</text>
    </svg>
    """.strip()


def _build_vertical_bar_chart_svg(
    title: str,
    labels: Sequence[object],
    values: Sequence[object],
    *,
    color: str = "#C8102E",
    width: int = 860,
    height: int = 228,
    y_max: float | None = None,
    formatter: Callable[[object], str] = _fmt_int,
) -> str:
    clean_labels = [_fmt_text(label) for label in labels]
    clean_values = _clean_series(values)
    if not clean_labels or not clean_values or max(clean_values, default=0) <= 0:
        return _empty_svg(title, "Geen data beschikbaar.", width=width, height=height)

    chart_max = y_max if y_max is not None else _nice_max(max(clean_values))
    margin_left, margin_right, margin_top, margin_bottom = 52, 22, 78, 42
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    slot_width = plot_width / max(1, len(clean_values))
    label_font = 9 if len(clean_labels) > 8 else 10
    grid_lines = 4
    panel_id = _chart_id(title, "trend-panel")
    plot_id = _chart_id(title, "trend-plot")
    area_id = _chart_id(title, "trend-area")
    accent_id = _chart_id(title, "trend-accent")
    line_shadow_id = _chart_id(title, "trend-line-shadow")
    point_shadow_id = _chart_id(title, "trend-point-shadow")
    avg_value = sum(clean_values) / max(1, len(clean_values))
    peak_value = max(clean_values, default=0.0)
    peak_indices = {idx for idx, value in enumerate(clean_values) if value == peak_value}
    last_value = clean_values[-1] if clean_values else 0.0
    line_color = _blend_hex(color, "#102033", 0.18)
    line_points = [
        (
            margin_left + slot_width * index + slot_width / 2,
            margin_top + plot_height - (0 if chart_max <= 0 else (value / chart_max) * plot_height),
        )
        for index, value in enumerate(clean_values)
    ]

    parts: list[str] = [
        f'<svg class="chart-svg" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{escape(title)}">',
        "<defs>",
        f'<linearGradient id="{panel_id}" x1="0" y1="0" x2="1" y2="1">',
        '<stop offset="0%" stop-color="#FFFFFF" />',
        '<stop offset="100%" stop-color="#F6F9FD" />',
        "</linearGradient>",
        f'<linearGradient id="{plot_id}" x1="0" y1="0" x2="0" y2="1">',
        '<stop offset="0%" stop-color="#FBFDFF" />',
        '<stop offset="100%" stop-color="#F4F7FB" />',
        "</linearGradient>",
        f'<linearGradient id="{area_id}" x1="0" y1="0" x2="0" y2="1">',
        f'<stop offset="0%" stop-color="{_alpha_hex(color, 0.34)}" />',
        f'<stop offset="58%" stop-color="{_alpha_hex(color, 0.14)}" />',
        f'<stop offset="100%" stop-color="{_alpha_hex(color, 0.02)}" />',
        "</linearGradient>",
        f'<linearGradient id="{accent_id}" x1="0" y1="0" x2="1" y2="0">',
        '<stop offset="0%" stop-color="#EFC7D0" />',
        f'<stop offset="100%" stop-color="{color}" />',
        "</linearGradient>",
        f'<filter id="{line_shadow_id}" x="-10%" y="-10%" width="140%" height="150%">',
        f'<feDropShadow dx="0" dy="4" stdDeviation="4" flood-color="{_alpha_hex(color, 0.18)}" />',
        "</filter>",
        f'<filter id="{point_shadow_id}" x="-20%" y="-20%" width="160%" height="160%">',
        '<feDropShadow dx="0" dy="2" stdDeviation="2.2" flood-color="rgba(15,23,42,0.16)" />',
        "</filter>",
        "</defs>",
        f'<rect x="8" y="10" width="{width - 16}" height="{height - 18}" rx="22" fill="url(#{panel_id})" stroke="#DEE6EF" />',
        f'<rect x="8" y="10" width="{width - 16}" height="6" rx="22" fill="url(#{accent_id})" />',
        f'<text x="24" y="34" font-size="17" font-weight="800" fill="#0F172A">{escape(title)}</text>',
        f'<text x="24" y="50" font-size="9.2" fill="#64748B">Executive trend view met gemiddelde, piek en laatste meetpunt</text>',
    ]

    chip_specs = [
        ("AVG", formatter(avg_value), _alpha_hex(color, 0.10), _alpha_hex(color, 0.22)),
        ("PEAK", formatter(peak_value), _alpha_hex("#F59E0B", 0.12), _alpha_hex("#F59E0B", 0.28)),
        ("LAST", formatter(last_value), _alpha_hex("#0F766E", 0.10), _alpha_hex("#0F766E", 0.22)),
    ]
    chip_x = width - 24
    for label, value_text, fill_color, stroke_color in reversed(chip_specs):
        chip_width = max(92, len(value_text) * 6.2 + 44)
        chip_x -= chip_width
        parts.append(f'<rect x="{chip_x:.1f}" y="20" width="{chip_width:.1f}" height="30" rx="15" fill="{fill_color}" stroke="{stroke_color}" />')
        parts.append(f'<text x="{chip_x + 12:.1f}" y="31" font-size="7.3" font-weight="800" fill="#64748B">{escape(label)}</text>')
        parts.append(f'<text x="{chip_x + 12:.1f}" y="43" font-size="10.2" font-weight="800" fill="#0F172A">{escape(value_text)}</text>')
        chip_x -= 7

    parts.append(
        f'<rect x="{margin_left:.1f}" y="{margin_top - 8:.1f}" width="{plot_width:.1f}" height="{plot_height + 12:.1f}" rx="18" fill="url(#{plot_id})" stroke="#E4EAF1" />'
    )

    for step in range(grid_lines + 1):
        ratio = step / grid_lines
        y = margin_top + plot_height - ratio * plot_height
        axis_value = chart_max * ratio
        if step < grid_lines:
            band_y = margin_top + plot_height - ((step + 1) / grid_lines) * plot_height
            parts.append(
                f'<rect x="{margin_left + 1:.1f}" y="{band_y:.1f}" width="{plot_width - 2:.1f}" height="{plot_height / grid_lines:.1f}" fill="{_alpha_hex(color, 0.022 if step % 2 == 0 else 0.012)}" />'
            )
        parts.append(f'<line x1="{margin_left:.1f}" y1="{y:.1f}" x2="{width - margin_right:.1f}" y2="{y:.1f}" stroke="#DAE3ED" stroke-dasharray="3 6" />')
        parts.append(f'<text x="{margin_left - 10:.1f}" y="{y + 3:.1f}" text-anchor="end" font-size="9" fill="#6B7B93">{escape(_fmt_axis(axis_value))}</text>')

    if len(line_points) >= 2:
        peak_index = max(range(len(clean_values)), key=lambda idx: clean_values[idx])
        peak_center = line_points[peak_index][0]
        parts.append(
            f'<rect x="{peak_center - slot_width * 0.38:.1f}" y="{margin_top:.1f}" width="{slot_width * 0.76:.1f}" height="{plot_height:.1f}" rx="14" fill="{_alpha_hex("#F59E0B", 0.06)}" />'
        )
        area_points = [(line_points[0][0], margin_top + plot_height)] + line_points + [(line_points[-1][0], margin_top + plot_height)]
        parts.append(f'<path d="{_series_path(area_points)} Z" fill="url(#{area_id})" />')
        parts.append(f'<path d="{_series_path(line_points)}" fill="none" stroke="#FFFFFF" stroke-width="6.6" stroke-linecap="round" stroke-linejoin="round" opacity="0.94" />')
        parts.append(
            f'<path d="{_series_path(line_points)}" fill="none" stroke="{line_color}" stroke-width="3.4" stroke-linecap="round" stroke-linejoin="round" filter="url(#{line_shadow_id})" />'
        )

    if avg_value > 0 and chart_max > 0:
        avg_y = margin_top + plot_height - (avg_value / chart_max) * plot_height
        parts.append(
            f'<line x1="{margin_left:.1f}" y1="{avg_y:.1f}" x2="{width - margin_right:.1f}" y2="{avg_y:.1f}" stroke="{_alpha_hex(line_color, 0.7)}" stroke-width="1.6" stroke-dasharray="8 6" />'
        )
        parts.append(
            f'<text x="{width - margin_right - 4:.1f}" y="{avg_y - 6:.1f}" text-anchor="end" font-size="8.7" font-weight="800" fill="{line_color}">Avg {escape(formatter(avg_value))}</text>'
        )

    for index, (label, value) in enumerate(zip(clean_labels, clean_values, strict=False)):
        x_center = margin_left + slot_width * index + slot_width / 2
        y = line_points[index][1]
        parts.append(f'<line x1="{x_center:.1f}" y1="{margin_top + plot_height:.1f}" x2="{x_center:.1f}" y2="{y:.1f}" stroke="{_alpha_hex(color, 0.08)}" stroke-dasharray="2 6" />')
        point_radius = 6.6 if index in peak_indices else 5.2
        point_fill = "#FFF7ED" if index in peak_indices else "#FFFFFF"
        point_stroke = "#D97706" if index in peak_indices else line_color
        parts.append(
            f'<circle cx="{x_center:.1f}" cy="{y:.1f}" r="{point_radius:.1f}" fill="{point_fill}" stroke="{point_stroke}" stroke-width="2.1" filter="url(#{point_shadow_id})" />'
        )
        parts.append(f'<circle cx="{x_center:.1f}" cy="{y:.1f}" r="2.8" fill="{line_color}" />')
        if value > 0:
            parts.append(
                f'<text x="{x_center:.1f}" y="{max(y - 12, margin_top + 10):.1f}" text-anchor="middle" font-size="9" font-weight="800" fill="#0F172A">{escape(formatter(value))}</text>'
            )
        label_y = height - 12
        parts.append(
            f'<text x="{x_center:.1f}" y="{label_y}" font-size="{label_font}" fill="#55657E" text-anchor="end" transform="rotate(-28 {x_center:.1f} {label_y})">{escape(label)}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def _build_grouped_bar_chart_svg(
    title: str,
    labels: Sequence[object],
    series: Sequence[dict[str, Any]],
    *,
    width: int = 860,
    height: int = 224,
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
    margin_left, margin_right, margin_top, margin_bottom = 48, 16, 44, 44
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    slot_width = plot_width / max(1, len(clean_labels))
    series_width = slot_width * 0.72
    bar_width = max(10, min(20, series_width / max(1, len(clean_series))))
    grid_lines = 4
    label_font = 9 if len(clean_labels) > 8 else 10
    shadow_id = _chart_id(title, "group-shadow")
    gradient_ids = [_chart_id(f"{title}-{item['label']}", "group-grad") for item in clean_series]
    panel_id = _chart_id(title, "group-panel")
    series_avg = [
        sum(item["values"]) / len(item["values"]) if item["values"] else 0.0
        for item in clean_series
    ]
    series_points: list[list[tuple[float, float]]] = [[] for _ in clean_series]

    parts: list[str] = [
        f'<svg class="chart-svg" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{escape(title)}">',
        "<defs>",
        f'<linearGradient id="{panel_id}" x1="0" y1="0" x2="1" y2="1">',
        '<stop offset="0%" stop-color="#FFFFFF" />',
        '<stop offset="100%" stop-color="#F6FAFF" />',
        "</linearGradient>",
        f'<filter id="{shadow_id}" x="-10%" y="-10%" width="140%" height="140%">',
        '<feDropShadow dx="0" dy="2" stdDeviation="2.0" flood-color="rgba(15,23,42,0.12)" />',
        "</filter>",
    ]
    for gradient_id, item in zip(gradient_ids, clean_series, strict=False):
        parts.extend(
            [
                f'<linearGradient id="{gradient_id}" x1="0" y1="0" x2="0" y2="1">',
                f'<stop offset="0%" stop-color="{_blend_hex(item["color"], "#FFFFFF", 0.26)}" />',
                f'<stop offset="100%" stop-color="{item["color"]}" />',
                "</linearGradient>",
            ]
        )
    parts.extend(
        [
            "</defs>",
        f'<text x="10" y="22" font-size="15" font-weight="800" fill="#0B1020">{escape(title)}</text>',
            f'<rect x="{margin_left - 10}" y="{margin_top - 8}" width="{plot_width + 20:.1f}" height="{plot_height + 16:.1f}" rx="14" fill="url(#{panel_id})" stroke="#DCE6F2" />',
        ]
    )

    legend_x = 10
    for item in clean_series:
        parts.append(f'<rect x="{legend_x}" y="28" width="30" height="12" rx="6" fill="{_alpha_hex(item["color"], 0.16)}" stroke="{_alpha_hex(item["color"], 0.28)}" />')
        parts.append(f'<rect x="{legend_x + 4}" y="31" width="8" height="6" rx="3" fill="{item["color"]}" />')
        parts.append(f'<text x="{legend_x + 18}" y="37" font-size="9" fill="#475569">{escape(item["label"])}</text>')
        legend_x += max(92, len(item["label"]) * 7 + 30)

    for step in range(grid_lines + 1):
        ratio = step / grid_lines
        y = margin_top + plot_height - ratio * plot_height
        axis_value = chart_max * ratio
        if step < grid_lines:
            band_y = margin_top + plot_height - ((step + 1) / grid_lines) * plot_height
            parts.append(
                f'<rect x="{margin_left:.1f}" y="{band_y:.1f}" width="{plot_width:.1f}" height="{plot_height / grid_lines:.1f}" fill="{_alpha_hex("#3B82F6", 0.022 if step % 2 == 0 else 0.012)}" />'
            )
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#E6EDF5" stroke-dasharray="3 5" />')
        parts.append(f'<text x="{margin_left - 10}" y="{y + 3:.1f}" text-anchor="end" font-size="9" fill="#64748B">{escape(_fmt_axis(axis_value))}</text>')

    for label_index, label in enumerate(clean_labels):
        x_slot = margin_left + slot_width * label_index
        x_start = x_slot + (slot_width - series_width) / 2
        for series_index, item in enumerate(clean_series):
            value = item["values"][label_index] if label_index < len(item["values"]) else 0.0
            bar_height = 0 if chart_max <= 0 else (value / chart_max) * plot_height
            x = x_start + series_index * bar_width
            y = margin_top + plot_height - bar_height
            x_center = x + (bar_width - 2) / 2
            series_points[series_index].append((x_center, y))
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width - 2:.1f}" height="{bar_height:.1f}" rx="4" fill="url(#{gradient_ids[series_index]})" filter="url(#{shadow_id})" />'
            )
        label_x = x_slot + slot_width / 2
        label_y = height - 12
        parts.append(
            f'<text x="{label_x:.1f}" y="{label_y}" font-size="{label_font}" fill="#475569" text-anchor="end" transform="rotate(-28 {label_x:.1f} {label_y})">{escape(label)}</text>'
        )

    for series_index, item in enumerate(clean_series):
        if len(series_points[series_index]) < 2:
            continue
        trend_color = _blend_hex(item["color"], "#274060", 0.45)
        parts.append(
            f'<path d="{_series_path(series_points[series_index])}" fill="none" stroke="{trend_color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" />'
        )
        for x_center, y in series_points[series_index]:
            parts.append(f'<circle cx="{x_center:.1f}" cy="{y:.1f}" r="2.8" fill="#FFFFFF" stroke="{trend_color}" stroke-width="1.4" />')

    avg_chip_x = width - 14
    for item, avg_value in reversed(list(zip(clean_series, series_avg, strict=False))):
        chip_width = max(92, len(item["label"]) * 6.2 + 52)
        avg_chip_x -= chip_width
        parts.append(f'<rect x="{avg_chip_x:.1f}" y="10" width="{chip_width:.1f}" height="20" rx="10" fill="{_alpha_hex(item["color"], 0.10)}" stroke="{_alpha_hex(item["color"], 0.24)}" />')
        parts.append(f'<text x="{avg_chip_x + 10:.1f}" y="23" font-size="8.8" font-weight="700" fill="{_blend_hex(item["color"], "#20324B", 0.34)}">{escape(item["label"])} avg {escape(_fmt_axis(avg_value))}</text>')
        avg_chip_x -= 6

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
    height: int = 224,
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
    margin_left, margin_right, margin_top, margin_bottom = 52, 22, 78, 42
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    slot_width = plot_width / max(1, len(clean_means))
    label_font = 9 if len(clean_labels) > 8 else 10
    grid_lines = 4
    panel_id = _chart_id(title, "band-panel")
    plot_id = _chart_id(title, "band-plot")
    band_id = _chart_id(title, "band-fill")
    accent_id = _chart_id(title, "band-accent")
    line_shadow_id = _chart_id(title, "band-shadow")
    point_shadow_id = _chart_id(title, "band-point-shadow")
    line_color = _blend_hex(color, "#102033", 0.18)
    avg_value = sum(clean_means) / max(1, len(clean_means))
    avg_error = sum(clean_errors) / max(1, len(clean_errors))
    peak_value = max(clean_means, default=0.0)
    peak_indices = {idx for idx, value in enumerate(clean_means) if value == peak_value}
    mean_points: list[tuple[float, float]] = []
    upper_points: list[tuple[float, float]] = []
    lower_points: list[tuple[float, float]] = []

    parts: list[str] = [
        f'<svg class="chart-svg" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{escape(title)}">',
        "<defs>",
        f'<linearGradient id="{panel_id}" x1="0" y1="0" x2="1" y2="1">',
        '<stop offset="0%" stop-color="#FFFFFF" />',
        '<stop offset="100%" stop-color="#F6F9FD" />',
        "</linearGradient>",
        f'<linearGradient id="{plot_id}" x1="0" y1="0" x2="0" y2="1">',
        '<stop offset="0%" stop-color="#FBFDFF" />',
        '<stop offset="100%" stop-color="#F4F7FB" />',
        "</linearGradient>",
        f'<linearGradient id="{band_id}" x1="0" y1="0" x2="0" y2="1">',
        f'<stop offset="0%" stop-color="{_alpha_hex(color, 0.22)}" />',
        f'<stop offset="100%" stop-color="{_alpha_hex(color, 0.04)}" />',
        "</linearGradient>",
        f'<linearGradient id="{accent_id}" x1="0" y1="0" x2="1" y2="0">',
        '<stop offset="0%" stop-color="#EFC7D0" />',
        f'<stop offset="100%" stop-color="{color}" />',
        "</linearGradient>",
        f'<filter id="{line_shadow_id}" x="-10%" y="-10%" width="140%" height="150%">',
        f'<feDropShadow dx="0" dy="4" stdDeviation="4" flood-color="{_alpha_hex(color, 0.16)}" />',
        "</filter>",
        f'<filter id="{point_shadow_id}" x="-20%" y="-20%" width="160%" height="160%">',
        '<feDropShadow dx="0" dy="2" stdDeviation="2.1" flood-color="rgba(15,23,42,0.14)" />',
        "</filter>",
        "</defs>",
        f'<rect x="8" y="10" width="{width - 16}" height="{height - 18}" rx="22" fill="url(#{panel_id})" stroke="#DEE6EF" />',
        f'<rect x="8" y="10" width="{width - 16}" height="6" rx="22" fill="url(#{accent_id})" />',
        f'<text x="24" y="34" font-size="17" font-weight="800" fill="#0F172A">{escape(title)}</text>',
        f'<text x="24" y="50" font-size="9.2" fill="#64748B">Gemiddelde lijn met onzekerheidsband en standaarddeviatie per dag</text>',
    ]

    chip_specs = [
        ("AVG", formatter(avg_value), _alpha_hex(color, 0.10), _alpha_hex(color, 0.22)),
        ("PEAK", formatter(peak_value), _alpha_hex("#F59E0B", 0.12), _alpha_hex("#F59E0B", 0.28)),
        ("AVG SD", formatter(avg_error), _alpha_hex("#5C7697", 0.10), _alpha_hex("#5C7697", 0.22)),
    ]
    chip_x = width - 24
    for label, value_text, fill_color, stroke_color in reversed(chip_specs):
        chip_width = max(96, len(value_text) * 6.2 + 46)
        chip_x -= chip_width
        parts.append(f'<rect x="{chip_x:.1f}" y="20" width="{chip_width:.1f}" height="30" rx="15" fill="{fill_color}" stroke="{stroke_color}" />')
        parts.append(f'<text x="{chip_x + 12:.1f}" y="31" font-size="7.3" font-weight="800" fill="#64748B">{escape(label)}</text>')
        parts.append(f'<text x="{chip_x + 12:.1f}" y="43" font-size="10.1" font-weight="800" fill="#0F172A">{escape(value_text)}</text>')
        chip_x -= 7

    parts.append(
        f'<rect x="{margin_left:.1f}" y="{margin_top - 8:.1f}" width="{plot_width:.1f}" height="{plot_height + 12:.1f}" rx="18" fill="url(#{plot_id})" stroke="#E4EAF1" />'
    )

    for step in range(grid_lines + 1):
        ratio = step / grid_lines
        y = margin_top + plot_height - ratio * plot_height
        axis_value = chart_max * ratio
        if step < grid_lines:
            band_y = margin_top + plot_height - ((step + 1) / grid_lines) * plot_height
            parts.append(
                f'<rect x="{margin_left + 1:.1f}" y="{band_y:.1f}" width="{plot_width - 2:.1f}" height="{plot_height / grid_lines:.1f}" fill="{_alpha_hex(color, 0.02 if step % 2 == 0 else 0.011)}" />'
            )
        parts.append(f'<line x1="{margin_left:.1f}" y1="{y:.1f}" x2="{width - margin_right:.1f}" y2="{y:.1f}" stroke="#DAE3ED" stroke-dasharray="3 6" />')
        parts.append(f'<text x="{margin_left - 10:.1f}" y="{y + 3:.1f}" text-anchor="end" font-size="9" fill="#6B7B93">{escape(_fmt_axis(axis_value))}</text>')

    for index, (label, mean_value, error_value) in enumerate(zip(clean_labels, clean_means, clean_errors, strict=False)):
        x_center = margin_left + slot_width * index + slot_width / 2
        mean_y = margin_top + plot_height - (0 if chart_max <= 0 else (mean_value / chart_max) * plot_height)
        upper_value = min(chart_max, mean_value + error_value)
        lower_value = max(0.0, mean_value - error_value)
        upper_y = margin_top + plot_height - (0 if chart_max <= 0 else (upper_value / chart_max) * plot_height)
        lower_y = margin_top + plot_height - (0 if chart_max <= 0 else (lower_value / chart_max) * plot_height)
        mean_points.append((x_center, mean_y))
        upper_points.append((x_center, upper_y))
        lower_points.append((x_center, lower_y))
        label_y = height - 12
        parts.append(
            f'<text x="{x_center:.1f}" y="{label_y}" font-size="{label_font}" fill="#55657E" text-anchor="end" transform="rotate(-28 {x_center:.1f} {label_y})">{escape(label)}</text>'
        )

    if len(mean_points) >= 2:
        peak_index = max(range(len(clean_means)), key=lambda idx: clean_means[idx])
        peak_center = mean_points[peak_index][0]
        parts.append(
            f'<rect x="{peak_center - slot_width * 0.38:.1f}" y="{margin_top:.1f}" width="{slot_width * 0.76:.1f}" height="{plot_height:.1f}" rx="14" fill="{_alpha_hex("#F59E0B", 0.05)}" />'
        )
        band_points = upper_points + list(reversed(lower_points))
        parts.append(f'<path d="{_series_path(band_points)} Z" fill="url(#{band_id})" />')
        parts.append(f'<path d="{_series_path(mean_points)}" fill="none" stroke="#FFFFFF" stroke-width="6.2" stroke-linecap="round" stroke-linejoin="round" opacity="0.92" />')
        parts.append(
            f'<path d="{_series_path(mean_points)}" fill="none" stroke="{line_color}" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round" filter="url(#{line_shadow_id})" />'
        )

    if avg_value > 0 and chart_max > 0:
        avg_y = margin_top + plot_height - (avg_value / chart_max) * plot_height
        parts.append(
            f'<line x1="{margin_left:.1f}" y1="{avg_y:.1f}" x2="{width - margin_right:.1f}" y2="{avg_y:.1f}" stroke="{_alpha_hex(line_color, 0.72)}" stroke-width="1.6" stroke-dasharray="8 6" />'
        )
        parts.append(
            f'<text x="{width - margin_right - 4:.1f}" y="{avg_y - 6:.1f}" text-anchor="end" font-size="8.7" font-weight="800" fill="{line_color}">Avg {escape(formatter(avg_value))}</text>'
        )

    for index, error_value in enumerate(clean_errors):
        x_center = margin_left + slot_width * index + slot_width / 2
        mean_y = mean_points[index][1]
        upper_y = upper_points[index][1]
        lower_y = lower_points[index][1]
        parts.append(f'<line x1="{x_center:.1f}" y1="{upper_y:.1f}" x2="{x_center:.1f}" y2="{lower_y:.1f}" stroke="{_alpha_hex(line_color, 0.46)}" stroke-width="1.5" />')
        parts.append(f'<line x1="{x_center - 6:.1f}" y1="{upper_y:.1f}" x2="{x_center + 6:.1f}" y2="{upper_y:.1f}" stroke="{_alpha_hex(line_color, 0.46)}" stroke-width="1.5" />')
        parts.append(f'<line x1="{x_center - 6:.1f}" y1="{lower_y:.1f}" x2="{x_center + 6:.1f}" y2="{lower_y:.1f}" stroke="{_alpha_hex(line_color, 0.46)}" stroke-width="1.5" />')
        point_fill = "#FFF7ED" if index in peak_indices else "#FFFFFF"
        point_stroke = "#D97706" if index in peak_indices else line_color
        parts.append(
            f'<circle cx="{x_center:.1f}" cy="{mean_y:.1f}" r="{6.4 if index in peak_indices else 5.0:.1f}" fill="{point_fill}" stroke="{point_stroke}" stroke-width="2.0" filter="url(#{point_shadow_id})" />'
        )
        parts.append(f'<circle cx="{x_center:.1f}" cy="{mean_y:.1f}" r="2.6" fill="{line_color}" />')
        if clean_means[index] > 0:
            parts.append(
                f'<text x="{x_center:.1f}" y="{max(mean_y - 12, margin_top + 10):.1f}" text-anchor="middle" font-size="8.9" font-weight="800" fill="#0F172A">{escape(formatter(clean_means[index]))}</text>'
            )

    parts.append("</svg>")
    return "".join(parts)


def _build_grouped_error_bar_chart_svg(
    title: str,
    labels: Sequence[object],
    series: Sequence[dict[str, Any]],
    *,
    width: int = 860,
    height: int = 232,
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
    margin_left, margin_right, margin_top, margin_bottom = 52, 22, 82, 42
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    slot_width = plot_width / max(1, len(clean_labels))
    label_font = 9 if len(clean_labels) > 8 else 10
    grid_lines = 4
    panel_id = _chart_id(title, "multi-panel")
    plot_id = _chart_id(title, "multi-plot")
    accent_id = _chart_id(title, "multi-accent")
    line_shadow_id = _chart_id(title, "multi-shadow")
    point_shadow_id = _chart_id(title, "multi-point-shadow")
    series_avg = [sum(item["values"]) / len(item["values"]) if item["values"] else 0.0 for item in clean_series]
    series_error_avg = [sum(item["errors"]) / len(item["errors"]) if item["errors"] else 0.0 for item in clean_series]
    series_points: list[list[tuple[float, float]]] = [[] for _ in clean_series]
    series_upper_points: list[list[tuple[float, float]]] = [[] for _ in clean_series]
    series_lower_points: list[list[tuple[float, float]]] = [[] for _ in clean_series]

    parts: list[str] = [
        f'<svg class="chart-svg" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{escape(title)}">',
        "<defs>",
        f'<linearGradient id="{panel_id}" x1="0" y1="0" x2="1" y2="1">',
        '<stop offset="0%" stop-color="#FFFFFF" />',
        '<stop offset="100%" stop-color="#F6F9FD" />',
        "</linearGradient>",
        f'<linearGradient id="{plot_id}" x1="0" y1="0" x2="0" y2="1">',
        '<stop offset="0%" stop-color="#FBFDFF" />',
        '<stop offset="100%" stop-color="#F4F7FB" />',
        "</linearGradient>",
        f'<linearGradient id="{accent_id}" x1="0" y1="0" x2="1" y2="0">',
        '<stop offset="0%" stop-color="#EFC7D0" />',
        '<stop offset="100%" stop-color="#C8102E" />',
        "</linearGradient>",
        f'<filter id="{line_shadow_id}" x="-10%" y="-10%" width="140%" height="150%">',
        '<feDropShadow dx="0" dy="4" stdDeviation="4" flood-color="rgba(15,23,42,0.14)" />',
        "</filter>",
        f'<filter id="{point_shadow_id}" x="-20%" y="-20%" width="160%" height="160%">',
        '<feDropShadow dx="0" dy="2" stdDeviation="2.0" flood-color="rgba(15,23,42,0.12)" />',
        "</filter>",
    ]
    for item in clean_series:
        band_id = _chart_id(f"{title}-{item['label']}", "multi-band")
        parts.extend(
            [
                f'<linearGradient id="{band_id}" x1="0" y1="0" x2="0" y2="1">',
                f'<stop offset="0%" stop-color="{_alpha_hex(item["color"], 0.16)}" />',
                f'<stop offset="100%" stop-color="{_alpha_hex(item["color"], 0.03)}" />',
                "</linearGradient>",
            ]
        )
    parts.extend(
        [
            "</defs>",
            f'<rect x="8" y="10" width="{width - 16}" height="{height - 18}" rx="22" fill="url(#{panel_id})" stroke="#DEE6EF" />',
            f'<rect x="8" y="10" width="{width - 16}" height="6" rx="22" fill="url(#{accent_id})" />',
            f'<text x="24" y="34" font-size="17" font-weight="800" fill="#0F172A">{escape(title)}</text>',
            f'<text x="24" y="50" font-size="9.2" fill="#64748B">Gecombineerde profielplot met meerdere reeksen en standaarddeviatie</text>',
            f'<rect x="{margin_left:.1f}" y="{margin_top - 8:.1f}" width="{plot_width:.1f}" height="{plot_height + 12:.1f}" rx="18" fill="url(#{plot_id})" stroke="#E4EAF1" />',
        ]
    )

    legend_x = 24
    for item in clean_series:
        pill_width = max(86, len(item["label"]) * 7 + 30)
        parts.append(f'<rect x="{legend_x:.1f}" y="58" width="{pill_width:.1f}" height="20" rx="10" fill="{_alpha_hex(item["color"], 0.11)}" stroke="{_alpha_hex(item["color"], 0.24)}" />')
        parts.append(f'<circle cx="{legend_x + 10:.1f}" cy="68" r="4.4" fill="{item["color"]}" />')
        parts.append(f'<text x="{legend_x + 20:.1f}" y="71" font-size="8.8" font-weight="700" fill="#475569">{escape(item["label"])}</text>')
        legend_x += pill_width + 8

    chip_x = width - 24
    for item, avg_value, avg_error in reversed(list(zip(clean_series, series_avg, series_error_avg, strict=False))):
        chip_text = f"{item['label']} {_fmt_axis(avg_value)}"
        chip_width = max(100, len(chip_text) * 6.0 + 28)
        chip_x -= chip_width
        parts.append(f'<rect x="{chip_x:.1f}" y="20" width="{chip_width:.1f}" height="30" rx="15" fill="{_alpha_hex(item["color"], 0.10)}" stroke="{_alpha_hex(item["color"], 0.24)}" />')
        parts.append(f'<text x="{chip_x + 12:.1f}" y="31" font-size="7.1" font-weight="800" fill="#64748B">AVG | SD {_fmt_axis(avg_error)}</text>')
        parts.append(f'<text x="{chip_x + 12:.1f}" y="43" font-size="9.4" font-weight="800" fill="{_blend_hex(item["color"], "#20324B", 0.32)}">{escape(chip_text)}</text>')
        chip_x -= 6

    for step in range(grid_lines + 1):
        ratio = step / grid_lines
        y = margin_top + plot_height - ratio * plot_height
        axis_value = chart_max * ratio
        if step < grid_lines:
            band_y = margin_top + plot_height - ((step + 1) / grid_lines) * plot_height
            parts.append(
                f'<rect x="{margin_left + 1:.1f}" y="{band_y:.1f}" width="{plot_width - 2:.1f}" height="{plot_height / grid_lines:.1f}" fill="{_alpha_hex("#3B82F6", 0.016 if step % 2 == 0 else 0.009)}" />'
            )
        parts.append(f'<line x1="{margin_left:.1f}" y1="{y:.1f}" x2="{width - margin_right:.1f}" y2="{y:.1f}" stroke="#DAE3ED" stroke-dasharray="3 6" />')
        parts.append(f'<text x="{margin_left - 10:.1f}" y="{y + 3:.1f}" text-anchor="end" font-size="9" fill="#6B7B93">{escape(_fmt_axis(axis_value))}</text>')

    for label_index, label in enumerate(clean_labels):
        x_center = margin_left + slot_width * label_index + slot_width / 2
        for series_index, item in enumerate(clean_series):
            value = item["values"][label_index] if label_index < len(item["values"]) else 0.0
            error = item["errors"][label_index] if label_index < len(item["errors"]) else 0.0
            mean_y = margin_top + plot_height - (0 if chart_max <= 0 else (value / chart_max) * plot_height)
            upper_y = margin_top + plot_height - (0 if chart_max <= 0 else (min(chart_max, value + error) / chart_max) * plot_height)
            lower_y = margin_top + plot_height - (0 if chart_max <= 0 else (max(0.0, value - error) / chart_max) * plot_height)
            series_points[series_index].append((x_center, mean_y))
            series_upper_points[series_index].append((x_center, upper_y))
            series_lower_points[series_index].append((x_center, lower_y))
        label_y = height - 12
        parts.append(
            f'<text x="{x_center:.1f}" y="{label_y}" font-size="{label_font}" fill="#55657E" text-anchor="end" transform="rotate(-28 {x_center:.1f} {label_y})">{escape(label)}</text>'
        )

    for series_index, item in enumerate(clean_series):
        if len(series_points[series_index]) < 2:
            continue
        trend_color = _blend_hex(item["color"], "#233247", 0.26)
        band_id = _chart_id(f"{title}-{item['label']}", "multi-band")
        band_points = series_upper_points[series_index] + list(reversed(series_lower_points[series_index]))
        parts.append(f'<path d="{_series_path(band_points)} Z" fill="url(#{band_id})" />')
        parts.append(
            f'<path d="{_series_path(series_points[series_index])}" fill="none" stroke="#FFFFFF" stroke-width="5.2" stroke-linecap="round" stroke-linejoin="round" opacity="0.88" />'
        )
        parts.append(
            f'<path d="{_series_path(series_points[series_index])}" fill="none" stroke="{trend_color}" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round" filter="url(#{line_shadow_id})" />'
        )
        for point_index, (x_center, y) in enumerate(series_points[series_index]):
            parts.append(f'<circle cx="{x_center:.1f}" cy="{y:.1f}" r="3.8" fill="#FFFFFF" stroke="{trend_color}" stroke-width="1.6" filter="url(#{point_shadow_id})" />')
            if point_index == len(series_points[series_index]) - 1:
                last_value = item["values"][-1] if item["values"] else 0.0
                parts.append(f'<text x="{x_center + 8:.1f}" y="{max(y - 6, margin_top + 12):.1f}" font-size="8.5" font-weight="800" fill="{trend_color}">{escape(_fmt_axis(last_value))}</text>')

    parts.append("</svg>")
    return "".join(parts)


def _build_horizontal_bar_chart_svg(
    title: str,
    labels: Sequence[object],
    values: Sequence[object],
    *,
    color: str = "#6E1222",
    width: int = 720,
    height: int = 220,
    formatter: Callable[[object], str] = _fmt_int,
) -> str:
    clean_labels = [_fmt_text(label) for label in labels]
    clean_values = _clean_series(values)
    if not clean_labels or not clean_values or max(clean_values, default=0) <= 0:
        return _empty_svg(title, "Geen data beschikbaar.", width=width, height=height)

    chart_max = max(clean_values)
    margin_left, margin_right, margin_top, margin_bottom = 136, 24, 54, 22
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    row_height = plot_height / max(1, len(clean_labels))
    bar_height = max(12, min(22, row_height * 0.6))
    gradient_id = _chart_id(title, "leader-grad")
    panel_id = _chart_id(title, "leader-panel")
    accent_id = _chart_id(title, "leader-accent")
    glow_id = _chart_id(title, "leader-shadow")
    track_fill = "#EEF3F8"
    avg_value = sum(clean_values) / max(1, len(clean_values))
    top_value = max(clean_values, default=0.0)

    parts: list[str] = [
        f'<svg class="chart-svg" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{escape(title)}">',
        "<defs>",
        f'<linearGradient id="{panel_id}" x1="0" y1="0" x2="1" y2="1">',
        '<stop offset="0%" stop-color="#FFFFFF" />',
        '<stop offset="100%" stop-color="#F7FAFD" />',
        "</linearGradient>",
        f'<linearGradient id="{gradient_id}" x1="0" y1="0" x2="1" y2="0">',
        f'<stop offset="0%" stop-color="{_blend_hex(color, "#FFFFFF", 0.16)}" />',
        f'<stop offset="100%" stop-color="{color}" />',
        "</linearGradient>",
        f'<linearGradient id="{accent_id}" x1="0" y1="0" x2="1" y2="0">',
        '<stop offset="0%" stop-color="#EFC7D0" />',
        f'<stop offset="100%" stop-color="{color}" />',
        "</linearGradient>",
        f'<filter id="{glow_id}" x="-10%" y="-10%" width="140%" height="160%">',
        '<feDropShadow dx="0" dy="2" stdDeviation="2.2" flood-color="rgba(15,23,42,0.12)" />',
        "</filter>",
        "</defs>",
        f'<rect x="8" y="10" width="{width - 16}" height="{height - 18}" rx="22" fill="url(#{panel_id})" stroke="#DEE6EF" />',
        f'<rect x="8" y="10" width="{width - 16}" height="6" rx="22" fill="url(#{accent_id})" />',
        f'<text x="24" y="34" font-size="17" font-weight="800" fill="#0F172A">{escape(title)}</text>',
        f'<text x="24" y="50" font-size="9.2" fill="#64748B">Ranglijst in staffstijl met relatieve bijdrage per speler</text>',
        f'<rect x="{width - 208:.1f}" y="20" width="184" height="30" rx="15" fill="{_alpha_hex(color, 0.10)}" stroke="{_alpha_hex(color, 0.24)}" />',
        f'<text x="{width - 194:.1f}" y="31" font-size="7.4" font-weight="800" fill="#64748B">AVG | TOP</text>',
        f'<text x="{width - 194:.1f}" y="43" font-size="10.0" font-weight="800" fill="{_blend_hex(color, "#20324B", 0.34)}">{escape(formatter(avg_value))} | {escape(formatter(top_value))}</text>',
        f'<rect x="{margin_left:.1f}" y="{margin_top - 8:.1f}" width="{plot_width:.1f}" height="{plot_height + 12:.1f}" rx="18" fill="#FBFCFE" stroke="#E4EAF1" />',
    ]

    for index, (label, value) in enumerate(zip(clean_labels, clean_values, strict=False)):
        y = margin_top + row_height * index + (row_height - bar_height) / 2
        width_value = 0 if chart_max <= 0 else (value / chart_max) * (plot_width - 16)
        value_x = min(width - margin_right - 4, margin_left + width_value + 8)
        rank_fill = color if index < 3 else _blend_hex(color, "#FFFFFF", 0.16)
        rank_text = "#FFFFFF" if index < 3 else color
        parts.append(f'<text x="{margin_left - 14:.1f}" y="{y + bar_height * 0.72:.1f}" text-anchor="end" font-size="9" fill="#334155">{escape(label)}</text>')
        parts.append(f'<rect x="{margin_left + 8:.1f}" y="{y:.1f}" width="{plot_width - 16:.1f}" height="{bar_height:.1f}" rx="7" fill="{track_fill}" />')
        parts.append(f'<rect x="{margin_left + 8:.1f}" y="{y:.1f}" width="{width_value:.1f}" height="{bar_height:.1f}" rx="7" fill="url(#{gradient_id})" filter="url(#{glow_id})" />')
        parts.append(f'<rect x="{margin_left - 28:.1f}" y="{y - 1:.1f}" width="20" height="{bar_height + 2:.1f}" rx="6" fill="{rank_fill}" />')
        parts.append(f'<text x="{margin_left - 18:.1f}" y="{y + bar_height * 0.72:.1f}" text-anchor="middle" font-size="8.2" font-weight="800" fill="{rank_text}">{index + 1}</text>')
        parts.append(f'<text x="{value_x:.1f}" y="{y + bar_height * 0.72:.1f}" font-size="9.2" font-weight="800" fill="#0F172A">{escape(formatter(value))}</text>')

    parts.append("</svg>")
    return "".join(parts)


def _build_share_chart_svg(
    title: str,
    labels: Sequence[object],
    values: Sequence[object],
    *,
    width: int = 860,
    height: int = 180,
) -> str:
    clean_labels = [_fmt_text(label) for label in labels]
    clean_values = _clean_series(values)
    total = sum(clean_values)
    if not clean_labels or total <= 0:
        return _empty_svg(title, "Geen data beschikbaar.", width=width, height=height)

    parts: list[str] = [
        f'<svg class="chart-svg" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{escape(title)}">',
        f'<text x="8" y="18" font-size="14" font-weight="700" fill="#0B1020">{escape(title)}</text>',
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


def _describe_zone_series(labels: Sequence[object], values: Sequence[object]) -> list[tuple[str, float, str]]:
    described: list[tuple[str, float, str]] = []
    clean_labels = [_fmt_text(label) for label in labels]
    clean_values = _clean_series(values)
    for index, (label, value) in enumerate(zip(clean_labels, clean_values, strict=False)):
        if value <= 0:
            continue
        color = ZONE_COLOR_LOOKUP.get(label, SVG_COLORS[index % len(SVG_COLORS)])
        described.append((label, value, color))
    return described


def _build_pie_chart_svg(
    title: str,
    labels: Sequence[object],
    values: Sequence[object],
    *,
    width: int = 860,
    height: int = 246,
) -> str:
    described = _describe_zone_series(labels, values)
    total = sum(value for _, value, _ in described)
    if total <= 0:
        return _empty_svg(title, "Geen data beschikbaar.", width=width, height=height)

    top_label, top_value, _ = max(described, key=lambda item: item[1])
    top_share = (top_value / total) * 100 if total > 0 else 0.0
    donut_panel_x = 18
    donut_panel_y = 42
    donut_panel_w = 324
    donut_panel_h = 176
    legend_panel_x = 360
    legend_panel_y = 42
    legend_panel_w = width - legend_panel_x - 18
    legend_panel_h = 176
    pie_cx = donut_panel_x + 128
    pie_cy = donut_panel_y + 88
    radius = 68
    inner_radius = 30
    legend_top = legend_panel_y + 40
    legend_row_height = 28
    ring_shadow = _chart_id(title, "donut-shadow")
    panel_id = _chart_id(title, "donut-panel")
    accent_id = _chart_id(title, "donut-accent")

    parts: list[str] = [
        f'<svg class="chart-svg" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{escape(title)}">',
        "<defs>",
        f'<linearGradient id="{panel_id}" x1="0" y1="0" x2="1" y2="1">',
        '<stop offset="0%" stop-color="#FFFFFF" />',
        '<stop offset="100%" stop-color="#F6FAFF" />',
        "</linearGradient>",
        f'<linearGradient id="{accent_id}" x1="0" y1="0" x2="1" y2="0">',
        '<stop offset="0%" stop-color="#F7CBD4" />',
        '<stop offset="100%" stop-color="#C8102E" />',
        "</linearGradient>",
        f'<filter id="{ring_shadow}" x="-20%" y="-20%" width="160%" height="160%">',
        '<feDropShadow dx="0" dy="3" stdDeviation="3.6" flood-color="rgba(15,23,42,0.16)" />',
        "</filter>",
        "</defs>",
        f'<text x="10" y="22" font-size="15" font-weight="800" fill="#0B1020">{escape(title)}</text>',
        f'<rect x="{donut_panel_x}" y="{donut_panel_y}" width="{donut_panel_w}" height="{donut_panel_h}" rx="14" fill="url(#{panel_id})" stroke="#DCE6F2" />',
        f'<rect x="{donut_panel_x}" y="{donut_panel_y}" width="{donut_panel_w}" height="8" rx="14" fill="url(#{accent_id})" />',
        f'<rect x="{legend_panel_x}" y="{legend_panel_y}" width="{legend_panel_w}" height="{legend_panel_h}" rx="14" fill="#FFFFFF" stroke="#DCE6F2" />',
        f'<rect x="{legend_panel_x}" y="{legend_panel_y}" width="{legend_panel_w}" height="8" rx="14" fill="{_alpha_hex("#5C7697", 0.78)}" />',
        f'<line x1="{legend_panel_x - 10:.1f}" y1="{legend_panel_y + 16:.1f}" x2="{legend_panel_x - 10:.1f}" y2="{legend_panel_y + legend_panel_h - 16:.1f}" stroke="#DCE6F2" />',
        f'<circle cx="{pie_cx:.1f}" cy="{pie_cy:.1f}" r="{radius:.1f}" fill="none" stroke="#EEF3F9" stroke-width="{radius - inner_radius:.1f}" />',
    ]

    if len(described) == 1:
        _, _, color = described[0]
        parts.append(
            f'<circle cx="{pie_cx:.1f}" cy="{pie_cy:.1f}" r="{(radius + inner_radius) / 2:.1f}" fill="none" stroke="{color}" stroke-width="{radius - inner_radius:.1f}" stroke-linecap="round" filter="url(#{ring_shadow})" />'
        )
    else:
        start_angle = -math.pi / 2
        for _, value, color in described:
            sweep = (value / total) * math.tau
            end_angle = start_angle + sweep
            arc_radius = (radius + inner_radius) / 2
            start_x = pie_cx + arc_radius * math.cos(start_angle)
            start_y = pie_cy + arc_radius * math.sin(start_angle)
            end_x = pie_cx + arc_radius * math.cos(end_angle)
            end_y = pie_cy + arc_radius * math.sin(end_angle)
            large_arc = 1 if sweep > math.pi else 0
            parts.append(
                f'<path d="M {start_x:.1f} {start_y:.1f} A {arc_radius:.1f} {arc_radius:.1f} 0 {large_arc} 1 {end_x:.1f} {end_y:.1f}" fill="none" stroke="{color}" stroke-width="{radius - inner_radius:.1f}" stroke-linecap="round" filter="url(#{ring_shadow})" />'
            )
            start_angle = end_angle

    parts.append(f'<circle cx="{pie_cx:.1f}" cy="{pie_cy:.1f}" r="{inner_radius - 3:.1f}" fill="#FFFFFF" stroke="#E2E8F0" />')
    parts.append(f'<text x="{pie_cx:.1f}" y="{pie_cy - 4:.1f}" text-anchor="middle" font-size="24" font-weight="800" fill="#0F172A">{escape(_fmt_dec(top_share, 0))}%</text>')
    parts.append(f'<text x="{pie_cx:.1f}" y="{pie_cy + 14:.1f}" text-anchor="middle" font-size="9.5" font-weight="700" fill="#64748B">{escape(top_label)}</text>')
    parts.append(f'<rect x="{donut_panel_x + 42:.1f}" y="{donut_panel_y + donut_panel_h - 38:.1f}" width="{donut_panel_w - 84:.1f}" height="24" rx="12" fill="{_alpha_hex("#C8102E", 0.08)}" stroke="{_alpha_hex("#C8102E", 0.18)}" />')
    parts.append(f'<text x="{pie_cx:.1f}" y="{donut_panel_y + donut_panel_h - 22:.1f}" text-anchor="middle" font-size="9.2" font-weight="800" fill="#334155">Total {escape(_fmt_distance_km(total))}</text>')
    parts.append(f'<text x="{legend_panel_x + 18:.1f}" y="{legend_panel_y + 24:.1f}" font-size="9.5" font-weight="800" fill="#64748B">ZONE MIX</text>')
    parts.append(f'<rect x="{legend_panel_x + legend_panel_w - 124:.1f}" y="{legend_panel_y + 14:.1f}" width="106" height="22" rx="11" fill="{_alpha_hex("#0F766E", 0.08)}" stroke="{_alpha_hex("#0F766E", 0.18)}" />')
    parts.append(f'<text x="{legend_panel_x + legend_panel_w - 114:.1f}" y="{legend_panel_y + 28:.1f}" font-size="8.6" font-weight="800" fill="#0F766E">Top zone {escape(_fmt_dec(top_share, 1))}% </text>')

    for index, (label, value, color) in enumerate(described):
        percentage = (value / total) * 100 if total > 0 else 0.0
        row_y = legend_top + index * legend_row_height
        row_box_y = row_y - 15
        progress_x = legend_panel_x + 42
        progress_w = 120
        parts.append(
            f'<rect x="{legend_panel_x + 14:.1f}" y="{row_box_y:.1f}" width="{legend_panel_w - 28:.1f}" height="22" rx="11" fill="{_alpha_hex(color, 0.06)}" />'
        )
        parts.append(f'<circle cx="{legend_panel_x + 28:.1f}" cy="{row_y - 4:.1f}" r="5.5" fill="{color}" />')
        parts.append(f'<text x="{legend_panel_x + 42:.1f}" y="{row_y - 1:.1f}" font-size="10.4" font-weight="700" fill="#334155">{escape(label)}</text>')
        parts.append(f'<rect x="{progress_x:.1f}" y="{row_y + 4.5:.1f}" width="{progress_w:.1f}" height="4.2" rx="2.1" fill="#E5EDF7" />')
        parts.append(f'<rect x="{progress_x:.1f}" y="{row_y + 4.5:.1f}" width="{progress_w * (percentage / 100):.1f}" height="4.2" rx="2.1" fill="{color}" />')
        parts.append(
            f'<text x="{legend_panel_x + legend_panel_w - 18:.1f}" y="{row_y - 1:.1f}" text-anchor="end" font-size="11.2" font-weight="800" fill="#0F172A">{escape(_fmt_dec(percentage, 1))}%</text>'
        )
        parts.append(f'<text x="{legend_panel_x + legend_panel_w - 18:.1f}" y="{row_y + 10:.1f}" text-anchor="end" font-size="8.8" fill="#64748B">{escape(_fmt_distance_km(value))}</text>')

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
    _ = monitoring_summary
    _day_table = day_table
    td_change = summary.get("td_vs_prev")
    hsr_change = summary.get("hsr_vs_prev")
    load_title_parts: list[str] = []
    valid_changes = [value for value in [td_change, hsr_change] if pd.notna(value)]
    if pd.notna(td_change):
        load_title_parts.append(f"TD {_fmt_percent(td_change, signed=True)}")
    if pd.notna(hsr_change):
        load_title_parts.append(f"HSR {_fmt_percent(hsr_change, signed=True)}")
    if not load_title_parts:
        return []
    return [
        {
            "eyebrow": "Load status",
            "title": " | ".join(load_title_parts),
            "body": "Vergeleken met de rolling 4-week referentie voor externe load.",
            "tone": _tone_from_change(max(valid_changes, key=lambda item: abs(float(item)))) if valid_changes else "neutral",
        }
    ]


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


def _build_zone_share_panels(zone_df: pd.DataFrame | None, zone_day_table: pd.DataFrame | None) -> list[dict[str, str]]:
    panels: list[dict[str, str]] = []
    zones = zone_df.copy() if isinstance(zone_df, pd.DataFrame) else pd.DataFrame()
    zone_days = zone_day_table.copy() if isinstance(zone_day_table, pd.DataFrame) else pd.DataFrame()

    if not zones.empty:
        panels.append(
            {
                "svg": _build_pie_chart_svg(
                    "Hele week",
                    zones.get("zone", pd.Series(dtype=str)).tolist(),
                    zones.get("value", pd.Series(dtype=float)).tolist(),
                )
            }
        )

    if not zone_days.empty:
        for _, row in zone_days.sort_values("datum").iterrows():
            labels: list[str] = []
            values: list[float] = []
            for label, column, _ in ZONE_SPECS:
                numeric = pd.to_numeric(row.get(column), errors="coerce")
                if pd.notna(numeric) and float(numeric) > 0:
                    labels.append(label)
                    values.append(float(numeric))
            if not values:
                continue
            panel_title = _fmt_text(row.get("label"))
            if panel_title == "--":
                panel_title = _weekday_label(row.get("datum"))
            panels.append({"svg": _build_pie_chart_svg(panel_title, labels, values)})

    if panels:
        return panels

    return [{"svg": _empty_svg("Distance Zone Share", "Geen data beschikbaar.", height=340)}]


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
    zone_day_table: pd.DataFrame | None = None,
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
    zone_day_distribution = zone_day_table.copy() if isinstance(zone_day_table, pd.DataFrame) else pd.DataFrame()
    monitoring_watchlist = monitoring_player_table.copy() if isinstance(monitoring_player_table, pd.DataFrame) else pd.DataFrame()
    if not monitoring_watchlist.empty:
        monitoring_watchlist = monitoring_watchlist.sort_values(
            ["readiness_score", "avg_rpe", "player_name"],
            ascending=[True, False, True],
            na_position="last",
        ).head(12)

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
        "report_description": "",
        "logo_src": LOGO_SRC,
        "report_header_meta": [],
        "badges": [],
        "cards": [
            {"label": "Total Distance", "value": _fmt_distance_km(summary.get("total_distance")), "foot": "Opgetelde teamload in de week"},
            {"label": "HSR / HSD", "value": _fmt_distance_km(summary.get("hsr_hsd")), "foot": "Sprint plus high sprint distance"},
            {"label": "Dist / Player", "value": _fmt_distance_km(summary.get("dist_per_player")), "foot": "Teamload gedeeld door actieve spelers"},
            {"label": "Sprints", "value": _fmt_int(summary.get("sprints")), "foot": "Totale sprintacties in deze week"},
            {"label": "Top Speed", "value": _fmt_speed(summary.get("top_speed")), "foot": "Hoogste gemeten snelheid"},
            {"label": "Speed Exposures", "value": _fmt_int(summary.get("speed_exposures")), "foot": "Sessies op 90% van seizoenstop"},
        ],
        "focus_cards": [],
        "day_cards": _build_week_day_cards(day_table),
        "leader_cards": [],
        "monitoring_cards": [],
        "chart_sections": [
            {
                "eyebrow": "Load profile",
                "title": "Weekly load rhythm",
                "subtitle": "Dagelijkse teambelasting en high-speed output binnen de geselecteerde microcycle.",
                "columns": 2,
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
                "columns": 2,
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
                "columns": 2,
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
                "title": "Daily speed exposures",
                "subtitle": "Sessies per dag op of boven 90% van de individuele seizoensmax.",
                "columns": 1,
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
                ],
            },
            {
                "eyebrow": "Locomotion zones",
                "title": "Distance zone share",
                "subtitle": "Verdeling van walking, jogging, running, sprint en high sprint voor de hele week en per actieve dag.",
                "columns": 2,
                "panels": _build_zone_share_panels(zone_df, zone_day_distribution),
            },
            {
                "eyebrow": "Monitoring",
                "title": "Daily wellness profile +/- SD",
                "subtitle": "Fysieke en mentale welzijnsindicatoren per dag met standaarddeviatie.",
                "columns": 2,
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
                "subtitle": "Snel overzicht van volume-, high-speed- en sprintleiders voor de weekevaluatie.",
                "columns": 3,
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
                    {
                        "svg": _build_horizontal_bar_chart_svg(
                            "Top Players by Sprints",
                            sprint_leaders.get("player_name", pd.Series(dtype=str)).tolist(),
                            sprint_leaders.get("sprints", pd.Series(dtype=float)).tolist(),
                            color="#EA3351",
                            formatter=_fmt_int,
                        )
                    },
                ],
            },
            {
                "eyebrow": "Leaders",
                "title": "Session RPE overview",
                "subtitle": "Sessie-RPE overzicht voor dagen met enkele of dubbele sessies.",
                "columns": 1,
                "panels": [
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
