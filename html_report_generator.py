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


def _chart_badge(title: str) -> str:
    normalized = title.lower()
    if "zone" in normalized:
        return "ZONE"
    if "hsr" in normalized:
        return "HSR"
    if "distance" in normalized:
        return "TD"
    if "speed exposure" in normalized:
        return "EXP"
    if "sprint" in normalized:
        return "SPR"
    if "accel" in normalized or "decel" in normalized:
        return "ACC"
    if "rpe" in normalized:
        return "RPE"
    if "wellness" in normalized:
        return "WEL"
    return "MVV"


def _append_stat_chip(
    parts: list[str],
    *,
    x: float,
    y: float,
    label: str,
    value: str,
    fill: str,
    stroke: str,
    value_color: str = "#0F172A",
) -> None:
    parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="88" height="28" rx="8" fill="{fill}" stroke="{stroke}" />')
    parts.append(f'<text x="{x + 10:.1f}" y="{y + 11:.1f}" font-size="6.8" font-weight="800" fill="#64748B">{escape(label)}</text>')
    parts.append(f'<text x="{x + 10:.1f}" y="{y + 22:.1f}" font-size="10.2" font-weight="800" fill="{value_color}">{escape(value)}</text>')


def _append_chart_footer(parts: list[str], *, width: int, height: int, accent: str) -> None:
    footer_y = height - 24
    parts.append(f'<line x1="12" y1="{footer_y:.1f}" x2="{width - 12}" y2="{footer_y:.1f}" stroke="#E4EAF1" />')
    parts.append(f'<text x="24" y="{height - 8:.1f}" font-size="8.4" font-weight="800" fill="{accent}" letter-spacing="0.08em">MVV MAASTRICHT</text>')
    stripe_x = width - 64
    parts.append(f'<polygon points="{stripe_x}, {height - 18} {stripe_x + 14}, {height - 18} {stripe_x + 28}, {height - 6} {stripe_x + 14}, {height - 6}" fill="{accent}" />')
    parts.append(f'<polygon points="{stripe_x + 18}, {height - 18} {stripe_x + 32}, {height - 18} {stripe_x + 46}, {height - 6} {stripe_x + 32}, {height - 6}" fill="{_blend_hex(accent, "#FFFFFF", 0.24)}" />')


def _append_brand_tile(parts: list[str], *, width: int, y: float = 18, accent: str = "#C8102E") -> None:
    tile_x = width - 58
    parts.append(f'<rect x="{tile_x:.1f}" y="{y:.1f}" width="36" height="44" rx="7" fill="{accent}" />')
    parts.append(f'<rect x="{tile_x + 5:.1f}" y="{y + 5:.1f}" width="26" height="16" rx="4" fill="{_alpha_hex("#FFFFFF", 0.18)}" stroke="{_alpha_hex("#FFFFFF", 0.38)}" />')
    parts.append(f'<text x="{tile_x + 18:.1f}" y="{y + 31:.1f}" text-anchor="middle" font-size="10" font-weight="800" fill="#FFFFFF">MVV</text>')
    parts.append(f'<text x="{tile_x + 18:.1f}" y="{y + 39:.1f}" text-anchor="middle" font-size="4.6" font-weight="700" fill="#FFFFFF" letter-spacing="0.08em">REPORT</text>')


def _build_vertical_bar_chart_svg(
    title: str,
    labels: Sequence[object],
    values: Sequence[object],
    *,
    color: str = "#C8102E",
    width: int = 860,
    height: int = 248,
    y_max: float | None = None,
    formatter: Callable[[object], str] = _fmt_int,
) -> str:
    clean_labels = [_fmt_text(label) for label in labels]
    clean_values = _clean_series(values)
    if not clean_labels or not clean_values or max(clean_values, default=0) <= 0:
        return _empty_svg(title, "Geen data beschikbaar.", width=width, height=height)

    chart_max = y_max if y_max is not None else _nice_max(max(clean_values))
    margin_left, margin_right, margin_top, margin_bottom = 56, 24, 94, 46
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    slot_width = plot_width / max(1, len(clean_values))
    label_font = 9 if len(clean_labels) > 8 else 10
    grid_lines = 4
    panel_id = _chart_id(title, "trend-panel")
    plot_id = _chart_id(title, "trend-plot")
    area_id = _chart_id(title, "trend-area")
    line_shadow_id = _chart_id(title, "trend-line-shadow")
    point_shadow_id = _chart_id(title, "trend-point-shadow")
    avg_value = sum(clean_values) / max(1, len(clean_values))
    peak_value = max(clean_values, default=0.0)
    peak_indices = {idx for idx, value in enumerate(clean_values) if value == peak_value}
    last_value = clean_values[-1] if clean_values else 0.0
    line_color = _blend_hex(color, "#102033", 0.18)
    badge = _chart_badge(title)
    line_points = [
        (
            margin_left + slot_width * index + slot_width / 2,
            margin_top + plot_height - (0 if chart_max <= 0 else (value / chart_max) * plot_height),
        )
        for index, value in enumerate(clean_values)
    ]
    subtitle = "Area and line view met piek, gemiddelde en laatste meetpunt"

    parts: list[str] = [
        f'<svg class="chart-svg" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{escape(title)}">',
        "<defs>",
        f'<linearGradient id="{panel_id}" x1="0" y1="0" x2="1" y2="1">',
        '<stop offset="0%" stop-color="#FFFFFF" />',
        '<stop offset="100%" stop-color="#F7FAFD" />',
        "</linearGradient>",
        f'<linearGradient id="{plot_id}" x1="0" y1="0" x2="0" y2="1">',
        '<stop offset="0%" stop-color="#FFFFFF" />',
        '<stop offset="100%" stop-color="#F6F8FC" />',
        "</linearGradient>",
        f'<linearGradient id="{area_id}" x1="0" y1="0" x2="0" y2="1">',
        f'<stop offset="0%" stop-color="{_alpha_hex(color, 0.42)}" />',
        f'<stop offset="54%" stop-color="{_alpha_hex(color, 0.16)}" />',
        f'<stop offset="100%" stop-color="{_alpha_hex(color, 0.02)}" />',
        "</linearGradient>",
        f'<filter id="{line_shadow_id}" x="-10%" y="-10%" width="140%" height="150%">',
        f'<feDropShadow dx="0" dy="4" stdDeviation="4" flood-color="{_alpha_hex(color, 0.20)}" />',
        "</filter>",
        f'<filter id="{point_shadow_id}" x="-20%" y="-20%" width="160%" height="160%">',
        '<feDropShadow dx="0" dy="2" stdDeviation="2.2" flood-color="rgba(15,23,42,0.16)" />',
        "</filter>",
        "</defs>",
        f'<rect x="8" y="10" width="{width - 16}" height="{height - 18}" rx="18" fill="url(#{panel_id})" stroke="#D8E1EC" />',
        f'<rect x="16" y="18" width="40" height="20" rx="3" fill="{color}" />',
        f'<text x="36" y="32" text-anchor="middle" font-size="8.8" font-weight="800" fill="#FFFFFF">{escape(badge)}</text>',
        f'<text x="24" y="58" font-size="21" font-weight="800" fill="#0F172A">{escape(title)}</text>',
        f'<text x="24" y="74" font-size="8.5" font-weight="800" fill="{color}" letter-spacing="0.06em">{escape(subtitle.upper())}</text>',
        f'<line x1="24" y1="82" x2="{width - 24}" y2="82" stroke="#E7EDF4" />',
    ]

    chip_x = width - 340
    _append_stat_chip(parts, x=chip_x, y=18, label="AVG", value=formatter(avg_value), fill=_alpha_hex(color, 0.08), stroke=_alpha_hex(color, 0.20))
    _append_stat_chip(parts, x=chip_x + 94, y=18, label="PEAK", value=formatter(peak_value), fill=_alpha_hex("#F59E0B", 0.10), stroke=_alpha_hex("#F59E0B", 0.24))
    _append_stat_chip(parts, x=chip_x + 188, y=18, label="LAST", value=formatter(last_value), fill=_alpha_hex("#0F766E", 0.08), stroke=_alpha_hex("#0F766E", 0.20))
    _append_brand_tile(parts, width=width, y=18, accent=color)

    parts.append(
        f'<rect x="{margin_left - 8:.1f}" y="{margin_top - 8:.1f}" width="{plot_width + 16:.1f}" height="{plot_height + 16:.1f}" rx="16" fill="url(#{plot_id})" stroke="#DFE7F0" />'
    )

    for step in range(grid_lines + 1):
        ratio = step / grid_lines
        y = margin_top + plot_height - ratio * plot_height
        axis_value = chart_max * ratio
        parts.append(f'<line x1="{margin_left:.1f}" y1="{y:.1f}" x2="{width - margin_right:.1f}" y2="{y:.1f}" stroke="#E7EDF4" stroke-dasharray="3 6" />')
        parts.append(f'<text x="{margin_left - 10:.1f}" y="{y + 3:.1f}" text-anchor="end" font-size="8.8" fill="#708199">{escape(_fmt_axis(axis_value))}</text>')

    if len(line_points) >= 2:
        peak_index = max(range(len(clean_values)), key=lambda idx: clean_values[idx])
        peak_center = line_points[peak_index][0]
        parts.append(
            f'<rect x="{peak_center - slot_width * 0.42:.1f}" y="{margin_top + 6:.1f}" width="{slot_width * 0.84:.1f}" height="{plot_height - 6:.1f}" rx="14" fill="{_alpha_hex(color, 0.05)}" />'
        )
        area_points = [(line_points[0][0], margin_top + plot_height)] + line_points + [(line_points[-1][0], margin_top + plot_height)]
        parts.append(f'<path d="{_series_path(area_points)} Z" fill="url(#{area_id})" />')
        parts.append(f'<path d="{_series_path(line_points)}" fill="none" stroke="#FFFFFF" stroke-width="7.8" stroke-linecap="round" stroke-linejoin="round" opacity="0.96" />')
        parts.append(
            f'<path d="{_series_path(line_points)}" fill="none" stroke="{line_color}" stroke-width="4.0" stroke-linecap="round" stroke-linejoin="round" filter="url(#{line_shadow_id})" />'
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
        point_radius = 7.0 if index in peak_indices else 5.8
        point_fill = "#FFF7ED" if index in peak_indices else "#FFFFFF"
        point_stroke = "#D97706" if index in peak_indices else line_color
        parts.append(
            f'<circle cx="{x_center:.1f}" cy="{y:.1f}" r="{point_radius:.1f}" fill="{point_fill}" stroke="{point_stroke}" stroke-width="2.1" filter="url(#{point_shadow_id})" />'
        )
        parts.append(f'<circle cx="{x_center:.1f}" cy="{y:.1f}" r="2.8" fill="{line_color}" />')
        if value > 0:
            parts.append(
                f'<text x="{x_center:.1f}" y="{max(y - 12, margin_top + 10):.1f}" text-anchor="middle" font-size="8.8" font-weight="800" fill="#0F172A">{escape(formatter(value))}</text>'
            )
        label_y = height - 34
        parts.append(
            f'<text x="{x_center:.1f}" y="{label_y}" font-size="{label_font}" fill="#55657E" text-anchor="end" transform="rotate(-28 {x_center:.1f} {label_y})">{escape(label)}</text>'
        )

    _append_chart_footer(parts, width=width, height=height, accent=color)
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
    height: int = 248,
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
    margin_left, margin_right, margin_top, margin_bottom = 56, 24, 94, 46
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    slot_width = plot_width / max(1, len(clean_means))
    label_font = 9 if len(clean_labels) > 8 else 10
    grid_lines = 4
    panel_id = _chart_id(title, "band-panel")
    plot_id = _chart_id(title, "band-plot")
    band_id = _chart_id(title, "band-fill")
    line_shadow_id = _chart_id(title, "band-shadow")
    point_shadow_id = _chart_id(title, "band-point-shadow")
    line_color = _blend_hex(color, "#102033", 0.18)
    avg_value = sum(clean_means) / max(1, len(clean_means))
    avg_error = sum(clean_errors) / max(1, len(clean_errors))
    peak_value = max(clean_means, default=0.0)
    peak_indices = {idx for idx, value in enumerate(clean_means) if value == peak_value}
    badge = _chart_badge(title)
    subtitle = "Mean trend met standaarddeviatie als band en executive staff-kpi's"
    mean_points: list[tuple[float, float]] = []
    upper_points: list[tuple[float, float]] = []
    lower_points: list[tuple[float, float]] = []

    parts: list[str] = [
        f'<svg class="chart-svg" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{escape(title)}">',
        "<defs>",
        f'<linearGradient id="{panel_id}" x1="0" y1="0" x2="1" y2="1">',
        '<stop offset="0%" stop-color="#FFFFFF" />',
        '<stop offset="100%" stop-color="#F7FAFD" />',
        "</linearGradient>",
        f'<linearGradient id="{plot_id}" x1="0" y1="0" x2="0" y2="1">',
        '<stop offset="0%" stop-color="#FFFFFF" />',
        '<stop offset="100%" stop-color="#F6F8FC" />',
        "</linearGradient>",
        f'<linearGradient id="{band_id}" x1="0" y1="0" x2="0" y2="1">',
        f'<stop offset="0%" stop-color="{_alpha_hex(color, 0.28)}" />',
        f'<stop offset="100%" stop-color="{_alpha_hex(color, 0.05)}" />',
        "</linearGradient>",
        f'<filter id="{line_shadow_id}" x="-10%" y="-10%" width="140%" height="150%">',
        f'<feDropShadow dx="0" dy="4" stdDeviation="4" flood-color="{_alpha_hex(color, 0.16)}" />',
        "</filter>",
        f'<filter id="{point_shadow_id}" x="-20%" y="-20%" width="160%" height="160%">',
        '<feDropShadow dx="0" dy="2" stdDeviation="2.1" flood-color="rgba(15,23,42,0.14)" />',
        "</filter>",
        "</defs>",
        f'<rect x="8" y="10" width="{width - 16}" height="{height - 18}" rx="18" fill="url(#{panel_id})" stroke="#D8E1EC" />',
        f'<rect x="16" y="18" width="40" height="20" rx="3" fill="{color}" />',
        f'<text x="36" y="32" text-anchor="middle" font-size="8.8" font-weight="800" fill="#FFFFFF">{escape(badge)}</text>',
        f'<text x="24" y="58" font-size="21" font-weight="800" fill="#0F172A">{escape(title)}</text>',
        f'<text x="24" y="74" font-size="8.5" font-weight="800" fill="{color}" letter-spacing="0.06em">{escape(subtitle.upper())}</text>',
        f'<line x1="24" y1="82" x2="{width - 24}" y2="82" stroke="#E7EDF4" />',
    ]

    chip_x = width - 340
    _append_stat_chip(parts, x=chip_x, y=18, label="AVG", value=formatter(avg_value), fill=_alpha_hex(color, 0.08), stroke=_alpha_hex(color, 0.20))
    _append_stat_chip(parts, x=chip_x + 94, y=18, label="PEAK", value=formatter(peak_value), fill=_alpha_hex("#F59E0B", 0.10), stroke=_alpha_hex("#F59E0B", 0.24))
    _append_stat_chip(parts, x=chip_x + 188, y=18, label="AVG SD", value=formatter(avg_error), fill=_alpha_hex("#5C7697", 0.08), stroke=_alpha_hex("#5C7697", 0.20))
    _append_brand_tile(parts, width=width, y=18, accent=color)
    parts.append(
        f'<rect x="{margin_left - 8:.1f}" y="{margin_top - 8:.1f}" width="{plot_width + 16:.1f}" height="{plot_height + 16:.1f}" rx="16" fill="url(#{plot_id})" stroke="#DFE7F0" />'
    )

    for step in range(grid_lines + 1):
        ratio = step / grid_lines
        y = margin_top + plot_height - ratio * plot_height
        axis_value = chart_max * ratio
        parts.append(f'<line x1="{margin_left:.1f}" y1="{y:.1f}" x2="{width - margin_right:.1f}" y2="{y:.1f}" stroke="#E7EDF4" stroke-dasharray="3 6" />')
        parts.append(f'<text x="{margin_left - 10:.1f}" y="{y + 3:.1f}" text-anchor="end" font-size="8.8" fill="#708199">{escape(_fmt_axis(axis_value))}</text>')

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
        label_y = height - 34
        parts.append(
            f'<text x="{x_center:.1f}" y="{label_y}" font-size="{label_font}" fill="#55657E" text-anchor="end" transform="rotate(-28 {x_center:.1f} {label_y})">{escape(label)}</text>'
        )

    if len(mean_points) >= 2:
        peak_index = max(range(len(clean_means)), key=lambda idx: clean_means[idx])
        peak_center = mean_points[peak_index][0]
        parts.append(
            f'<rect x="{peak_center - slot_width * 0.42:.1f}" y="{margin_top + 6:.1f}" width="{slot_width * 0.84:.1f}" height="{plot_height - 6:.1f}" rx="14" fill="{_alpha_hex(color, 0.05)}" />'
        )
        band_points = upper_points + list(reversed(lower_points))
        parts.append(f'<path d="{_series_path(band_points)} Z" fill="url(#{band_id})" />')
        parts.append(f'<path d="{_series_path(mean_points)}" fill="none" stroke="#FFFFFF" stroke-width="7.4" stroke-linecap="round" stroke-linejoin="round" opacity="0.94" />')
        parts.append(
            f'<path d="{_series_path(mean_points)}" fill="none" stroke="{line_color}" stroke-width="3.8" stroke-linecap="round" stroke-linejoin="round" filter="url(#{line_shadow_id})" />'
        )

    if avg_value > 0 and chart_max > 0:
        avg_y = margin_top + plot_height - (avg_value / chart_max) * plot_height
        parts.append(
            f'<line x1="{margin_left:.1f}" y1="{avg_y:.1f}" x2="{width - margin_right:.1f}" y2="{avg_y:.1f}" stroke="{_alpha_hex(line_color, 0.72)}" stroke-width="1.6" stroke-dasharray="8 6" />'
        )
        parts.append(
            f'<text x="{width - margin_right - 4:.1f}" y="{avg_y - 6:.1f}" text-anchor="end" font-size="8.7" font-weight="800" fill="{line_color}">Avg {escape(formatter(avg_value))}</text>'
        )

    for index, _ in enumerate(clean_errors):
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
            f'<circle cx="{x_center:.1f}" cy="{mean_y:.1f}" r="{6.8 if index in peak_indices else 5.6:.1f}" fill="{point_fill}" stroke="{point_stroke}" stroke-width="2.0" filter="url(#{point_shadow_id})" />'
        )
        parts.append(f'<circle cx="{x_center:.1f}" cy="{mean_y:.1f}" r="2.6" fill="{line_color}" />')
        if clean_means[index] > 0:
            parts.append(
                f'<text x="{x_center:.1f}" y="{max(mean_y - 12, margin_top + 10):.1f}" text-anchor="middle" font-size="8.9" font-weight="800" fill="#0F172A">{escape(formatter(clean_means[index]))}</text>'
            )

    _append_chart_footer(parts, width=width, height=height, accent=color)
    parts.append("</svg>")
    return "".join(parts)


def _build_grouped_error_bar_chart_svg(
    title: str,
    labels: Sequence[object],
    series: Sequence[dict[str, Any]],
    *,
    width: int = 860,
    height: int = 248,
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
    margin_left, margin_right, margin_top, margin_bottom = 56, 24, 94, 46
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    slot_width = plot_width / max(1, len(clean_labels))
    label_font = 9 if len(clean_labels) > 8 else 10
    grid_lines = 4
    panel_id = _chart_id(title, "multi-panel")
    plot_id = _chart_id(title, "multi-plot")
    line_shadow_id = _chart_id(title, "multi-shadow")
    point_shadow_id = _chart_id(title, "multi-point-shadow")
    series_avg = [sum(item["values"]) / len(item["values"]) if item["values"] else 0.0 for item in clean_series]
    series_error_avg = [sum(item["errors"]) / len(item["errors"]) if item["errors"] else 0.0 for item in clean_series]
    badge = _chart_badge(title)
    subtitle = "Multi-series profile met overlap, spreiding en staff-kpi's"
    series_points: list[list[tuple[float, float]]] = [[] for _ in clean_series]
    series_upper_points: list[list[tuple[float, float]]] = [[] for _ in clean_series]
    series_lower_points: list[list[tuple[float, float]]] = [[] for _ in clean_series]

    parts: list[str] = [
        f'<svg class="chart-svg" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{escape(title)}">',
        "<defs>",
        f'<linearGradient id="{panel_id}" x1="0" y1="0" x2="1" y2="1">',
        '<stop offset="0%" stop-color="#FFFFFF" />',
        '<stop offset="100%" stop-color="#F7FAFD" />',
        "</linearGradient>",
        f'<linearGradient id="{plot_id}" x1="0" y1="0" x2="0" y2="1">',
        '<stop offset="0%" stop-color="#FFFFFF" />',
        '<stop offset="100%" stop-color="#F6F8FC" />',
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
            f'<rect x="8" y="10" width="{width - 16}" height="{height - 18}" rx="18" fill="url(#{panel_id})" stroke="#D8E1EC" />',
            f'<rect x="16" y="18" width="40" height="20" rx="3" fill="#C8102E" />',
            f'<text x="36" y="32" text-anchor="middle" font-size="8.8" font-weight="800" fill="#FFFFFF">{escape(badge)}</text>',
            f'<text x="24" y="58" font-size="21" font-weight="800" fill="#0F172A">{escape(title)}</text>',
            f'<text x="24" y="74" font-size="8.5" font-weight="800" fill="#C8102E" letter-spacing="0.06em">{escape(subtitle.upper())}</text>',
            f'<line x1="24" y1="82" x2="{width - 24}" y2="82" stroke="#E7EDF4" />',
            f'<rect x="{margin_left - 8:.1f}" y="{margin_top - 8:.1f}" width="{plot_width + 16:.1f}" height="{plot_height + 16:.1f}" rx="16" fill="url(#{plot_id})" stroke="#DFE7F0" />',
        ]
    )
    _append_brand_tile(parts, width=width, y=18, accent="#C8102E")

    legend_x = 24
    for item in clean_series:
        pill_width = max(86, len(item["label"]) * 7 + 30)
        parts.append(f'<rect x="{legend_x:.1f}" y="84" width="{pill_width:.1f}" height="20" rx="10" fill="{_alpha_hex(item["color"], 0.11)}" stroke="{_alpha_hex(item["color"], 0.24)}" />')
        parts.append(f'<circle cx="{legend_x + 10:.1f}" cy="94" r="4.4" fill="{item["color"]}" />')
        parts.append(f'<text x="{legend_x + 20:.1f}" y="97" font-size="8.8" font-weight="700" fill="#475569">{escape(item["label"])}</text>')
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
        parts.append(f'<line x1="{margin_left:.1f}" y1="{y:.1f}" x2="{width - margin_right:.1f}" y2="{y:.1f}" stroke="#E7EDF4" stroke-dasharray="3 6" />')
        parts.append(f'<text x="{margin_left - 10:.1f}" y="{y + 3:.1f}" text-anchor="end" font-size="8.8" fill="#708199">{escape(_fmt_axis(axis_value))}</text>')

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
        label_y = height - 34
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
            f'<path d="{_series_path(series_points[series_index])}" fill="none" stroke="#FFFFFF" stroke-width="6.6" stroke-linecap="round" stroke-linejoin="round" opacity="0.90" />'
        )
        parts.append(
            f'<path d="{_series_path(series_points[series_index])}" fill="none" stroke="{trend_color}" stroke-width="3.3" stroke-linecap="round" stroke-linejoin="round" filter="url(#{line_shadow_id})" />'
        )
        for point_index, (x_center, y) in enumerate(series_points[series_index]):
            parts.append(f'<circle cx="{x_center:.1f}" cy="{y:.1f}" r="4.6" fill="#FFFFFF" stroke="{trend_color}" stroke-width="1.7" filter="url(#{point_shadow_id})" />')
            if point_index == len(series_points[series_index]) - 1:
                last_value = item["values"][-1] if item["values"] else 0.0
                parts.append(f'<text x="{x_center + 8:.1f}" y="{max(y - 6, margin_top + 12):.1f}" font-size="8.5" font-weight="800" fill="{trend_color}">{escape(_fmt_axis(last_value))}</text>')

    _append_chart_footer(parts, width=width, height=height, accent="#C8102E")
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
    margin_left, margin_right, margin_top, margin_bottom = 138, 24, 72, 34
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    row_height = plot_height / max(1, len(clean_labels))
    bar_height = max(12, min(22, row_height * 0.6))
    gradient_id = _chart_id(title, "leader-grad")
    panel_id = _chart_id(title, "leader-panel")
    glow_id = _chart_id(title, "leader-shadow")
    track_fill = "#EEF3F8"
    avg_value = sum(clean_values) / max(1, len(clean_values))
    top_value = max(clean_values, default=0.0)
    badge = _chart_badge(title)
    subtitle = "Compacte MVV-ranglijst met volumeleiders en directe staff-readout"

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
        f'<filter id="{glow_id}" x="-10%" y="-10%" width="140%" height="160%">',
        '<feDropShadow dx="0" dy="2" stdDeviation="2.2" flood-color="rgba(15,23,42,0.12)" />',
        "</filter>",
        "</defs>",
        f'<rect x="8" y="10" width="{width - 16}" height="{height - 18}" rx="18" fill="url(#{panel_id})" stroke="#D8E1EC" />',
        f'<rect x="16" y="18" width="40" height="20" rx="3" fill="{color}" />',
        f'<text x="36" y="32" text-anchor="middle" font-size="8.8" font-weight="800" fill="#FFFFFF">{escape(badge)}</text>',
        f'<text x="24" y="58" font-size="20" font-weight="800" fill="#0F172A">{escape(title)}</text>',
        f'<text x="24" y="74" font-size="8.4" font-weight="800" fill="{color}" letter-spacing="0.06em">{escape(subtitle.upper())}</text>',
        f'<rect x="{width - 208:.1f}" y="18" width="184" height="30" rx="8" fill="{_alpha_hex(color, 0.10)}" stroke="{_alpha_hex(color, 0.24)}" />',
        f'<text x="{width - 194:.1f}" y="29" font-size="7.0" font-weight="800" fill="#64748B">AVG | TOP</text>',
        f'<text x="{width - 194:.1f}" y="41" font-size="10.0" font-weight="800" fill="{_blend_hex(color, "#20324B", 0.34)}">{escape(formatter(avg_value))} | {escape(formatter(top_value))}</text>',
        f'<line x1="24" y1="82" x2="{width - 24}" y2="82" stroke="#E7EDF4" />',
        f'<rect x="{margin_left:.1f}" y="{margin_top - 8:.1f}" width="{plot_width:.1f}" height="{plot_height + 12:.1f}" rx="16" fill="#FBFCFE" stroke="#E4EAF1" />',
    ]
    _append_brand_tile(parts, width=width, y=18, accent=color)

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

    _append_chart_footer(parts, width=width, height=height, accent=color)
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
    width: int = 560,
    height: int = 146,
) -> str:
    described = _describe_zone_series(labels, values)
    total = sum(value for _, value, _ in described)
    if total <= 0:
        return _empty_svg(title, "Geen data beschikbaar.", width=width, height=height)

    panel_id = _chart_id(title, "pie-panel")
    pie_shadow = _chart_id(title, "pie-shadow")
    badge = _chart_badge(title)
    pie_cx = 108
    pie_cy = 96
    radius = 43
    legend_x = 202
    legend_top = 74
    legend_row_height = 13.4

    parts: list[str] = [
        f'<svg class="chart-svg" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{escape(title)}">',
        "<defs>",
        f'<linearGradient id="{panel_id}" x1="0" y1="0" x2="1" y2="1">',
        '<stop offset="0%" stop-color="#FFFFFF" />',
        '<stop offset="100%" stop-color="#F6FAFF" />',
        "</linearGradient>",
        f'<filter id="{pie_shadow}" x="-20%" y="-20%" width="160%" height="160%">',
        '<feDropShadow dx="0" dy="3" stdDeviation="3.2" flood-color="rgba(15,23,42,0.14)" />',
        "</filter>",
        "</defs>",
        f'<rect x="8" y="10" width="{width - 16}" height="{height - 18}" rx="18" fill="#FFFFFF" stroke="#D8E1EC" />',
        f'<rect x="16" y="18" width="40" height="18" rx="3" fill="#C8102E" />',
        f'<text x="36" y="31" text-anchor="middle" font-size="8.2" font-weight="800" fill="#FFFFFF">{escape(badge)}</text>',
        f'<text x="24" y="51" font-size="16.4" font-weight="800" fill="#0F172A">{escape(title)}</text>',
        f'<text x="{width - 26}" y="31" text-anchor="end" font-size="7.6" font-weight="800" fill="#64748B" letter-spacing="0.08em">TOTAAL {escape(_fmt_distance_km(total))}</text>',
        f'<line x1="24" y1="58" x2="{width - 24}" y2="58" stroke="#E7EDF4" />',
        f'<rect x="22" y="66" width="170" height="68" rx="15" fill="url(#{panel_id})" stroke="#DFE7F0" />',
        f'<rect x="198" y="66" width="{width - 220}" height="68" rx="15" fill="#FFFFFF" stroke="#DFE7F0" />',
    ]
    _append_brand_tile(parts, width=width, y=18, accent="#C8102E")

    start_angle = -math.pi / 2
    for label, value, color in described:
        sweep = (value / total) * math.tau
        end_angle = start_angle + sweep
        x1 = pie_cx + radius * math.cos(start_angle)
        y1 = pie_cy + radius * math.sin(start_angle)
        x2 = pie_cx + radius * math.cos(end_angle)
        y2 = pie_cy + radius * math.sin(end_angle)
        large_arc = 1 if sweep > math.pi else 0
        parts.append(
            f'<path d="M {pie_cx:.1f} {pie_cy:.1f} L {x1:.1f} {y1:.1f} A {radius:.1f} {radius:.1f} 0 {large_arc} 1 {x2:.1f} {y2:.1f} Z" fill="{color}" stroke="#FFFFFF" stroke-width="1.8" filter="url(#{pie_shadow})" />'
        )
        start_angle = end_angle

    parts.append(f'<text x="{pie_cx:.1f}" y="{pie_cy + radius + 18:.1f}" text-anchor="middle" font-size="8.0" font-weight="800" fill="#64748B" letter-spacing="0.08em">VERDELING</text>')
    for index, (label, value, color) in enumerate(described):
        percentage = (value / total) * 100 if total > 0 else 0.0
        row_y = legend_top + index * legend_row_height
        parts.append(f'<rect x="{legend_x:.1f}" y="{row_y - 7:.1f}" width="10" height="10" rx="2" fill="{color}" />')
        parts.append(f'<text x="{legend_x + 16:.1f}" y="{row_y:.1f}" font-size="8.4" font-weight="700" fill="#334155">{escape(label)}</text>')
        parts.append(f'<text x="{width - 24:.1f}" y="{row_y:.1f}" text-anchor="end" font-size="8.4" font-weight="800" fill="#0F172A">{escape(_fmt_dec(percentage, 1))}%</text>')
        parts.append(f'<text x="{width - 24:.1f}" y="{row_y + 9:.1f}" text-anchor="end" font-size="7.1" fill="#64748B">{escape(_fmt_distance_km(value))}</text>')

    _append_chart_footer(parts, width=width, height=height, accent="#C8102E")
    parts.append("</svg>")
    return "".join(parts)


def _event_fill_color(event_group: object, session_code: object) -> str:
    group = str(event_group or "Training").strip().lower()
    code = str(session_code or "").strip().upper()
    index = 1
    if code[-1:].isdigit():
        index = int(code[-1])
    if group == "match":
        palette = ["#6E1222", "#9F2440", "#C8102E"]
    else:
        palette = ["#C8102E", "#EA3351", "#F59E0B"]
    return palette[(index - 1) % len(palette)]


def _build_session_metric_chart_svg(
    title: str,
    session_df: pd.DataFrame,
    value_column: str,
    *,
    width: int = 860,
    height: int = 248,
    y_max: float | None = None,
    formatter: Callable[[object], str] = _fmt_int,
    footer_text: str = "Training en wedstrijd worden per dag gegroepeerd getoond.",
) -> str:
    if not isinstance(session_df, pd.DataFrame) or session_df.empty or value_column not in session_df.columns:
        return _empty_svg(title, "Geen data beschikbaar.", width=width, height=height)

    events = session_df.copy()
    events[value_column] = pd.to_numeric(events[value_column], errors="coerce").fillna(0.0)
    events = events[events[value_column].gt(0)].copy()
    if events.empty:
        return _empty_svg(title, "Geen data beschikbaar.", width=width, height=height)

    day_groups = list(events.groupby("day_label", sort=False))
    if not day_groups:
        return _empty_svg(title, "Geen data beschikbaar.", width=width, height=height)

    clean_values = events[value_column].tolist()
    day_totals = [float(rows[value_column].sum()) for _, rows in day_groups]
    chart_max = y_max if y_max is not None else _nice_max(max(clean_values + day_totals))
    margin_left, margin_right, margin_top, margin_bottom = 56, 24, 92, 52
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    cluster_width = plot_width / max(1, len(day_groups))
    grid_lines = 4
    label_font = 8.6 if len(day_groups) > 5 else 9.3
    avg_value = sum(clean_values) / max(1, len(clean_values))
    peak_value = max(day_totals)
    last_value = day_totals[-1]
    panel_id = _chart_id(title, "session-panel")
    plot_id = _chart_id(title, "session-plot")
    area_id = _chart_id(title, "session-area")
    line_shadow_id = _chart_id(title, "session-line-shadow")
    shadow_id = _chart_id(title, "session-shadow")
    badge = _chart_badge(title)

    parts: list[str] = [
        f'<svg class="chart-svg" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{escape(title)}">',
        "<defs>",
        f'<linearGradient id="{panel_id}" x1="0" y1="0" x2="1" y2="1">',
        '<stop offset="0%" stop-color="#FFFFFF" />',
        '<stop offset="100%" stop-color="#F7FAFD" />',
        "</linearGradient>",
        f'<linearGradient id="{plot_id}" x1="0" y1="0" x2="0" y2="1">',
        '<stop offset="0%" stop-color="#FFFFFF" />',
        '<stop offset="100%" stop-color="#F6F8FC" />',
        "</linearGradient>",
        f'<linearGradient id="{area_id}" x1="0" y1="0" x2="0" y2="1">',
        '<stop offset="0%" stop-color="rgba(200,16,46,0.34)" />',
        '<stop offset="50%" stop-color="rgba(200,16,46,0.12)" />',
        '<stop offset="100%" stop-color="rgba(200,16,46,0.02)" />',
        "</linearGradient>",
        f'<filter id="{line_shadow_id}" x="-10%" y="-10%" width="140%" height="150%">',
        '<feDropShadow dx="0" dy="4" stdDeviation="4" flood-color="rgba(110,18,34,0.18)" />',
        "</filter>",
        f'<filter id="{shadow_id}" x="-10%" y="-10%" width="140%" height="160%">',
        '<feDropShadow dx="0" dy="2" stdDeviation="2.2" flood-color="rgba(15,23,42,0.14)" />',
        "</filter>",
        "</defs>",
        f'<rect x="8" y="10" width="{width - 16}" height="{height - 18}" rx="18" fill="url(#{panel_id})" stroke="#D8E1EC" />',
        f'<rect x="16" y="18" width="40" height="20" rx="3" fill="#C8102E" />',
        f'<text x="36" y="32" text-anchor="middle" font-size="8.8" font-weight="800" fill="#FFFFFF">{escape(badge)}</text>',
        f'<text x="24" y="58" font-size="20" font-weight="800" fill="#0F172A">{escape(title)}</text>',
        f'<text x="24" y="74" font-size="8.3" font-weight="800" fill="#C8102E" letter-spacing="0.05em">AREA + EVENT BARS | {escape(footer_text.upper())}</text>',
        f'<line x1="24" y1="82" x2="{width - 24}" y2="82" stroke="#E7EDF4" />',
        f'<rect x="{margin_left - 8:.1f}" y="{margin_top - 8:.1f}" width="{plot_width + 16:.1f}" height="{plot_height + 16:.1f}" rx="16" fill="url(#{plot_id})" stroke="#DFE7F0" />',
    ]
    _append_brand_tile(parts, width=width, y=18, accent="#C8102E")
    _append_stat_chip(parts, x=width - 314, y=18, label="AVG EVENT", value=formatter(avg_value), fill=_alpha_hex("#C8102E", 0.08), stroke=_alpha_hex("#C8102E", 0.20))
    _append_stat_chip(parts, x=width - 220, y=18, label="PEAK DAY", value=formatter(peak_value), fill=_alpha_hex("#F59E0B", 0.10), stroke=_alpha_hex("#F59E0B", 0.24))
    _append_stat_chip(parts, x=width - 126, y=18, label="LAST DAY", value=formatter(last_value), fill=_alpha_hex("#0F766E", 0.08), stroke=_alpha_hex("#0F766E", 0.20))

    legend_items = [("Training", "#C8102E"), ("Match", "#6E1222")]
    legend_x = 24
    for label, color in legend_items:
        pill_w = 74
        parts.append(f'<rect x="{legend_x:.1f}" y="84" width="{pill_w}" height="18" rx="9" fill="{_alpha_hex(color, 0.08)}" stroke="{_alpha_hex(color, 0.24)}" />')
        parts.append(f'<circle cx="{legend_x + 10:.1f}" cy="93" r="3.8" fill="{color}" />')
        parts.append(f'<text x="{legend_x + 18:.1f}" y="96" font-size="8.0" font-weight="800" fill="#475569">{escape(label)}</text>')
        legend_x += pill_w + 8
    parts.append(f'<text x="{legend_x + 6:.1f}" y="96" font-size="7.4" font-weight="700" fill="#64748B">T/M tonen de events binnen dezelfde dag</text>')

    for step in range(grid_lines + 1):
        ratio = step / grid_lines
        y = margin_top + plot_height - ratio * plot_height
        axis_value = chart_max * ratio
        parts.append(f'<line x1="{margin_left:.1f}" y1="{y:.1f}" x2="{width - margin_right:.1f}" y2="{y:.1f}" stroke="#E7EDF4" stroke-dasharray="3 6" />')
        parts.append(f'<text x="{margin_left - 10:.1f}" y="{y + 3:.1f}" text-anchor="end" font-size="8.6" fill="#708199">{escape(_fmt_axis(axis_value))}</text>')

    line_points = [
        (
            margin_left + cluster_width * index + cluster_width / 2,
            margin_top + plot_height - (0 if chart_max <= 0 else (value / chart_max) * plot_height),
        )
        for index, value in enumerate(day_totals)
    ]
    peak_index = max(range(len(day_totals)), key=lambda idx: day_totals[idx]) if day_totals else 0
    if len(line_points) >= 2:
        peak_center = line_points[peak_index][0]
        parts.append(
            f'<rect x="{peak_center - cluster_width * 0.42:.1f}" y="{margin_top + 6:.1f}" width="{cluster_width * 0.84:.1f}" height="{plot_height - 6:.1f}" rx="14" fill="{_alpha_hex("#C8102E", 0.05)}" />'
        )
        area_points = [(line_points[0][0], margin_top + plot_height)] + line_points + [(line_points[-1][0], margin_top + plot_height)]
        parts.append(f'<path d="{_series_path(area_points)} Z" fill="url(#{area_id})" />')

    for cluster_index, (day_label, rows) in enumerate(day_groups):
        rows = rows.reset_index(drop=True)
        cluster_x = margin_left + cluster_width * cluster_index
        if cluster_index % 2 == 0:
            parts.append(
                f'<rect x="{cluster_x + 4:.1f}" y="{margin_top + 4:.1f}" width="{cluster_width - 8:.1f}" height="{plot_height - 8:.1f}" rx="14" fill="{_alpha_hex("#E2E8F0", 0.24)}" />'
            )
        count = len(rows.index)
        gap = 8
        usable_width = cluster_width * 0.74
        bar_width = min(34, max(18, (usable_width - gap * max(0, count - 1)) / max(1, count)))
        total_width = count * bar_width + gap * max(0, count - 1)
        start_x = cluster_x + (cluster_width - total_width) / 2
        for row_index, (_, row) in enumerate(rows.iterrows()):
            value = float(row.get(value_column) or 0.0)
            bar_height = 0 if chart_max <= 0 else (value / chart_max) * plot_height
            bar_x = start_x + row_index * (bar_width + gap)
            bar_y = margin_top + plot_height - bar_height
            bar_center = bar_x + bar_width / 2
            color = _event_fill_color(row.get("event_group"), row.get("session_code"))
            parts.append(
                f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="7" fill="{color}" filter="url(#{shadow_id})" />'
            )
            parts.append(
                f'<text x="{bar_center:.1f}" y="{max(bar_y - 8, margin_top + 10):.1f}" text-anchor="middle" font-size="8.5" font-weight="800" fill="#0F172A">{escape(formatter(value))}</text>'
            )
            parts.append(
                f'<rect x="{bar_center - 11:.1f}" y="{height - 38:.1f}" width="22" height="12" rx="6" fill="{_alpha_hex(color, 0.12)}" stroke="{_alpha_hex(color, 0.26)}" />'
            )
            parts.append(
                f'<text x="{bar_center:.1f}" y="{height - 29:.1f}" text-anchor="middle" font-size="7.2" font-weight="800" fill="{color}">{escape(_fmt_text(row.get("session_code_display")))}</text>'
            )
        parts.append(
            f'<text x="{cluster_x + cluster_width / 2:.1f}" y="{height - 14:.1f}" text-anchor="middle" font-size="{label_font}" font-weight="700" fill="#55657E">{escape(_fmt_text(day_label))}</text>'
        )

    if avg_value > 0 and chart_max > 0:
        avg_y = margin_top + plot_height - (avg_value / chart_max) * plot_height
        parts.append(
            f'<line x1="{margin_left:.1f}" y1="{avg_y:.1f}" x2="{width - margin_right:.1f}" y2="{avg_y:.1f}" stroke="{_alpha_hex("#6E1222", 0.72)}" stroke-width="1.6" stroke-dasharray="8 6" />'
        )
        parts.append(
            f'<text x="{width - margin_right - 4:.1f}" y="{avg_y - 6:.1f}" text-anchor="end" font-size="8.5" font-weight="800" fill="#6E1222">Avg event {escape(formatter(avg_value))}</text>'
        )

    if len(line_points) >= 2:
        parts.append(
            f'<path d="{_series_path(line_points)}" fill="none" stroke="#FFFFFF" stroke-width="7.6" stroke-linecap="round" stroke-linejoin="round" opacity="0.96" />'
        )
        parts.append(
            f'<path d="{_series_path(line_points)}" fill="none" stroke="#6E1222" stroke-width="3.8" stroke-linecap="round" stroke-linejoin="round" filter="url(#{line_shadow_id})" />'
        )
        for index, (x_center, y_value) in enumerate(line_points):
            highlight = index == peak_index
            point_fill = "#FFF7ED" if highlight else "#FFFFFF"
            point_stroke = "#D97706" if highlight else "#6E1222"
            point_radius = 6.8 if highlight else 5.4
            parts.append(
                f'<circle cx="{x_center:.1f}" cy="{y_value:.1f}" r="{point_radius:.1f}" fill="{point_fill}" stroke="{point_stroke}" stroke-width="2.1" />'
            )
            parts.append(
                f'<circle cx="{x_center:.1f}" cy="{y_value:.1f}" r="2.5" fill="{point_stroke}" />'
            )
            parts.append(
                f'<text x="{x_center:.1f}" y="{max(y_value - 12, margin_top + 10):.1f}" text-anchor="middle" font-size="8.5" font-weight="800" fill="#0F172A">{escape(formatter(day_totals[index]))}</text>'
            )

    _append_chart_footer(parts, width=width, height=height, accent="#C8102E")
    parts.append("</svg>")
    return "".join(parts)


def _build_session_metric_error_chart_svg(
    title: str,
    session_stats: pd.DataFrame,
    mean_column: str,
    error_column: str,
    *,
    width: int = 860,
    height: int = 248,
    y_max: float | None = None,
    formatter: Callable[[object], str] = _fmt_int,
    footer_text: str = "Gemiddelde per speler met standaarddeviatie per sessie-event.",
) -> str:
    if (
        not isinstance(session_stats, pd.DataFrame)
        or session_stats.empty
        or mean_column not in session_stats.columns
        or error_column not in session_stats.columns
    ):
        return _empty_svg(title, "Geen data beschikbaar.", width=width, height=height)

    events = session_stats.copy()
    events[mean_column] = pd.to_numeric(events[mean_column], errors="coerce").fillna(0.0)
    events[error_column] = pd.to_numeric(events[error_column], errors="coerce").fillna(0.0)
    events = events[events[mean_column].gt(0)].copy()
    if events.empty:
        return _empty_svg(title, "Geen data beschikbaar.", width=width, height=height)

    maxima = (events[mean_column] + events[error_column]).tolist()
    chart_max = y_max if y_max is not None else _nice_max(max(maxima))
    margin_left, margin_right, margin_top, margin_bottom = 56, 24, 92, 52
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    day_groups = list(events.groupby("day_label", sort=False))
    cluster_width = plot_width / max(1, len(day_groups))
    grid_lines = 4
    avg_value = float(events[mean_column].mean())
    avg_sd = float(events[error_column].mean())
    day_mean_values = [float(rows[mean_column].mean()) for _, rows in day_groups]
    panel_id = _chart_id(title, "session-sd-panel")
    plot_id = _chart_id(title, "session-sd-plot")
    area_id = _chart_id(title, "session-sd-area")
    line_shadow_id = _chart_id(title, "session-sd-line-shadow")
    shadow_id = _chart_id(title, "session-sd-shadow")
    badge = _chart_badge(title)

    parts: list[str] = [
        f'<svg class="chart-svg" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{escape(title)}">',
        "<defs>",
        f'<linearGradient id="{panel_id}" x1="0" y1="0" x2="1" y2="1">',
        '<stop offset="0%" stop-color="#FFFFFF" />',
        '<stop offset="100%" stop-color="#F7FAFD" />',
        "</linearGradient>",
        f'<linearGradient id="{plot_id}" x1="0" y1="0" x2="0" y2="1">',
        '<stop offset="0%" stop-color="#FFFFFF" />',
        '<stop offset="100%" stop-color="#F6F8FC" />',
        "</linearGradient>",
        f'<linearGradient id="{area_id}" x1="0" y1="0" x2="0" y2="1">',
        '<stop offset="0%" stop-color="rgba(110,18,34,0.26)" />',
        '<stop offset="54%" stop-color="rgba(110,18,34,0.10)" />',
        '<stop offset="100%" stop-color="rgba(110,18,34,0.02)" />',
        "</linearGradient>",
        f'<filter id="{line_shadow_id}" x="-10%" y="-10%" width="140%" height="150%">',
        '<feDropShadow dx="0" dy="4" stdDeviation="4" flood-color="rgba(110,18,34,0.18)" />',
        "</filter>",
        f'<filter id="{shadow_id}" x="-10%" y="-10%" width="140%" height="160%">',
        '<feDropShadow dx="0" dy="2" stdDeviation="2.2" flood-color="rgba(15,23,42,0.14)" />',
        "</filter>",
        "</defs>",
        f'<rect x="8" y="10" width="{width - 16}" height="{height - 18}" rx="18" fill="url(#{panel_id})" stroke="#D8E1EC" />',
        f'<rect x="16" y="18" width="40" height="20" rx="3" fill="#C8102E" />',
        f'<text x="36" y="32" text-anchor="middle" font-size="8.8" font-weight="800" fill="#FFFFFF">{escape(badge)}</text>',
        f'<text x="24" y="58" font-size="20" font-weight="800" fill="#0F172A">{escape(title)}</text>',
        f'<text x="24" y="74" font-size="8.3" font-weight="800" fill="#C8102E" letter-spacing="0.05em">TREND + SPREAD | {escape(footer_text.upper())}</text>',
        f'<line x1="24" y1="82" x2="{width - 24}" y2="82" stroke="#E7EDF4" />',
        f'<rect x="{margin_left - 8:.1f}" y="{margin_top - 8:.1f}" width="{plot_width + 16:.1f}" height="{plot_height + 16:.1f}" rx="16" fill="url(#{plot_id})" stroke="#DFE7F0" />',
    ]
    _append_brand_tile(parts, width=width, y=18, accent="#C8102E")
    _append_stat_chip(parts, x=width - 314, y=18, label="AVG", value=formatter(avg_value), fill=_alpha_hex("#C8102E", 0.08), stroke=_alpha_hex("#C8102E", 0.20))
    _append_stat_chip(parts, x=width - 220, y=18, label="PEAK", value=formatter(events[mean_column].max()), fill=_alpha_hex("#F59E0B", 0.10), stroke=_alpha_hex("#F59E0B", 0.24))
    _append_stat_chip(parts, x=width - 126, y=18, label="AVG SD", value=formatter(avg_sd), fill=_alpha_hex("#0F766E", 0.08), stroke=_alpha_hex("#0F766E", 0.20))

    legend_x = 24
    for label, color in [("Training", "#C8102E"), ("Match", "#6E1222")]:
        parts.append(f'<rect x="{legend_x:.1f}" y="84" width="74" height="18" rx="9" fill="{_alpha_hex(color, 0.08)}" stroke="{_alpha_hex(color, 0.24)}" />')
        parts.append(f'<circle cx="{legend_x + 10:.1f}" cy="93" r="3.8" fill="{color}" />')
        parts.append(f'<text x="{legend_x + 18:.1f}" y="96" font-size="8.0" font-weight="800" fill="#475569">{escape(label)}</text>')
        legend_x += 82

    for step in range(grid_lines + 1):
        ratio = step / grid_lines
        y = margin_top + plot_height - ratio * plot_height
        axis_value = chart_max * ratio
        parts.append(f'<line x1="{margin_left:.1f}" y1="{y:.1f}" x2="{width - margin_right:.1f}" y2="{y:.1f}" stroke="#E7EDF4" stroke-dasharray="3 6" />')
        parts.append(f'<text x="{margin_left - 10:.1f}" y="{y + 3:.1f}" text-anchor="end" font-size="8.6" fill="#708199">{escape(_fmt_axis(axis_value))}</text>')

    trend_points = [
        (
            margin_left + cluster_width * index + cluster_width / 2,
            margin_top + plot_height - (0 if chart_max <= 0 else (value / chart_max) * plot_height),
        )
        for index, value in enumerate(day_mean_values)
    ]
    peak_index = max(range(len(day_mean_values)), key=lambda idx: day_mean_values[idx]) if day_mean_values else 0
    if len(trend_points) >= 2:
        peak_center = trend_points[peak_index][0]
        parts.append(
            f'<rect x="{peak_center - cluster_width * 0.42:.1f}" y="{margin_top + 6:.1f}" width="{cluster_width * 0.84:.1f}" height="{plot_height - 6:.1f}" rx="14" fill="{_alpha_hex("#6E1222", 0.05)}" />'
        )
        area_points = [(trend_points[0][0], margin_top + plot_height)] + trend_points + [(trend_points[-1][0], margin_top + plot_height)]
        parts.append(f'<path d="{_series_path(area_points)} Z" fill="url(#{area_id})" />')

    for cluster_index, (day_label, rows) in enumerate(day_groups):
        rows = rows.reset_index(drop=True)
        cluster_x = margin_left + cluster_width * cluster_index
        if cluster_index % 2 == 0:
            parts.append(
                f'<rect x="{cluster_x + 4:.1f}" y="{margin_top + 4:.1f}" width="{cluster_width - 8:.1f}" height="{plot_height - 8:.1f}" rx="14" fill="{_alpha_hex("#E2E8F0", 0.24)}" />'
            )
        count = len(rows.index)
        gap = 8
        usable_width = cluster_width * 0.74
        bar_width = min(34, max(18, (usable_width - gap * max(0, count - 1)) / max(1, count)))
        total_width = count * bar_width + gap * max(0, count - 1)
        start_x = cluster_x + (cluster_width - total_width) / 2
        for row_index, (_, row) in enumerate(rows.iterrows()):
            value = float(row.get(mean_column) or 0.0)
            error = float(row.get(error_column) or 0.0)
            bar_height = 0 if chart_max <= 0 else (value / chart_max) * plot_height
            bar_x = start_x + row_index * (bar_width + gap)
            bar_y = margin_top + plot_height - bar_height
            bar_center = bar_x + bar_width / 2
            color = _event_fill_color(row.get("event_group"), row.get("session_code"))
            upper_y = margin_top + plot_height - (min(chart_max, value + error) / chart_max) * plot_height
            lower_y = margin_top + plot_height - (max(0.0, value - error) / chart_max) * plot_height
            parts.append(f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="7" fill="{color}" filter="url(#{shadow_id})" />')
            parts.append(f'<line x1="{bar_center:.1f}" y1="{upper_y:.1f}" x2="{bar_center:.1f}" y2="{lower_y:.1f}" stroke="{_alpha_hex(color, 0.58)}" stroke-width="1.5" />')
            parts.append(f'<line x1="{bar_center - 5:.1f}" y1="{upper_y:.1f}" x2="{bar_center + 5:.1f}" y2="{upper_y:.1f}" stroke="{_alpha_hex(color, 0.58)}" stroke-width="1.5" />')
            parts.append(f'<line x1="{bar_center - 5:.1f}" y1="{lower_y:.1f}" x2="{bar_center + 5:.1f}" y2="{lower_y:.1f}" stroke="{_alpha_hex(color, 0.58)}" stroke-width="1.5" />')
            parts.append(f'<text x="{bar_center:.1f}" y="{max(bar_y - 8, margin_top + 10):.1f}" text-anchor="middle" font-size="8.5" font-weight="800" fill="#0F172A">{escape(formatter(value))}</text>')
            parts.append(f'<rect x="{bar_center - 11:.1f}" y="{height - 38:.1f}" width="22" height="12" rx="6" fill="{_alpha_hex(color, 0.12)}" stroke="{_alpha_hex(color, 0.26)}" />')
            parts.append(f'<text x="{bar_center:.1f}" y="{height - 29:.1f}" text-anchor="middle" font-size="7.2" font-weight="800" fill="{color}">{escape(_fmt_text(row.get("session_code_display")))}</text>')
        parts.append(f'<text x="{cluster_x + cluster_width / 2:.1f}" y="{height - 14:.1f}" text-anchor="middle" font-size="8.9" font-weight="700" fill="#55657E">{escape(_fmt_text(day_label))}</text>')

    if avg_value > 0 and chart_max > 0:
        avg_y = margin_top + plot_height - (avg_value / chart_max) * plot_height
        parts.append(
            f'<line x1="{margin_left:.1f}" y1="{avg_y:.1f}" x2="{width - margin_right:.1f}" y2="{avg_y:.1f}" stroke="{_alpha_hex("#6E1222", 0.70)}" stroke-width="1.5" stroke-dasharray="8 6" />'
        )
        parts.append(
            f'<text x="{width - margin_right - 4:.1f}" y="{avg_y - 6:.1f}" text-anchor="end" font-size="8.5" font-weight="800" fill="#6E1222">Avg {_fmt_text(formatter(avg_value))}</text>'
        )

    if len(trend_points) >= 2:
        parts.append(f'<path d="{_series_path(trend_points)}" fill="none" stroke="#FFFFFF" stroke-width="6.8" stroke-linecap="round" stroke-linejoin="round" opacity="0.96" />')
        parts.append(f'<path d="{_series_path(trend_points)}" fill="none" stroke="#6E1222" stroke-width="3.3" stroke-linecap="round" stroke-linejoin="round" filter="url(#{line_shadow_id})" />')
        for index, (x_center, y_value) in enumerate(trend_points):
            highlight = index == peak_index
            point_fill = "#FFF7ED" if highlight else "#FFFFFF"
            point_stroke = "#D97706" if highlight else "#6E1222"
            parts.append(f'<circle cx="{x_center:.1f}" cy="{y_value:.1f}" r="{5.6 if highlight else 5.0:.1f}" fill="{point_fill}" stroke="{point_stroke}" stroke-width="1.9" />')
            parts.append(f'<circle cx="{x_center:.1f}" cy="{y_value:.1f}" r="2.2" fill="{point_stroke}" />')

    _append_chart_footer(parts, width=width, height=height, accent="#C8102E")
    parts.append("</svg>")
    return "".join(parts)


def _build_session_dual_metric_error_chart_svg(
    title: str,
    session_stats: pd.DataFrame,
    left_spec: dict[str, Any],
    right_spec: dict[str, Any],
    *,
    width: int = 860,
    height: int = 248,
    y_max: float | None = None,
    footer_text: str = "Twee outputmaten per sessie-event met standaarddeviatie per speler.",
) -> str:
    if not isinstance(session_stats, pd.DataFrame) or session_stats.empty:
        return _empty_svg(title, "Geen data beschikbaar.", width=width, height=height)

    events = session_stats.copy()
    specs = [left_spec, right_spec]
    for spec in specs:
        events[spec["mean"]] = pd.to_numeric(events[spec["mean"]], errors="coerce").fillna(0.0)
        events[spec["std"]] = pd.to_numeric(events[spec["std"]], errors="coerce").fillna(0.0)
    if events[[left_spec["mean"], right_spec["mean"]]].max().max() <= 0:
        return _empty_svg(title, "Geen data beschikbaar.", width=width, height=height)

    maxima: list[float] = []
    for spec in specs:
        maxima.extend((events[spec["mean"]] + events[spec["std"]]).tolist())
    chart_max = y_max if y_max is not None else _nice_max(max(maxima))
    margin_left, margin_right, margin_top, margin_bottom = 56, 24, 92, 52
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    day_groups = list(events.groupby("day_label", sort=False))
    cluster_width = plot_width / max(1, len(day_groups))
    grid_lines = 4
    left_day_means = [float(rows[left_spec["mean"]].mean()) for _, rows in day_groups]
    right_day_means = [float(rows[right_spec["mean"]].mean()) for _, rows in day_groups]
    panel_id = _chart_id(title, "dual-panel")
    plot_id = _chart_id(title, "dual-plot")
    left_line_shadow = _chart_id(title, "dual-left-line-shadow")
    right_line_shadow = _chart_id(title, "dual-right-line-shadow")
    shadow_id = _chart_id(title, "dual-shadow")
    badge = _chart_badge(title)

    parts: list[str] = [
        f'<svg class="chart-svg" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{escape(title)}">',
        "<defs>",
        f'<linearGradient id="{panel_id}" x1="0" y1="0" x2="1" y2="1">',
        '<stop offset="0%" stop-color="#FFFFFF" />',
        '<stop offset="100%" stop-color="#F7FAFD" />',
        "</linearGradient>",
        f'<linearGradient id="{plot_id}" x1="0" y1="0" x2="0" y2="1">',
        '<stop offset="0%" stop-color="#FFFFFF" />',
        '<stop offset="100%" stop-color="#F6F8FC" />',
        "</linearGradient>",
        f'<filter id="{left_line_shadow}" x="-10%" y="-10%" width="140%" height="150%">',
        '<feDropShadow dx="0" dy="4" stdDeviation="4" flood-color="rgba(110,18,34,0.16)" />',
        "</filter>",
        f'<filter id="{right_line_shadow}" x="-10%" y="-10%" width="140%" height="150%">',
        '<feDropShadow dx="0" dy="4" stdDeviation="4" flood-color="rgba(234,51,81,0.14)" />',
        "</filter>",
        f'<filter id="{shadow_id}" x="-10%" y="-10%" width="140%" height="160%">',
        '<feDropShadow dx="0" dy="2" stdDeviation="2.2" flood-color="rgba(15,23,42,0.14)" />',
        "</filter>",
        "</defs>",
        f'<rect x="8" y="10" width="{width - 16}" height="{height - 18}" rx="18" fill="url(#{panel_id})" stroke="#D8E1EC" />',
        f'<rect x="16" y="18" width="40" height="20" rx="3" fill="#C8102E" />',
        f'<text x="36" y="32" text-anchor="middle" font-size="8.8" font-weight="800" fill="#FFFFFF">{escape(badge)}</text>',
        f'<text x="24" y="58" font-size="20" font-weight="800" fill="#0F172A">{escape(title)}</text>',
        f'<text x="24" y="74" font-size="8.3" font-weight="800" fill="#C8102E" letter-spacing="0.05em">DUAL TREND + SPREAD | {escape(footer_text.upper())}</text>',
        f'<line x1="24" y1="82" x2="{width - 24}" y2="82" stroke="#E7EDF4" />',
        f'<rect x="{margin_left - 8:.1f}" y="{margin_top - 8:.1f}" width="{plot_width + 16:.1f}" height="{plot_height + 16:.1f}" rx="16" fill="url(#{plot_id})" stroke="#DFE7F0" />',
    ]
    _append_brand_tile(parts, width=width, y=18, accent="#C8102E")
    _append_stat_chip(parts, x=width - 314, y=18, label="AVG A", value=_fmt_dec(sum(left_day_means) / max(1, len(left_day_means)), 1), fill=_alpha_hex(str(left_spec["color"]), 0.10), stroke=_alpha_hex(str(left_spec["color"]), 0.24))
    _append_stat_chip(parts, x=width - 220, y=18, label="AVG B", value=_fmt_dec(sum(right_day_means) / max(1, len(right_day_means)), 1), fill=_alpha_hex(str(right_spec["color"]), 0.10), stroke=_alpha_hex(str(right_spec["color"]), 0.24))
    _append_stat_chip(parts, x=width - 126, y=18, label="PEAK MIX", value=_fmt_dec(max(max(left_day_means, default=0.0), max(right_day_means, default=0.0)), 1), fill=_alpha_hex("#F59E0B", 0.10), stroke=_alpha_hex("#F59E0B", 0.24))

    legend_x = 24
    for spec in specs:
        color = str(spec["color"])
        label = str(spec["label"])
        pill_w = max(86, len(label) * 7 + 22)
        parts.append(f'<rect x="{legend_x:.1f}" y="84" width="{pill_w:.1f}" height="18" rx="9" fill="{_alpha_hex(color, 0.08)}" stroke="{_alpha_hex(color, 0.24)}" />')
        parts.append(f'<circle cx="{legend_x + 10:.1f}" cy="93" r="3.8" fill="{color}" />')
        parts.append(f'<text x="{legend_x + 18:.1f}" y="96" font-size="8.0" font-weight="800" fill="#475569">{escape(label)}</text>')
        legend_x += pill_w + 8

    for step in range(grid_lines + 1):
        ratio = step / grid_lines
        y = margin_top + plot_height - ratio * plot_height
        axis_value = chart_max * ratio
        parts.append(f'<line x1="{margin_left:.1f}" y1="{y:.1f}" x2="{width - margin_right:.1f}" y2="{y:.1f}" stroke="#E7EDF4" stroke-dasharray="3 6" />')
        parts.append(f'<text x="{margin_left - 10:.1f}" y="{y + 3:.1f}" text-anchor="end" font-size="8.6" fill="#708199">{escape(_fmt_axis(axis_value))}</text>')

    left_points = [
        (
            margin_left + cluster_width * index + cluster_width / 2,
            margin_top + plot_height - (0 if chart_max <= 0 else (value / chart_max) * plot_height),
        )
        for index, value in enumerate(left_day_means)
    ]
    right_points = [
        (
            margin_left + cluster_width * index + cluster_width / 2,
            margin_top + plot_height - (0 if chart_max <= 0 else (value / chart_max) * plot_height),
        )
        for index, value in enumerate(right_day_means)
    ]
    for cluster_index, (day_label, rows) in enumerate(day_groups):
        rows = rows.reset_index(drop=True)
        cluster_x = margin_left + cluster_width * cluster_index
        if cluster_index % 2 == 0:
            parts.append(f'<rect x="{cluster_x + 4:.1f}" y="{margin_top + 4:.1f}" width="{cluster_width - 8:.1f}" height="{plot_height - 8:.1f}" rx="14" fill="{_alpha_hex("#E2E8F0", 0.24)}" />')
        count = len(rows.index)
        gap = 10
        event_width = min(56, max(28, (cluster_width * 0.74 - gap * max(0, count - 1)) / max(1, count)))
        total_width = count * event_width + gap * max(0, count - 1)
        start_x = cluster_x + (cluster_width - total_width) / 2
        bar_gap = 4
        inner_bar_width = max(10, (event_width - bar_gap) / 2)
        for row_index, (_, row) in enumerate(rows.iterrows()):
            event_x = start_x + row_index * (event_width + gap)
            event_color = _event_fill_color(row.get("event_group"), row.get("session_code"))
            for spec_index, spec in enumerate(specs):
                mean_value = float(row.get(spec["mean"]) or 0.0)
                error_value = float(row.get(spec["std"]) or 0.0)
                bar_height = 0 if chart_max <= 0 else (mean_value / chart_max) * plot_height
                bar_x = event_x + spec_index * (inner_bar_width + bar_gap)
                bar_y = margin_top + plot_height - bar_height
                bar_center = bar_x + inner_bar_width / 2
                color = str(spec["color"])
                upper_y = margin_top + plot_height - (min(chart_max, mean_value + error_value) / chart_max) * plot_height
                lower_y = margin_top + plot_height - (max(0.0, mean_value - error_value) / chart_max) * plot_height
                parts.append(f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{inner_bar_width:.1f}" height="{bar_height:.1f}" rx="6" fill="{color}" filter="url(#{shadow_id})" />')
                parts.append(f'<line x1="{bar_center:.1f}" y1="{upper_y:.1f}" x2="{bar_center:.1f}" y2="{lower_y:.1f}" stroke="{_alpha_hex(color, 0.58)}" stroke-width="1.4" />')
                parts.append(f'<line x1="{bar_center - 4:.1f}" y1="{upper_y:.1f}" x2="{bar_center + 4:.1f}" y2="{upper_y:.1f}" stroke="{_alpha_hex(color, 0.58)}" stroke-width="1.4" />')
                parts.append(f'<line x1="{bar_center - 4:.1f}" y1="{lower_y:.1f}" x2="{bar_center + 4:.1f}" y2="{lower_y:.1f}" stroke="{_alpha_hex(color, 0.58)}" stroke-width="1.4" />')
            parts.append(f'<rect x="{event_x + event_width / 2 - 11:.1f}" y="{height - 38:.1f}" width="22" height="12" rx="6" fill="{_alpha_hex(event_color, 0.12)}" stroke="{_alpha_hex(event_color, 0.26)}" />')
            parts.append(f'<text x="{event_x + event_width / 2:.1f}" y="{height - 29:.1f}" text-anchor="middle" font-size="7.2" font-weight="800" fill="{event_color}">{escape(_fmt_text(row.get("session_code_display")))}</text>')
        parts.append(f'<text x="{cluster_x + cluster_width / 2:.1f}" y="{height - 14:.1f}" text-anchor="middle" font-size="8.9" font-weight="700" fill="#55657E">{escape(_fmt_text(day_label))}</text>')

    if len(left_points) >= 2:
        parts.append(f'<path d="{_series_path(left_points)}" fill="none" stroke="#FFFFFF" stroke-width="5.8" stroke-linecap="round" stroke-linejoin="round" opacity="0.90" />')
        parts.append(f'<path d="{_series_path(left_points)}" fill="none" stroke="{str(left_spec["color"])}" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round" filter="url(#{left_line_shadow})" opacity="0.88" />')
        for x_center, y_value in left_points:
            parts.append(f'<circle cx="{x_center:.1f}" cy="{y_value:.1f}" r="4.4" fill="#FFFFFF" stroke="{str(left_spec["color"])}" stroke-width="1.7" />')
            parts.append(f'<circle cx="{x_center:.1f}" cy="{y_value:.1f}" r="2.0" fill="{str(left_spec["color"])}" />')
    if len(right_points) >= 2:
        parts.append(f'<path d="{_series_path(right_points)}" fill="none" stroke="#FFFFFF" stroke-width="5.8" stroke-linecap="round" stroke-linejoin="round" opacity="0.82" />')
        parts.append(f'<path d="{_series_path(right_points)}" fill="none" stroke="{str(right_spec["color"])}" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round" filter="url(#{right_line_shadow})" opacity="0.88" />')
        for x_center, y_value in right_points:
            parts.append(f'<circle cx="{x_center:.1f}" cy="{y_value:.1f}" r="4.4" fill="#FFFFFF" stroke="{str(right_spec["color"])}" stroke-width="1.7" />')
            parts.append(f'<circle cx="{x_center:.1f}" cy="{y_value:.1f}" r="2.0" fill="{str(right_spec["color"])}" />')

    _append_chart_footer(parts, width=width, height=height, accent="#C8102E")
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
        zone_days = zone_days.sort_values("datum").reset_index(drop=True)
        for _, row in zone_days.iterrows():
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
    session_table: pd.DataFrame | None = None,
    type_table: pd.DataFrame,
    player_table: pd.DataFrame,
    monitoring_day_table: pd.DataFrame,
    notes: Iterable[str],
    day_stats: pd.DataFrame | None = None,
    session_stats: pd.DataFrame | None = None,
    zone_df: pd.DataFrame | None = None,
    zone_day_table: pd.DataFrame | None = None,
    zone_session_table: pd.DataFrame | None = None,
    rpe_session_day_table: pd.DataFrame | None = None,
    monitoring_player_table: pd.DataFrame | None = None,
    report_revision: str | None = None,
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
        rpe_session_timeline["day_label"] = rpe_session_timeline.get("label", pd.Series(dtype=str)).fillna("").astype(str)
        rpe_session_timeline["session_code"] = rpe_session_timeline.get("session_index", pd.Series(dtype=int)).apply(lambda value: f"R{int(value)}")
        rpe_session_timeline["event_group"] = "Training"
        rpe_session_timeline["events_in_day"] = rpe_session_timeline.groupby("day_label")["day_label"].transform("size")
        rpe_session_timeline["session_code_display"] = rpe_session_timeline.apply(
            lambda row: f"R{int(row.get('session_index', 1) or 1)}" if int(row.get("events_in_day", 1) or 1) > 1 else "R",
            axis=1,
        )
        rpe_session_timeline["axis_label"] = rpe_session_timeline.apply(
            lambda row: f"{_fmt_text(row.get('label'))} S{_fmt_int(row.get('session_index'))}",
            axis=1,
        )
    squad_spread = day_stats.copy() if isinstance(day_stats, pd.DataFrame) else pd.DataFrame()
    session_flow = session_table.copy() if isinstance(session_table, pd.DataFrame) else pd.DataFrame()
    session_spread = session_stats.copy() if isinstance(session_stats, pd.DataFrame) else pd.DataFrame()
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

    revision_label = _fmt_text(report_revision) if report_revision else "REV-HTML"
    export_stamp = pd.Timestamp.now().strftime("%d-%m-%Y %H:%M")

    context = {
        "document_title": f"Week Report | {week_label}",
        "report_title": "Week Report",
        "report_kicker": "MVV Maastricht | Reports | Team Week Overview",
        "report_subtitle": f"{week_label} | {iso_label} | {revision_label}",
        "report_description": "",
        "logo_src": LOGO_SRC,
        "report_header_meta": [
            {"label": "Revision", "value": revision_label, "foot": "Actieve HTML exportbuild"},
            {"label": "Export", "value": export_stamp, "foot": "Gegenereerd op deze builder"},
        ],
        "badges": [f"Export {revision_label}", export_stamp],
        "cards": [
            {"label": "Total Distance", "value": _fmt_distance_km(summary.get("total_distance")), "foot": "Opgetelde teamload in de week"},
            {"label": "HSR / HSD", "value": _fmt_distance_km(summary.get("hsr_hsd")), "foot": "Sprint plus high sprint distance"},
            {"label": "Dist / Player", "value": _fmt_distance_km(summary.get("dist_per_player")), "foot": "Teamload gedeeld door actieve spelers"},
            {"label": "Sprints", "value": _fmt_int(summary.get("sprints")), "foot": "Totale sprintacties in deze week"},
            {"label": "Top Speed", "value": _fmt_speed(summary.get("top_speed")), "foot": "Hoogste gemeten snelheid"},
            {"label": "Speed Exposures", "value": _fmt_int(summary.get("speed_exposures")), "foot": "Spelersessies >= 90% van individuele seizoensmax"},
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
                        "svg": _build_session_metric_chart_svg(
                            "Daily Team Distance",
                            session_flow,
                            "total_distance",
                            formatter=_fmt_distance,
                            footer_text="Trainingen en wedstrijden staan per dag naast elkaar gegroepeerd.",
                        )
                    },
                    {
                        "svg": _build_session_metric_chart_svg(
                            "Daily Team HSR / HSD",
                            session_flow,
                            "hsr_hsd",
                            formatter=_fmt_distance,
                            footer_text="High-speed output uitgesplitst per event in plaats van alleen per dagtotaal.",
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
                        "svg": _build_session_metric_error_chart_svg(
                            "Player Avg Total Distance +/- SD",
                            session_spread,
                            "total_distance_mean",
                            "total_distance_std",
                            formatter=_fmt_distance,
                            footer_text="Per sessie-event: gemiddelde afstand per speler met standaarddeviatie.",
                        )
                    },
                    {
                        "svg": _build_session_metric_error_chart_svg(
                            "Player Avg HSR / HSD +/- SD",
                            session_spread,
                            "hsr_hsd_mean",
                            "hsr_hsd_std",
                            formatter=_fmt_distance,
                            footer_text="Per sessie-event: gemiddelde HSR/HSD per speler met standaarddeviatie.",
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
                        "svg": _build_session_dual_metric_error_chart_svg(
                            "Player Avg Accel / Decel +/- SD",
                            session_spread,
                            {"label": "Accelerations", "color": "#6E1222", "mean": "total_accelerations_mean", "std": "total_accelerations_std"},
                            {"label": "Decelerations", "color": "#EA3351", "mean": "total_decelerations_mean", "std": "total_decelerations_std"},
                            footer_text="Explosieve output per sessie-event, inclusief spreiding per speler.",
                        )
                    },
                    {
                        "svg": _build_session_metric_error_chart_svg(
                            "Player Avg Sprints +/- SD",
                            session_spread,
                            "sprints_mean",
                            "sprints_std",
                            formatter=lambda value: _fmt_dec(value, 1),
                            footer_text="Sprintgemiddelden per speler, los getoond voor elke training of wedstrijd op dezelfde dag.",
                        )
                    },
                ],
            },
            {
                "eyebrow": "Speed profile",
                "title": "Daily speed exposures",
                "subtitle": "Aantal spelersessies per event op of boven 90% van de individuele seizoensmax.",
                "columns": 1,
                "panels": [
                    {
                        "svg": _build_session_metric_chart_svg(
                            "Daily Speed Exposures",
                            session_flow,
                            "speed_exposures",
                            formatter=_fmt_int,
                            footer_text="Speed exposure wordt per training of wedstrijd getoond, niet meer verstopt in een dagtotaal.",
                        )
                    },
                ],
            },
            {
                "eyebrow": "Locomotion zones",
                "title": "Distance zone share",
                "subtitle": "Normale cirkeldiagrammen voor de hele week en voor elke actieve dag binnen dezelfde week.",
                "columns": 3,
                "page_break": True,
                "panels": _build_zone_share_panels(zone_df, zone_day_table),
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
                "subtitle": "Sessie-RPE per dag, met aparte balken wanneer twee sessies op dezelfde dag plaatsvinden.",
                "columns": 1,
                "panels": [
                    {
                        "svg": _build_session_metric_error_chart_svg(
                            "Session RPE +/- SD",
                            rpe_session_timeline,
                            "avg_rpe",
                            "avg_rpe_std",
                            formatter=lambda value: _fmt_dec(value, 1),
                            footer_text="RPE per sessie binnen de dag gegroepeerd, zodat dubbele trainingsmomenten naast elkaar zichtbaar blijven.",
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
                        "Session flow",
                        "Per event binnen de dag: training(en) en wedstrijd(en) naast elkaar.",
                        session_flow,
                        [
                            ("day_label", "Dag", None),
                            ("session_code_display", "Event", None),
                            ("session_display", "Type", None),
                            ("total_distance", "TD", _fmt_distance),
                            ("distance_per_player", "Dist / Player", _fmt_distance),
                            ("hsr_hsd", "HSR / HSD", _fmt_distance),
                            ("sprints", "Sprints", _fmt_int),
                            ("speed_exposures", "Exposures", _fmt_int),
                            ("max_speed", "Top Speed", _fmt_speed),
                        ],
                        empty_message="Geen sessie-overzicht beschikbaar.",
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
                "title": "Top players",
                "subtitle": "Volumeleiders voor de staffbespreking van deze microcycle.",
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
