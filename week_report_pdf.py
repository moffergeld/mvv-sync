from __future__ import annotations

from io import BytesIO
from typing import Iterable

import pandas as pd


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


def build_week_report_pdf_bytes(
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
) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.graphics.charts.barcharts import HorizontalBarChart, VerticalBarChart
    from reportlab.graphics.charts.linecharts import HorizontalLineChart
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.graphics.shapes import Drawing, Line, Rect, String
    from reportlab.graphics.widgets.markers import makeMarker
    from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "week_title",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=24,
        textColor=colors.HexColor("#0B1020"),
        spaceAfter=2,
    )
    kicker_style = ParagraphStyle(
        "week_kicker",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        textColor=colors.HexColor("#C8102E"),
        leading=11,
        spaceAfter=3,
    )
    hero_body_style = ParagraphStyle(
        "week_hero_body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.2,
        leading=12,
        textColor=colors.HexColor("#4C5668"),
        alignment=TA_LEFT,
    )
    section_style = ParagraphStyle(
        "week_section",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        textColor=colors.HexColor("#0B1020"),
        spaceAfter=6,
        spaceBefore=8,
    )
    body_style = ParagraphStyle(
        "week_body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#182134"),
    )
    card_label_style = ParagraphStyle(
        "week_card_label",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=7.2,
        leading=9,
        textColor=colors.HexColor("#6A768B"),
        alignment=TA_LEFT,
    )
    card_value_style = ParagraphStyle(
        "week_card_value",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=15.4,
        leading=17,
        textColor=colors.HexColor("#0B1020"),
        alignment=TA_LEFT,
    )
    card_foot_style = ParagraphStyle(
        "week_card_foot",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=7.1,
        leading=9,
        textColor=colors.HexColor("#7A8598"),
        alignment=TA_LEFT,
    )
    note_style = ParagraphStyle(
        "week_note",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#182134"),
        leftIndent=4,
    )

    card_width = doc.width / 4.0

    def build_metric_card(label: str, value: str, foot: str, background_hex: str, border_hex: str) -> Table:
        card = Table(
            [
                [Paragraph(label.upper(), card_label_style)],
                [Paragraph(value, card_value_style)],
                [Paragraph(foot, card_foot_style)],
            ],
            colWidths=[card_width - 10],
        )
        card.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(background_hex)),
                    ("BOX", (0, 0), (-1, -1), 0.9, colors.HexColor(border_hex)),
                    ("LINEABOVE", (0, 0), (-1, 0), 1.3, colors.HexColor(border_hex)),
                    ("LEFTPADDING", (0, 0), (-1, -1), 9),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        return card

    def build_card_grid(cards: list[Table], columns: int = 4) -> Table:
        rows: list[list[object]] = []
        for index in range(0, len(cards), columns):
            row: list[object] = list(cards[index : index + columns])
            while len(row) < columns:
                row.append("")
            rows.append(row)
        grid = Table(rows, colWidths=[doc.width / columns] * columns, hAlign="LEFT")
        grid.setStyle(
            TableStyle(
                [
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        return grid

    def build_bar_chart_drawing(
        title: str,
        labels: list[str],
        data_series: list[list[float]],
        series_colors: list[str],
        legend_labels: list[str],
        width: float = 352,
        height: float = 220,
    ) -> Drawing:
        drawing = Drawing(width, height)
        drawing.add(Rect(0, 0, width, height, fillColor=colors.HexColor("#FBFCFE"), strokeColor=colors.HexColor("#D7DEE8"), strokeWidth=1))
        drawing.add(String(16, height - 20, title, fontName="Helvetica-Bold", fontSize=11, fillColor=colors.HexColor("#0B1020")))

        chart = VerticalBarChart()
        chart.x = 32
        chart.y = 44
        chart.height = height - 82
        chart.width = width - 56
        chart.data = data_series
        chart.strokeColor = colors.HexColor("#94A3B8")
        chart.valueAxis.valueMin = 0
        peak = max((max(series) if series else 0) for series in data_series)
        chart.valueAxis.valueMax = max(1, peak * 1.18)
        chart.valueAxis.strokeColor = colors.HexColor("#B8C4D3")
        chart.valueAxis.gridStrokeColor = colors.HexColor("#E5EAF1")
        chart.valueAxis.gridStrokeDashArray = [2, 2]
        chart.valueAxis.visibleGrid = True
        chart.valueAxis.labels.fillColor = colors.HexColor("#5B6576")
        chart.valueAxis.labels.fontName = "Helvetica"
        chart.valueAxis.labels.fontSize = 7
        chart.categoryAxis.categoryNames = labels
        chart.categoryAxis.strokeColor = colors.HexColor("#B8C4D3")
        chart.categoryAxis.labels.boxAnchor = "ne"
        chart.categoryAxis.labels.angle = 30
        chart.categoryAxis.labels.dx = -4
        chart.categoryAxis.labels.dy = -2
        chart.categoryAxis.labels.fillColor = colors.HexColor("#5B6576")
        chart.categoryAxis.labels.fontName = "Helvetica"
        chart.categoryAxis.labels.fontSize = 7
        chart.barSpacing = 3
        chart.groupSpacing = 9
        chart.barWidth = 8 if len(data_series) > 1 else 14

        for index, fill_hex in enumerate(series_colors):
            chart.bars[index].fillColor = colors.HexColor(fill_hex)
            chart.bars[index].strokeColor = colors.HexColor(fill_hex)

        drawing.add(chart)

        legend_y = 16
        legend_x = 16
        for fill_hex, legend_label in zip(series_colors, legend_labels):
            drawing.add(Rect(legend_x, legend_y, 8, 8, fillColor=colors.HexColor(fill_hex), strokeColor=colors.HexColor(fill_hex)))
            drawing.add(String(legend_x + 12, legend_y + 1, legend_label, fontName="Helvetica", fontSize=7.5, fillColor=colors.HexColor("#4C5668")))
            legend_x += 96
        return drawing

    def build_line_chart_drawing(
        title: str,
        labels: list[str],
        data_series: list[list[float]],
        series_colors: list[str],
        legend_labels: list[str],
        width: float = 724,
        height: float = 230,
        y_max: float = 10,
    ) -> Drawing:
        drawing = Drawing(width, height)
        drawing.add(Rect(0, 0, width, height, fillColor=colors.HexColor("#FBFCFE"), strokeColor=colors.HexColor("#D7DEE8"), strokeWidth=1))
        drawing.add(String(16, height - 20, title, fontName="Helvetica-Bold", fontSize=11, fillColor=colors.HexColor("#0B1020")))

        chart = HorizontalLineChart()
        chart.x = 32
        chart.y = 42
        chart.height = height - 80
        chart.width = width - 56
        chart.data = data_series
        chart.joinedLines = 1
        chart.valueAxis.valueMin = 0
        chart.valueAxis.valueMax = y_max
        chart.valueAxis.strokeColor = colors.HexColor("#B8C4D3")
        chart.valueAxis.gridStrokeColor = colors.HexColor("#E5EAF1")
        chart.valueAxis.gridStrokeDashArray = [2, 2]
        chart.valueAxis.visibleGrid = True
        chart.valueAxis.labels.fillColor = colors.HexColor("#5B6576")
        chart.valueAxis.labels.fontName = "Helvetica"
        chart.valueAxis.labels.fontSize = 7
        chart.categoryAxis.categoryNames = labels
        chart.categoryAxis.strokeColor = colors.HexColor("#B8C4D3")
        chart.categoryAxis.labels.fillColor = colors.HexColor("#5B6576")
        chart.categoryAxis.labels.fontName = "Helvetica"
        chart.categoryAxis.labels.fontSize = 7
        chart.categoryAxis.labels.angle = 25
        chart.categoryAxis.labels.boxAnchor = "ne"
        chart.categoryAxis.labels.dx = -2

        for index, fill_hex in enumerate(series_colors):
            chart.lines[index].strokeColor = colors.HexColor(fill_hex)
            chart.lines[index].strokeWidth = 2
            chart.lines[index].symbol = makeMarker("FilledCircle")

        drawing.add(chart)

        legend_y = 16
        legend_x = 16
        for fill_hex, legend_label in zip(series_colors, legend_labels):
            drawing.add(Rect(legend_x, legend_y, 8, 8, fillColor=colors.HexColor(fill_hex), strokeColor=colors.HexColor(fill_hex)))
            drawing.add(String(legend_x + 12, legend_y + 1, legend_label, fontName="Helvetica", fontSize=7.5, fillColor=colors.HexColor("#4C5668")))
            legend_x += 96
        return drawing

    def build_standard_table(rows: list[list[object]], col_widths: list[float], header_hex: str, body_hex: str, alt_hex: str) -> Table:
        table = Table(rows, colWidths=col_widths, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_hex)),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor(body_hex)),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor(body_hex), colors.HexColor(alt_hex)]),
                    ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#182134")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D7DEE8")),
                    ("FONTSIZE", (0, 0), (-1, -1), 7.6),
                    ("LEADING", (0, 0), (-1, -1), 9.2),
                    ("TOPPADDING", (0, 0), (-1, -1), 4.2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4.2),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        return table

    def _series_to_floats(values: object) -> list[float]:
        if values is None:
            return []
        if isinstance(values, pd.Series):
            return pd.to_numeric(values, errors="coerce").fillna(0).astype(float).tolist()
        return pd.to_numeric(pd.Series(values), errors="coerce").fillna(0).astype(float).tolist()

    def build_vertical_error_chart_drawing(
        title: str,
        labels: list[str],
        values: list[float],
        errors: list[float] | None,
        bar_color: str,
        legend_label: str,
        width: float = 352,
        height: float = 220,
        y_max: float | None = None,
    ) -> Drawing:
        drawing = Drawing(width, height)
        drawing.add(Rect(0, 0, width, height, fillColor=colors.HexColor("#FBFCFE"), strokeColor=colors.HexColor("#D7DEE8"), strokeWidth=1))
        drawing.add(String(16, height - 20, title, fontName="Helvetica-Bold", fontSize=11, fillColor=colors.HexColor("#0B1020")))

        labels = [str(value) for value in labels]
        values = [max(float(value or 0), 0.0) for value in values]
        errors = [max(float(value or 0), 0.0) for value in (errors or [0.0] * len(values))]

        plot_x = 40
        plot_y = 44
        plot_w = width - 58
        plot_h = height - 84
        max_value = max((value + err for value, err in zip(values, errors, strict=False)), default=0.0)
        chart_max = float(y_max) if y_max is not None else max(1.0, max_value * 1.18)

        for step in range(6):
            ratio = step / 5.0
            y = plot_y + (plot_h * ratio)
            tick_value = chart_max * ratio
            drawing.add(Line(plot_x, y, plot_x + plot_w, y, strokeColor=colors.HexColor("#E5EAF1"), strokeWidth=0.8))
            tick_label = _fmt_int(tick_value) if chart_max > 20 else _fmt_dec(tick_value, 1).rstrip("0").rstrip(",")
            drawing.add(String(6, y - 3, tick_label, fontName="Helvetica", fontSize=6.5, fillColor=colors.HexColor("#5B6576")))

        drawing.add(Line(plot_x, plot_y, plot_x, plot_y + plot_h, strokeColor=colors.HexColor("#B8C4D3"), strokeWidth=1))
        drawing.add(Line(plot_x, plot_y, plot_x + plot_w, plot_y, strokeColor=colors.HexColor("#B8C4D3"), strokeWidth=1))

        slot_count = max(len(labels), 1)
        slot_w = plot_w / slot_count
        bar_w = min(24.0, slot_w * 0.58)
        for index, (label, value, error) in enumerate(zip(labels, values, errors, strict=False)):
            bar_x = plot_x + (slot_w * index) + ((slot_w - bar_w) / 2)
            bar_h = 0 if chart_max <= 0 else (value / chart_max) * plot_h
            bar_top = plot_y + bar_h
            drawing.add(
                Rect(
                    bar_x,
                    plot_y,
                    bar_w,
                    bar_h,
                    fillColor=colors.HexColor(bar_color),
                    strokeColor=colors.HexColor(bar_color),
                    strokeWidth=0.8,
                )
            )
            if error > 0:
                err_top = plot_y + (((value + error) / chart_max) * plot_h)
                center_x = bar_x + (bar_w / 2)
                drawing.add(Line(center_x, bar_top, center_x, err_top, strokeColor=colors.HexColor("#8793A8"), strokeWidth=1))
                drawing.add(Line(center_x - 4, err_top, center_x + 4, err_top, strokeColor=colors.HexColor("#8793A8"), strokeWidth=1))
            value_text = _fmt_int(value) if chart_max > 20 else _fmt_dec(value, 1).rstrip("0").rstrip(",")
            drawing.add(String(bar_x + (bar_w / 2), bar_top + 5, value_text, fontName="Helvetica-Bold", fontSize=6.5, fillColor=colors.HexColor("#182134"), textAnchor="middle"))
            drawing.add(
                String(
                    bar_x + (bar_w / 2),
                    plot_y - 10,
                    label,
                    fontName="Helvetica",
                    fontSize=6.2,
                    fillColor=colors.HexColor("#5B6576"),
                    textAnchor="middle",
                )
            )

        drawing.add(Rect(16, 16, 8, 8, fillColor=colors.HexColor(bar_color), strokeColor=colors.HexColor(bar_color)))
        drawing.add(String(28, 17, legend_label, fontName="Helvetica", fontSize=7.5, fillColor=colors.HexColor("#4C5668")))
        return drawing

    def build_grouped_vertical_error_chart_drawing(
        title: str,
        labels: list[str],
        series_values: list[list[float]],
        series_errors: list[list[float]],
        series_colors: list[str],
        legend_labels: list[str],
        width: float = 352,
        height: float = 220,
        y_max: float | None = None,
    ) -> Drawing:
        drawing = Drawing(width, height)
        drawing.add(Rect(0, 0, width, height, fillColor=colors.HexColor("#FBFCFE"), strokeColor=colors.HexColor("#D7DEE8"), strokeWidth=1))
        drawing.add(String(16, height - 20, title, fontName="Helvetica-Bold", fontSize=11, fillColor=colors.HexColor("#0B1020")))

        labels = [str(value) for value in labels]
        prepared_values = [[max(float(value or 0), 0.0) for value in values] for values in series_values]
        prepared_errors = [[max(float(value or 0), 0.0) for value in values] for values in series_errors]

        plot_x = 40
        plot_y = 44
        plot_w = width - 58
        plot_h = height - 84
        max_value = 0.0
        for values, errors in zip(prepared_values, prepared_errors, strict=False):
            for value, error in zip(values, errors, strict=False):
                max_value = max(max_value, value + error)
        chart_max = max(1.0, y_max if y_max is not None else (max_value * 1.18))

        for step in range(6):
            ratio = step / 5.0
            y = plot_y + (plot_h * ratio)
            tick_value = chart_max * ratio
            drawing.add(Line(plot_x, y, plot_x + plot_w, y, strokeColor=colors.HexColor("#E5EAF1"), strokeWidth=0.8))
            tick_label = _fmt_int(tick_value) if chart_max > 20 else _fmt_dec(tick_value, 1).rstrip("0").rstrip(",")
            drawing.add(String(6, y - 3, tick_label, fontName="Helvetica", fontSize=6.5, fillColor=colors.HexColor("#5B6576")))

        drawing.add(Line(plot_x, plot_y, plot_x, plot_y + plot_h, strokeColor=colors.HexColor("#B8C4D3"), strokeWidth=1))
        drawing.add(Line(plot_x, plot_y, plot_x + plot_w, plot_y, strokeColor=colors.HexColor("#B8C4D3"), strokeWidth=1))

        slot_count = max(len(labels), 1)
        series_count = max(len(prepared_values), 1)
        slot_w = plot_w / slot_count
        grouped_w = slot_w * 0.72
        bar_w = min(12.0, grouped_w / max(series_count, 1))
        group_offset = (slot_w - (bar_w * series_count)) / 2

        for index, label in enumerate(labels):
            for series_index, (values, errors, fill_hex) in enumerate(zip(prepared_values, prepared_errors, series_colors, strict=False)):
                value = values[index] if index < len(values) else 0.0
                error = errors[index] if index < len(errors) else 0.0
                bar_x = plot_x + (slot_w * index) + group_offset + (series_index * bar_w)
                bar_h = 0 if chart_max <= 0 else (value / chart_max) * plot_h
                bar_top = plot_y + bar_h
                drawing.add(
                    Rect(
                        bar_x,
                        plot_y,
                        bar_w - 1,
                        bar_h,
                        fillColor=colors.HexColor(fill_hex),
                        strokeColor=colors.HexColor(fill_hex),
                        strokeWidth=0.8,
                    )
                )
                if error > 0:
                    err_top = plot_y + (((value + error) / chart_max) * plot_h)
                    center_x = bar_x + ((bar_w - 1) / 2)
                    drawing.add(Line(center_x, bar_top, center_x, err_top, strokeColor=colors.HexColor("#8793A8"), strokeWidth=1))
                    drawing.add(Line(center_x - 3, err_top, center_x + 3, err_top, strokeColor=colors.HexColor("#8793A8"), strokeWidth=1))
                if value > 0:
                    value_text = _fmt_int(value) if chart_max > 20 else _fmt_dec(value, 1).rstrip("0").rstrip(",")
                    drawing.add(
                        String(
                            bar_x + ((bar_w - 1) / 2),
                            bar_top + 5,
                            value_text,
                            fontName="Helvetica-Bold",
                            fontSize=6.2,
                            fillColor=colors.HexColor("#182134"),
                            textAnchor="middle",
                        )
                    )
            drawing.add(
                String(
                    plot_x + (slot_w * index) + (slot_w / 2),
                    plot_y - 10,
                    label,
                    fontName="Helvetica",
                    fontSize=6.2,
                    fillColor=colors.HexColor("#5B6576"),
                    textAnchor="middle",
                )
            )

        legend_x = 16
        for fill_hex, legend_label in zip(series_colors, legend_labels, strict=False):
            drawing.add(Rect(legend_x, 16, 8, 8, fillColor=colors.HexColor(fill_hex), strokeColor=colors.HexColor(fill_hex)))
            drawing.add(String(legend_x + 12, 17, legend_label, fontName="Helvetica", fontSize=7.5, fillColor=colors.HexColor("#4C5668")))
            legend_x += 94
        return drawing

    def build_pie_chart_drawing(
        title: str,
        labels: list[str],
        values: list[float],
        width: float = 724,
        height: float = 240,
    ) -> Drawing:
        drawing = Drawing(width, height)
        drawing.add(Rect(0, 0, width, height, fillColor=colors.HexColor("#FBFCFE"), strokeColor=colors.HexColor("#D7DEE8"), strokeWidth=1))
        drawing.add(String(16, height - 20, title, fontName="Helvetica-Bold", fontSize=11, fillColor=colors.HexColor("#0B1020")))
        clean_values = [max(float(value or 0), 0.0) for value in values]
        total_value = sum(clean_values)
        if total_value <= 0:
            drawing.add(String(16, height - 46, "Geen zoneverdeling beschikbaar", fontName="Helvetica", fontSize=9, fillColor=colors.HexColor("#5B6576")))
            return drawing

        pie = Pie()
        pie.x = 24
        pie.y = 26
        pie.width = 180
        pie.height = 180
        pie.data = clean_values
        pie.labels = [""] * len(labels)
        pie.sideLabels = False
        pie.strokeColor = colors.HexColor("#FFFFFF")
        palette = ["#F5D2D8", "#F1A4B5", "#E97A93", "#D92B4D", "#A4102B", "#6E1222"]
        for index, fill_hex in enumerate(palette[: len(pie.data)]):
            pie.slices[index].fillColor = colors.HexColor(fill_hex)
            pie.slices[index].strokeColor = colors.white
        drawing.add(pie)

        legend_x = 246
        legend_top = height - 42
        drawing.add(String(legend_x, legend_top, "Zone", fontName="Helvetica-Bold", fontSize=8, fillColor=colors.HexColor("#4C5668")))
        drawing.add(String(legend_x + 180, legend_top, "Distance", fontName="Helvetica-Bold", fontSize=8, fillColor=colors.HexColor("#4C5668")))
        drawing.add(String(width - 58, legend_top, "Share", fontName="Helvetica-Bold", fontSize=8, fillColor=colors.HexColor("#4C5668"), textAnchor="end"))

        row_y = legend_top - 18
        for index, (label, value) in enumerate(zip(labels, clean_values, strict=False)):
            fill_hex = palette[index % len(palette)]
            share_pct = (value / total_value) * 100 if total_value > 0 else 0.0
            drawing.add(Rect(legend_x, row_y - 5, 8, 8, fillColor=colors.HexColor(fill_hex), strokeColor=colors.HexColor(fill_hex)))
            drawing.add(String(legend_x + 14, row_y, str(label), fontName="Helvetica", fontSize=8.2, fillColor=colors.HexColor("#182134")))
            drawing.add(String(legend_x + 180, row_y, _fmt_distance(value), fontName="Helvetica", fontSize=8.2, fillColor=colors.HexColor("#182134")))
            drawing.add(
                String(
                    width - 58,
                    row_y,
                    f"{_fmt_dec(share_pct, 1)}%",
                    fontName="Helvetica-Bold",
                    fontSize=8.2,
                    fillColor=colors.HexColor("#182134"),
                    textAnchor="end",
                )
            )
            row_y -= 18
        return drawing

    def build_horizontal_bar_chart_drawing(
        title: str,
        labels: list[str],
        values: list[float],
        bar_color: str,
        width: float = 352,
        height: float = 240,
    ) -> Drawing:
        drawing = Drawing(width, height)
        drawing.add(Rect(0, 0, width, height, fillColor=colors.HexColor("#FBFCFE"), strokeColor=colors.HexColor("#D7DEE8"), strokeWidth=1))
        drawing.add(String(16, height - 20, title, fontName="Helvetica-Bold", fontSize=11, fillColor=colors.HexColor("#0B1020")))

        chart = HorizontalBarChart()
        chart.x = 88
        chart.y = 30
        chart.height = height - 64
        chart.width = width - 116
        chart.data = [_series_to_floats(values)]
        chart.strokeColor = colors.HexColor("#94A3B8")
        chart.valueAxis.valueMin = 0
        peak = max(_series_to_floats(values), default=0.0)
        chart.valueAxis.valueMax = max(1.0, peak * 1.18)
        chart.valueAxis.strokeColor = colors.HexColor("#B8C4D3")
        chart.valueAxis.gridStrokeColor = colors.HexColor("#E5EAF1")
        chart.valueAxis.gridStrokeDashArray = [2, 2]
        chart.valueAxis.visibleGrid = True
        chart.valueAxis.labels.fillColor = colors.HexColor("#5B6576")
        chart.valueAxis.labels.fontName = "Helvetica"
        chart.valueAxis.labels.fontSize = 7
        chart.categoryAxis.categoryNames = [str(value) for value in labels]
        chart.categoryAxis.strokeColor = colors.HexColor("#B8C4D3")
        chart.categoryAxis.labels.fillColor = colors.HexColor("#5B6576")
        chart.categoryAxis.labels.fontName = "Helvetica"
        chart.categoryAxis.labels.fontSize = 6.5
        chart.categoryAxis.labels.boxAnchor = "e"
        chart.bars[0].fillColor = colors.HexColor(bar_color)
        chart.bars[0].strokeColor = colors.HexColor(bar_color)
        chart.barSpacing = 3
        drawing.add(chart)

        drawing.add(Rect(16, 12, 8, 8, fillColor=colors.HexColor(bar_color), strokeColor=colors.HexColor(bar_color)))
        return drawing

    story: list[object] = []

    hero_table = Table(
        [
            [Paragraph("Week Report", title_style)],
            [Paragraph(f"MVV Maastricht | {week_label} | {iso_label}", kicker_style)],
            [
                Paragraph(
                    "Compacte weekrapportage met teamload, squad spread, wellness, RPE en staffnotities voor dezelfde selectie als in het dashboard.",
                    hero_body_style,
                )
            ],
        ],
        colWidths=[doc.width],
    )
    hero_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FBFCFE")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#D7DEE8")),
                ("LINEABOVE", (0, 0), (-1, 0), 2, colors.HexColor("#C8102E")),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.append(hero_table)
    story.append(Spacer(1, 10))

    cards = [
        build_metric_card("Active Players", _fmt_int(summary.get("active_players")), "Unieke GPS-spelers in deze week", "#FBFCFE", "#D7DEE8"),
        build_metric_card("Player Sessions", _fmt_int(summary.get("player_sessions")), "Totaal aantal Summary-sessies", "#FBFCFE", "#D7DEE8"),
        build_metric_card("Total Distance", _fmt_distance(summary.get("total_distance")), "Opgetelde teamload in de week", "#FBFCFE", "#D7DEE8"),
        build_metric_card("HSR / HSD", _fmt_distance(summary.get("hsr_hsd")), "Sprint plus high sprint distance", "#FBFCFE", "#D7DEE8"),
        build_metric_card("Sprints", _fmt_int(summary.get("sprints")), "Totale sprintacties in deze week", "#FFF7F8", "#E8C5CB"),
        build_metric_card("Speed Exposures", _fmt_int(summary.get("speed_exposures")), "Sessies >= 90% van seizoenstop", "#FFF7F8", "#E8C5CB"),
        build_metric_card("Dist / Player", _fmt_distance(summary.get("dist_per_player")), "Teamload gedeeld door actieve spelers", "#F8FAFC", "#D7DEE8"),
        build_metric_card("Top Speed", _fmt_speed(summary.get("top_speed")), "Hoogste gemeten snelheid", "#F8FAFC", "#D7DEE8"),
        build_metric_card("Readiness", _fmt_dec(monitoring_summary.get("readiness_avg"), 1), "Gemiddelde readiness-score", "#FFF7F8", "#E8C5CB"),
        build_metric_card("Avg RPE", _fmt_dec(monitoring_summary.get("avg_rpe"), 1), "Gemiddelde RPE in deze week", "#FFF7F8", "#E8C5CB"),
        build_metric_card("Wellness Entries", _fmt_int(monitoring_summary.get("wellness_entries")), "Aantal wellnessregistraties", "#FFF7F8", "#E8C5CB"),
        build_metric_card("RPE Entries", _fmt_int(monitoring_summary.get("rpe_entries")), "Aantal RPE-registraties", "#FFF7F8", "#E8C5CB"),
    ]
    story.append(
        KeepTogether(
            [
                Paragraph("Visual Snapshot", section_style),
                build_card_grid(cards),
                Spacer(1, 8),
            ]
        )
    )

    if isinstance(zone_df, pd.DataFrame) and not zone_df.empty:
        story.append(Paragraph("Distance Zone Profile", section_style))
        story.append(
            build_pie_chart_drawing(
                "Distance Zone Share",
                [str(value) for value in zone_df["zone"].fillna("--").tolist()],
                _series_to_floats(zone_df["value"]),
                width=doc.width,
                height=238,
            )
        )
        story.append(Spacer(1, 8))

    if isinstance(day_stats, pd.DataFrame) and not day_stats.empty:
        spread_row_one = Table(
            [[
                build_vertical_error_chart_drawing(
                    "Player Avg Distance +/- SD",
                    [str(value) for value in day_stats["label"].fillna("--").tolist()],
                    _series_to_floats(day_stats["total_distance_mean"]),
                    _series_to_floats(day_stats["total_distance_std"]),
                    "#6E1222",
                    "Total Distance",
                ),
                build_vertical_error_chart_drawing(
                    "Player Avg HSR / HSD +/- SD",
                    [str(value) for value in day_stats["label"].fillna("--").tolist()],
                    _series_to_floats(day_stats["hsr_hsd_mean"]),
                    _series_to_floats(day_stats["hsr_hsd_std"]),
                    "#EA3351",
                    "HSR / HSD",
                ),
            ]],
            colWidths=[doc.width / 2.0, doc.width / 2.0],
            hAlign="LEFT",
        )
        spread_row_two = Table(
            [[
                build_grouped_vertical_error_chart_drawing(
                    "Player Avg Accel / Decel +/- SD",
                    [str(value) for value in day_stats["label"].fillna("--").tolist()],
                    [
                        _series_to_floats(day_stats["total_accelerations_mean"]),
                        _series_to_floats(day_stats["total_decelerations_mean"]),
                    ],
                    [
                        _series_to_floats(day_stats["total_accelerations_std"]),
                        _series_to_floats(day_stats["total_decelerations_std"]),
                    ],
                    ["#EA3351", "#6E1222"],
                    ["Accelerations", "Decelerations"],
                ),
                build_vertical_error_chart_drawing(
                    "Player Avg Sprints +/- SD",
                    [str(value) for value in day_stats["label"].fillna("--").tolist()],
                    _series_to_floats(day_stats["sprints_mean"]),
                    _series_to_floats(day_stats["sprints_std"]),
                    "#6E1222",
                    "Sprints",
                ),
            ]],
            colWidths=[doc.width / 2.0, doc.width / 2.0],
            hAlign="LEFT",
        )
        for grid in (spread_row_one, spread_row_two):
            grid.setStyle(
                TableStyle(
                    [
                        ("LEFTPADDING", (0, 0), (-1, -1), 0),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                        ("TOPPADDING", (0, 0), (-1, -1), 0),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )
        story.append(Paragraph("Squad Spread Charts", section_style))
        story.append(spread_row_one)
        story.append(Spacer(1, 8))
        story.append(spread_row_two)
        story.append(Spacer(1, 8))

    if isinstance(monitoring_day_table, pd.DataFrame) and not monitoring_day_table.empty:
        monitoring_labels = [str(value) for value in monitoring_day_table["label"].fillna("--").tolist()]
        physical_draw = build_grouped_vertical_error_chart_drawing(
            "Physical Wellness +/- SD",
            monitoring_labels,
            [
                _series_to_floats(monitoring_day_table["muscle_soreness"]),
                _series_to_floats(monitoring_day_table["fatigue"]),
            ],
            [
                _series_to_floats(monitoring_day_table.get("muscle_soreness_std")),
                _series_to_floats(monitoring_day_table.get("fatigue_std")),
            ],
            ["#6E1222", "#EA3351"],
            ["Muscle Soreness", "Fatigue"],
            width=352,
            height=220,
            y_max=10,
        )
        mental_draw = build_grouped_vertical_error_chart_drawing(
            "Mental Wellness +/- SD",
            monitoring_labels,
            [
                _series_to_floats(monitoring_day_table["sleep_quality"]),
                _series_to_floats(monitoring_day_table["stress"]),
                _series_to_floats(monitoring_day_table["mood"]),
            ],
            [
                _series_to_floats(monitoring_day_table.get("sleep_quality_std")),
                _series_to_floats(monitoring_day_table.get("stress_std")),
                _series_to_floats(monitoring_day_table.get("mood_std")),
            ],
            ["#6E1222", "#EA3351", "#F59E0B"],
            ["Sleep Quality", "Stress", "Mood"],
            width=352,
            height=220,
            y_max=10,
        )
        rpe_draw: Drawing
        if isinstance(rpe_session_day_table, pd.DataFrame) and not rpe_session_day_table.empty:
            rpe_tmp = rpe_session_day_table.copy()
            rpe_tmp["entry_date"] = pd.to_datetime(rpe_tmp["entry_date"], errors="coerce")
            rpe_tmp["session_index"] = pd.to_numeric(rpe_tmp["session_index"], errors="coerce").fillna(1).astype(int)
            rpe_tmp = rpe_tmp.dropna(subset=["entry_date"]).sort_values(["entry_date", "session_index"]).reset_index(drop=True)

            rpe_day_labels = (
                rpe_tmp.drop_duplicates(subset=["entry_date"])
                .sort_values("entry_date")["label"]
                .fillna("--")
                .astype(str)
                .tolist()
            )
            rpe_days = (
                rpe_tmp.drop_duplicates(subset=["entry_date"])
                .sort_values("entry_date")["entry_date"]
                .tolist()
            )
            session_indexes = sorted(rpe_tmp["session_index"].dropna().astype(int).unique().tolist())
            series_values: list[list[float]] = []
            series_errors: list[list[float]] = []
            series_colors: list[str] = []
            legend_labels: list[str] = []
            palette = ["#EA3351", "#6E1222", "#F59E0B", "#F5D2D8"]

            for palette_index, session_index in enumerate(session_indexes):
                session_values: list[float] = []
                session_errors: list[float] = []
                for day_value in rpe_days:
                    row = rpe_tmp[
                        (rpe_tmp["entry_date"] == day_value) & (rpe_tmp["session_index"] == session_index)
                    ]
                    if row.empty:
                        session_values.append(0.0)
                        session_errors.append(0.0)
                    else:
                        avg_value = pd.to_numeric(row.iloc[0].get("avg_rpe"), errors="coerce")
                        std_value = pd.to_numeric(row.iloc[0].get("avg_rpe_std"), errors="coerce")
                        session_values.append(float(avg_value) if pd.notna(avg_value) else 0.0)
                        session_errors.append(float(std_value) if pd.notna(std_value) else 0.0)
                series_values.append(session_values)
                series_errors.append(session_errors)
                series_colors.append(palette[palette_index % len(palette)])
                legend_labels.append(f"Sessie {session_index}")

            rpe_draw = build_grouped_vertical_error_chart_drawing(
                "Daily Avg RPE per Session +/- SD",
                rpe_day_labels,
                series_values,
                series_errors,
                series_colors,
                legend_labels,
                width=doc.width,
                height=220,
                y_max=10,
            )
        else:
            rpe_draw = build_vertical_error_chart_drawing(
                "Daily Avg RPE +/- SD",
                monitoring_labels,
                _series_to_floats(monitoring_day_table["avg_rpe"]),
                _series_to_floats(monitoring_day_table.get("avg_rpe_std")),
                "#EA3351",
                "Avg RPE",
                width=doc.width,
                height=220,
                y_max=10,
            )
        story.append(Paragraph("Monitoring Charts", section_style))
        monitoring_row = Table(
            [[physical_draw, mental_draw]],
            colWidths=[doc.width / 2.0, doc.width / 2.0],
            hAlign="LEFT",
        )
        monitoring_row.setStyle(
            TableStyle(
                [
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(monitoring_row)
        story.append(Spacer(1, 8))
        story.append(rpe_draw)
        story.append(Spacer(1, 8))

    if isinstance(player_table, pd.DataFrame) and not player_table.empty:
        top_distance = player_table.nlargest(10, "total_distance").sort_values("total_distance", ascending=True)
        top_hsr = player_table.nlargest(10, "hsr_hsd").sort_values("hsr_hsd", ascending=True)
        top_sprints = player_table.nlargest(10, "sprints").sort_values("sprints", ascending=True)
        leader_row = Table(
            [[
                build_horizontal_bar_chart_drawing(
                    "Top 10 Total Distance",
                    [str(value) for value in top_distance["player_name"].fillna("--").tolist()],
                    _series_to_floats(top_distance["total_distance"]),
                    "#6E1222",
                ),
                build_horizontal_bar_chart_drawing(
                    "Top 10 HSR / HSD",
                    [str(value) for value in top_hsr["player_name"].fillna("--").tolist()],
                    _series_to_floats(top_hsr["hsr_hsd"]),
                    "#EA3351",
                ),
            ]],
            colWidths=[doc.width / 2.0, doc.width / 2.0],
            hAlign="LEFT",
        )
        leader_row.setStyle(
            TableStyle(
                [
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(Paragraph("Leaderboards", section_style))
        story.append(leader_row)
        story.append(Spacer(1, 8))
        story.append(
            build_horizontal_bar_chart_drawing(
                "Top 10 Sprints",
                [str(value) for value in top_sprints["player_name"].fillna("--").tolist()],
                _series_to_floats(top_sprints["sprints"]),
                "#6E1222",
                width=352,
                height=240,
            )
        )
        story.append(Spacer(1, 8))

    summary_rows = [
        ["Dag", "Players", "Sessions", "Distance", "HSR/HSD", "Sprints", "Dist / Player"],
    ]
    if isinstance(day_table, pd.DataFrame) and not day_table.empty:
        for _, row in day_table.iterrows():
            summary_rows.append(
                [
                    str(row.get("label") or "--"),
                    _fmt_int(row.get("active_players")),
                    _fmt_int(row.get("player_sessions")),
                    _fmt_distance(row.get("total_distance")),
                    _fmt_distance(row.get("hsr_hsd")),
                    _fmt_int(row.get("sprints")),
                    _fmt_distance(row.get("distance_per_player")),
                ]
            )
        story.append(
            KeepTogether(
                [
                    Paragraph("Weekdays", section_style),
                    build_standard_table(
                        summary_rows,
                        [28 * mm, 22 * mm, 22 * mm, 30 * mm, 28 * mm, 18 * mm, 30 * mm],
                        "#0B1020",
                        "#F8FAFC",
                        "#EEF2F7",
                    ),
                    Spacer(1, 8),
                ]
            )
        )

    type_rows = [["Type", "Players", "Sessions", "Distance", "HSR/HSD", "Sprints", "Top Speed"]]
    if isinstance(type_table, pd.DataFrame) and not type_table.empty:
        for _, row in type_table.iterrows():
            type_rows.append(
                [
                    str(row.get("session_category") or "--"),
                    _fmt_int(row.get("active_players")),
                    _fmt_int(row.get("player_sessions")),
                    _fmt_distance(row.get("total_distance")),
                    _fmt_distance(row.get("hsr_hsd")),
                    _fmt_int(row.get("sprints")),
                    _fmt_speed(row.get("max_speed")),
                ]
            )
        story.append(
            KeepTogether(
                [
                    Paragraph("Training vs Match", section_style),
                    build_standard_table(
                        type_rows,
                        [34 * mm, 22 * mm, 24 * mm, 32 * mm, 28 * mm, 20 * mm, 24 * mm],
                        "#C8102E",
                        "#FFF7F8",
                        "#FCEBED",
                    ),
                    Spacer(1, 8),
                ]
            )
        )

    player_rows = [["Speler", "Sessies", "Distance", "HSR/HSD", "Sprints", "Accel", "Decel", "Top Speed"]]
    player_preview = player_table.head(12).copy() if isinstance(player_table, pd.DataFrame) else pd.DataFrame()
    if not player_preview.empty:
        for _, row in player_preview.iterrows():
            player_rows.append(
                [
                    str(row.get("player_name") or "--"),
                    _fmt_int(row.get("sessions")),
                    _fmt_distance(row.get("total_distance")),
                    _fmt_distance(row.get("hsr_hsd")),
                    _fmt_int(row.get("sprints")),
                    _fmt_int(row.get("total_accelerations")),
                    _fmt_int(row.get("total_decelerations")),
                    _fmt_speed(row.get("max_speed")),
                ]
            )
        story.extend(
            [
                Paragraph("Player Summary", section_style),
                build_standard_table(
                    player_rows,
                    [48 * mm, 18 * mm, 28 * mm, 26 * mm, 16 * mm, 18 * mm, 18 * mm, 22 * mm],
                    "#0B1020",
                    "#FFFFFF",
                    "#F5F7FB",
                ),
                Spacer(1, 8),
            ]
        )

    monitoring_rows = [["Dag", "Muscle", "Fatigue", "Sleep", "Stress", "Mood", "Readiness", "Avg RPE"]]
    monitoring_preview = monitoring_day_table.copy() if isinstance(monitoring_day_table, pd.DataFrame) else pd.DataFrame()
    if not monitoring_preview.empty:
        for _, row in monitoring_preview.iterrows():
            monitoring_rows.append(
                [
                    str(row.get("label") or "--"),
                    _fmt_dec(row.get("muscle_soreness"), 1),
                    _fmt_dec(row.get("fatigue"), 1),
                    _fmt_dec(row.get("sleep_quality"), 1),
                    _fmt_dec(row.get("stress"), 1),
                    _fmt_dec(row.get("mood"), 1),
                    _fmt_dec(row.get("readiness_score"), 1),
                    _fmt_dec(row.get("avg_rpe"), 1),
                ]
            )
        story.extend(
            [
                Paragraph("Wellness and RPE by Day", section_style),
                build_standard_table(
                    monitoring_rows,
                    [24 * mm, 18 * mm, 18 * mm, 18 * mm, 18 * mm, 18 * mm, 22 * mm, 20 * mm],
                    "#C8102E",
                    "#FFF7F8",
                    "#FCEBED",
                ),
                Spacer(1, 8),
            ]
        )

    note_items = list(notes)
    if note_items:
        story.append(Paragraph("Week Notes", section_style))
        for note in note_items:
            story.append(Paragraph(f"- {note}", note_style))
        story.append(Spacer(1, 8))

    story.append(
        Paragraph(
            "PDF-export gebruikt dezelfde weekselectie en monitoringrange als de dashboardweergave.",
            body_style,
        )
    )

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
