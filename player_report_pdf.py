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


def _fmt_minutes(value: object) -> str:
    base = _fmt_int(value)
    return "--" if base == "--" else f"{base} min"


def build_player_report_pdf_bytes(
    *,
    player_name: str,
    scope_label: str,
    period_label: str,
    summary: dict[str, object],
    monitoring_summary: dict[str, object],
    sessions_df: pd.DataFrame,
    monitoring_group_df: pd.DataFrame,
    notes: Iterable[str],
) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.charts.linecharts import HorizontalLineChart
    from reportlab.graphics.shapes import Drawing, Rect, String
    from reportlab.graphics.widgets.markers import makeMarker
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

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
        "mvv_title",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=24,
        textColor=colors.white,
        spaceAfter=2,
    )
    kicker_style = ParagraphStyle(
        "mvv_kicker",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        textColor=colors.HexColor("#F6B8C4"),
        leading=11,
        spaceAfter=3,
    )
    hero_body_style = ParagraphStyle(
        "mvv_hero_body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.2,
        leading=12,
        textColor=colors.HexColor("#E7ECF4"),
        alignment=TA_LEFT,
    )
    section_style = ParagraphStyle(
        "mvv_section",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        textColor=colors.HexColor("#0B1020"),
        spaceAfter=6,
        spaceBefore=8,
    )
    body_style = ParagraphStyle(
        "mvv_body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#182134"),
    )
    card_label_style = ParagraphStyle(
        "mvv_card_label",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=7.2,
        leading=9,
        textColor=colors.HexColor("#A9B7CC"),
        alignment=TA_LEFT,
    )
    card_value_style = ParagraphStyle(
        "mvv_card_value",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=15.6,
        leading=17,
        textColor=colors.white,
        alignment=TA_LEFT,
    )
    card_foot_style = ParagraphStyle(
        "mvv_card_foot",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=7.2,
        leading=9,
        textColor=colors.HexColor("#D7DFEB"),
        alignment=TA_LEFT,
    )
    note_style = ParagraphStyle(
        "mvv_note",
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

    def build_session_chart_labels(chart_df: pd.DataFrame) -> list[str]:
        counts: dict[str, int] = {}
        labels: list[str] = []
        for _, row in chart_df.iterrows():
            base = str(row.get("datum_label") or "--")[:5]
            counts[base] = counts.get(base, 0) + 1
            session_suffix = counts[base]
            type_value = str(row.get("type") or "").strip().lower()
            type_suffix = "M" if "match" in type_value or "wedstrijd" in type_value else "T"
            duplicate_suffix = f"-{session_suffix}" if session_suffix > 1 else ""
            labels.append(f"{base}{duplicate_suffix}-{type_suffix}")
        return labels

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
        drawing.add(Rect(0, 0, width, height, fillColor=colors.HexColor("#0B1020"), strokeColor=colors.HexColor("#1E2A3E"), strokeWidth=1))
        drawing.add(String(16, height - 20, title, fontName="Helvetica-Bold", fontSize=11, fillColor=colors.white))

        chart = VerticalBarChart()
        chart.x = 32
        chart.y = 44
        chart.height = height - 82
        chart.width = width - 56
        chart.data = data_series
        chart.strokeColor = colors.HexColor("#A9B7CC")
        chart.valueAxis.valueMin = 0
        peak = max((max(series) if series else 0) for series in data_series)
        chart.valueAxis.valueMax = max(1, peak * 1.18)
        chart.valueAxis.strokeColor = colors.HexColor("#4A5870")
        chart.valueAxis.gridStrokeColor = colors.HexColor("#283347")
        chart.valueAxis.gridStrokeDashArray = [2, 2]
        chart.valueAxis.visibleGrid = True
        chart.valueAxis.labels.fillColor = colors.HexColor("#D7DFEB")
        chart.valueAxis.labels.fontName = "Helvetica"
        chart.valueAxis.labels.fontSize = 7
        chart.categoryAxis.categoryNames = labels
        chart.categoryAxis.strokeColor = colors.HexColor("#4A5870")
        chart.categoryAxis.labels.boxAnchor = "ne"
        chart.categoryAxis.labels.angle = 30
        chart.categoryAxis.labels.dx = -4
        chart.categoryAxis.labels.dy = -2
        chart.categoryAxis.labels.fillColor = colors.HexColor("#D7DFEB")
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
            drawing.add(String(legend_x + 12, legend_y + 1, legend_label, fontName="Helvetica", fontSize=7.5, fillColor=colors.HexColor("#D7DFEB")))
            legend_x += 92
        return drawing

    def build_line_chart_drawing(
        title: str,
        labels: list[str],
        data_series: list[list[float]],
        series_colors: list[str],
        legend_labels: list[str],
        width: float = 724,
        height: float = 230,
    ) -> Drawing:
        drawing = Drawing(width, height)
        drawing.add(Rect(0, 0, width, height, fillColor=colors.HexColor("#0B1020"), strokeColor=colors.HexColor("#1E2A3E"), strokeWidth=1))
        drawing.add(String(16, height - 20, title, fontName="Helvetica-Bold", fontSize=11, fillColor=colors.white))

        chart = HorizontalLineChart()
        chart.x = 32
        chart.y = 42
        chart.height = height - 80
        chart.width = width - 56
        chart.data = data_series
        chart.joinedLines = 1
        chart.valueAxis.valueMin = 0
        chart.valueAxis.valueMax = 10
        chart.valueAxis.strokeColor = colors.HexColor("#4A5870")
        chart.valueAxis.gridStrokeColor = colors.HexColor("#283347")
        chart.valueAxis.gridStrokeDashArray = [2, 2]
        chart.valueAxis.visibleGrid = True
        chart.valueAxis.labels.fillColor = colors.HexColor("#D7DFEB")
        chart.valueAxis.labels.fontName = "Helvetica"
        chart.valueAxis.labels.fontSize = 7
        chart.categoryAxis.categoryNames = labels
        chart.categoryAxis.strokeColor = colors.HexColor("#4A5870")
        chart.categoryAxis.labels.fillColor = colors.HexColor("#D7DFEB")
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
            drawing.add(String(legend_x + 12, legend_y + 1, legend_label, fontName="Helvetica", fontSize=7.5, fillColor=colors.HexColor("#D7DFEB")))
            legend_x += 96
        return drawing

    story: list[object] = []

    hero_table = Table(
        [
            [Paragraph("Player Report", title_style)],
            [Paragraph(f"MVV Maastricht | {player_name} | {scope_label} | {period_label}", kicker_style)],
            [
                Paragraph(
                    "Individuele rapportage met GPS-load, accelerations, decelerations, wellness en RPE voor dezelfde geselecteerde scope als in het dashboard.",
                    hero_body_style,
                )
            ],
        ],
        colWidths=[doc.width],
    )
    hero_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#0B1020")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#1E2A3E")),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.append(hero_table)
    story.append(Spacer(1, 10))

    visual_cards = [
        build_metric_card("Sessies", _fmt_int(summary.get("sessions")), "Summary-sessies in deze selectie", "#121A2B", "#30415E"),
        build_metric_card("Actieve dagen", _fmt_int(summary.get("active_days")), "Unieke trainings- of wedstrijddagen", "#121A2B", "#30415E"),
        build_metric_card("Total Distance", _fmt_distance(summary.get("total_distance")), "Totale loopbelasting binnen scope", "#121A2B", "#30415E"),
        build_metric_card("HSR / HSD", _fmt_distance(summary.get("hsr_hsd")), "Sprint plus high sprint distance", "#121A2B", "#30415E"),
        build_metric_card("Sprints", _fmt_int(summary.get("sprints")), "Totaal aantal sprintacties", "#171223", "#5B2740"),
        build_metric_card("Accelerations", _fmt_int(summary.get("total_accelerations")), "Totale acceleraties in scope", "#171223", "#5B2740"),
        build_metric_card("Decelerations", _fmt_int(summary.get("total_decelerations")), "Totale deceleraties in scope", "#171223", "#5B2740"),
        build_metric_card("Top Speed", _fmt_speed(summary.get("top_speed")), "Hoogste gemeten snelheid", "#171223", "#5B2740"),
        build_metric_card("Duur", _fmt_minutes(summary.get("duration_min")), "Totale sessieduur", "#0F1726", "#2B3952"),
        build_metric_card(
            "Avg Intensity",
            _fmt_dec(summary.get("distance_per_min"), 1) + " m/min" if not pd.isna(summary.get("distance_per_min")) else "--",
            "Gemiddelde meters per minuut",
            "#0F1726",
            "#2B3952",
        ),
        build_metric_card("Trainingen", _fmt_int(summary.get("training_sessions")), "Aantal trainingssessies", "#0F1726", "#2B3952"),
        build_metric_card("Wedstrijden", _fmt_int(summary.get("match_sessions")), "Aantal matchsituaties", "#0F1726", "#2B3952"),
        build_metric_card("Readiness", _fmt_dec(monitoring_summary.get("readiness_avg"), 1), "Gemiddelde readiness-score", "#111826", "#2F3D56"),
        build_metric_card("Avg RPE", _fmt_dec(monitoring_summary.get("avg_rpe"), 1), "Gewogen gemiddelde RPE", "#111826", "#2F3D56"),
        build_metric_card("RPE Load", _fmt_int(monitoring_summary.get("rpe_load")), "Totale duration x RPE", "#111826", "#2F3D56"),
        build_metric_card("Wellness entries", _fmt_int(monitoring_summary.get("wellness_entries")), "Aantal wellnessregistraties", "#111826", "#2F3D56"),
    ]
    story.append(Paragraph("Visual Snapshot", section_style))
    story.append(build_card_grid(visual_cards))
    story.append(Spacer(1, 8))

    chart_sessions_df = sessions_df.head(8).copy() if isinstance(sessions_df, pd.DataFrame) else pd.DataFrame()
    if not chart_sessions_df.empty:
        chart_sessions_df = chart_sessions_df.iloc[::-1].reset_index(drop=True)
        session_labels = build_session_chart_labels(chart_sessions_df)
        session_distance_chart = build_bar_chart_drawing(
            "Session Distance per sessie",
            session_labels,
            [pd.to_numeric(chart_sessions_df["total_distance"], errors="coerce").fillna(0).tolist()],
            ["#6E1222"],
            ["Total Distance"],
        )
        session_accel_chart = build_bar_chart_drawing(
            "Accelerations vs Decelerations per sessie",
            session_labels,
            [
                pd.to_numeric(chart_sessions_df["total_accelerations"], errors="coerce").fillna(0).tolist(),
                pd.to_numeric(chart_sessions_df["total_decelerations"], errors="coerce").fillna(0).tolist(),
            ],
            ["#EA3351", "#F5D2D8"],
            ["Accelerations", "Decelerations"],
        )
        chart_grid = Table(
            [[session_distance_chart, session_accel_chart]],
            colWidths=[doc.width / 2.0, doc.width / 2.0],
            hAlign="LEFT",
        )
        chart_grid.setStyle(
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
        story.append(Paragraph("Session Charts", section_style))
        story.append(chart_grid)
        story.append(Spacer(1, 8))

    chart_monitoring_df = monitoring_group_df.tail(8).copy() if isinstance(monitoring_group_df, pd.DataFrame) else pd.DataFrame()
    if not chart_monitoring_df.empty and {"label", "readiness_score", "avg_rpe"}.issubset(chart_monitoring_df.columns):
        monitoring_labels = [str(value) for value in chart_monitoring_df["label"].fillna("--").tolist()]
        monitoring_chart = build_line_chart_drawing(
            "Readiness en RPE trend",
            monitoring_labels,
            [
                pd.to_numeric(chart_monitoring_df["readiness_score"], errors="coerce").fillna(0).tolist(),
                pd.to_numeric(chart_monitoring_df["avg_rpe"], errors="coerce").fillna(0).tolist(),
            ],
            ["#EA3351", "#F5D2D8"],
            ["Readiness", "Avg RPE"],
        )
        story.append(Paragraph("Monitoring Chart", section_style))
        story.append(monitoring_chart)
        story.append(Spacer(1, 8))

    summary_rows = [
        ["Metriek", "Waarde", "Metriek", "Waarde"],
        ["Sessies", _fmt_int(summary.get("sessions")), "Actieve dagen", _fmt_int(summary.get("active_days"))],
        ["Total Distance", _fmt_distance(summary.get("total_distance")), "HSR / HSD", _fmt_distance(summary.get("hsr_hsd"))],
        ["Sprints", _fmt_int(summary.get("sprints")), "Duur", _fmt_minutes(summary.get("duration_min"))],
        ["Accelerations", _fmt_int(summary.get("total_accelerations")), "Decelerations", _fmt_int(summary.get("total_decelerations"))],
        [
            "Avg Intensity",
            _fmt_dec(summary.get("distance_per_min"), 1) + " m/min" if not pd.isna(summary.get("distance_per_min")) else "--",
            "Top Speed",
            _fmt_speed(summary.get("top_speed")),
        ],
        ["Trainingen", _fmt_int(summary.get("training_sessions")), "Wedstrijden", _fmt_int(summary.get("match_sessions"))],
    ]
    summary_table = Table(summary_rows, colWidths=[42 * mm, 34 * mm, 42 * mm, 34 * mm])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B1020")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#182134")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F8FAFC")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F8FAFC"), colors.HexColor("#EEF2F7")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D7DEE8")),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("LEADING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(Paragraph("Detailed Summary", section_style))
    story.append(summary_table)
    story.append(Spacer(1, 8))

    monitoring_rows = [
        ["Monitoring", "Waarde", "Monitoring", "Waarde"],
        ["Muscle Soreness", _fmt_dec(monitoring_summary.get("muscle_soreness"), 1), "Fatigue", _fmt_dec(monitoring_summary.get("fatigue"), 1)],
        ["Sleep Quality", _fmt_dec(monitoring_summary.get("sleep_quality"), 1), "Stress", _fmt_dec(monitoring_summary.get("stress"), 1)],
        ["Mood", _fmt_dec(monitoring_summary.get("mood"), 1), "Readiness", _fmt_dec(monitoring_summary.get("readiness_avg"), 1)],
        ["Avg RPE", _fmt_dec(monitoring_summary.get("avg_rpe"), 1), "RPE Load", _fmt_int(monitoring_summary.get("rpe_load"))],
        ["Wellness entries", _fmt_int(monitoring_summary.get("wellness_entries")), "RPE entries", _fmt_int(monitoring_summary.get("rpe_entries"))],
    ]
    monitoring_table = Table(monitoring_rows, colWidths=[42 * mm, 34 * mm, 42 * mm, 34 * mm])
    monitoring_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#C8102E")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#182134")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#FFF7F8")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#FFF7F8"), colors.HexColor("#FCEBED")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E8C5CB")),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(Paragraph("Wellness &amp; RPE", section_style))
    story.append(monitoring_table)
    story.append(Spacer(1, 8))

    sessions_preview = sessions_df.head(12).copy() if isinstance(sessions_df, pd.DataFrame) else pd.DataFrame()
    if not sessions_preview.empty:
        session_rows = [["Datum", "Type", "Event", "Distance", "HSR/HSD", "Sprints", "Accel", "Decel", "Max Speed"]]
        for _, row in sessions_preview.iterrows():
            session_rows.append(
                [
                    str(row.get("datum_label") or "--"),
                    str(row.get("type") or "--"),
                    str(row.get("event") or "--"),
                    _fmt_distance(row.get("total_distance")),
                    _fmt_distance(row.get("hsr_hsd")),
                    _fmt_int(row.get("number_of_sprints")),
                    _fmt_int(row.get("total_accelerations")),
                    _fmt_int(row.get("total_decelerations")),
                    _fmt_speed(row.get("max_speed")),
                ]
            )
        session_table = Table(
            session_rows,
            colWidths=[22 * mm, 24 * mm, 38 * mm, 24 * mm, 24 * mm, 15 * mm, 16 * mm, 16 * mm, 21 * mm],
            repeatRows=1,
        )
        session_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B1020")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7FB")]),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D7DEE8")),
                    ("FONTSIZE", (0, 0), (-1, -1), 7.4),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(Paragraph("Recent Sessions", section_style))
        story.append(session_table)
        story.append(Spacer(1, 8))

    monitoring_preview = monitoring_group_df.head(12).copy() if isinstance(monitoring_group_df, pd.DataFrame) else pd.DataFrame()
    if not monitoring_preview.empty:
        grouped_rows = [["Periode", "Muscle", "Fatigue", "Sleep", "Stress", "Mood", "Avg RPE", "RPE Load"]]
        for _, row in monitoring_preview.iterrows():
            grouped_rows.append(
                [
                    str(row.get("label") or row.get("week_label") or "--"),
                    _fmt_dec(row.get("muscle_soreness"), 1),
                    _fmt_dec(row.get("fatigue"), 1),
                    _fmt_dec(row.get("sleep_quality"), 1),
                    _fmt_dec(row.get("stress"), 1),
                    _fmt_dec(row.get("mood"), 1),
                    _fmt_dec(row.get("avg_rpe"), 1),
                    _fmt_int(row.get("rpe_load")),
                ]
            )
        grouped_table = Table(
            grouped_rows,
            colWidths=[26 * mm, 18 * mm, 18 * mm, 18 * mm, 18 * mm, 18 * mm, 18 * mm, 22 * mm],
            repeatRows=1,
        )
        grouped_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#C8102E")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#FFF7F8"), colors.HexColor("#FCEBED")]),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E8C5CB")),
                    ("FONTSIZE", (0, 0), (-1, -1), 7.3),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(Paragraph("Monitoring Timeline", section_style))
        story.append(grouped_table)
        story.append(Spacer(1, 8))

    story.append(Paragraph("Notes", section_style))
    note_list = list(notes)
    if not note_list:
        note_list = ["Geen extra notities beschikbaar voor deze selectie."]
    for note in note_list[:8]:
        story.append(Paragraph(f"&bull; {note}", note_style))
        story.append(Spacer(1, 2))

    doc.build(story)
    return buffer.getvalue()
