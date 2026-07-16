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
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
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
        fontSize=23,
        textColor=colors.HexColor("#0B1020"),
        spaceAfter=4,
    )
    kicker_style = ParagraphStyle(
        "mvv_kicker",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        textColor=colors.HexColor("#C8102E"),
        leading=11,
        spaceAfter=12,
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

    story: list[object] = []
    story.append(Paragraph("Player Report", title_style))
    story.append(Paragraph(f"MVV Maastricht | {player_name} | {scope_label} | {period_label}", kicker_style))

    summary_rows = [
        ["Metriek", "Waarde", "Metriek", "Waarde"],
        ["Sessies", _fmt_int(summary.get("sessions")), "Actieve dagen", _fmt_int(summary.get("active_days"))],
        ["Total Distance", _fmt_distance(summary.get("total_distance")), "HSR / HSD", _fmt_distance(summary.get("hsr_hsd"))],
        ["Sprints", _fmt_int(summary.get("sprints")), "Duur", _fmt_int(summary.get("duration_min")) + " min" if not pd.isna(summary.get("duration_min")) else "--"],
        ["Avg Intensity", _fmt_dec(summary.get("distance_per_min"), 1) + " m/min" if not pd.isna(summary.get("distance_per_min")) else "--", "Top Speed", _fmt_speed(summary.get("top_speed"))],
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
    story.append(Paragraph("Summary", section_style))
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
        session_rows = [["Datum", "Type", "Event", "Distance", "HSR/HSD", "Sprints", "Max Speed"]]
        for _, row in sessions_preview.iterrows():
            session_rows.append(
                [
                    str(row.get("datum_label") or "--"),
                    str(row.get("type") or "--"),
                    str(row.get("event") or "--"),
                    _fmt_distance(row.get("total_distance")),
                    _fmt_distance(row.get("hsr_hsd")),
                    _fmt_int(row.get("number_of_sprints")),
                    _fmt_speed(row.get("max_speed")),
                ]
            )
        session_table = Table(
            session_rows,
            colWidths=[24 * mm, 28 * mm, 42 * mm, 28 * mm, 28 * mm, 18 * mm, 24 * mm],
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
                    ("FONTSIZE", (0, 0), (-1, -1), 7.5),
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
        story.append(Paragraph(f"• {note}", body_style))
        story.append(Spacer(1, 2))

    doc.build(story)
    return buffer.getvalue()
