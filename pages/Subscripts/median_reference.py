from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


MetricSpec = tuple[str, str, str, float, int]


def build_selected_medians(
    frame: pd.DataFrame,
    selected_players: Sequence[str],
    metric_specs: Sequence[MetricSpec],
    *,
    player_column: str,
) -> pd.DataFrame:
    """Return one median row per metric for the selected player cohort."""
    columns = ["metric", "parameter", "median", "unit", "players", "decimals"]
    if frame is None or frame.empty or player_column not in frame.columns or not selected_players:
        return pd.DataFrame(columns=columns)

    selected = {str(player) for player in selected_players}
    cohort = frame.loc[frame[player_column].fillna("").astype(str).isin(selected)].copy()
    if cohort.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for metric, label, unit, scale, decimals in metric_specs:
        if metric not in cohort.columns:
            continue
        values = pd.to_numeric(cohort[metric], errors="coerce").dropna().mul(float(scale))
        if values.empty:
            continue
        rows.append(
            {
                "metric": metric,
                "parameter": label,
                "median": float(values.median()),
                "unit": unit,
                "players": int(values.size),
                "decimals": int(decimals),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def median_map(summary: pd.DataFrame) -> dict[str, float]:
    if summary is None or summary.empty:
        return {}
    return {
        str(row.metric): float(row.median)
        for row in summary.itertuples(index=False)
        if pd.notna(row.median)
    }


def display_median_table(summary: pd.DataFrame) -> pd.DataFrame:
    if summary is None or summary.empty:
        return pd.DataFrame(columns=["Parameter", "Mediaan", "Eenheid", "Spelers"])
    rows: list[dict[str, object]] = []
    for row in summary.itertuples(index=False):
        decimals = int(row.decimals)
        formatted = f"{float(row.median):,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
        rows.append(
            {
                "Parameter": str(row.parameter),
                "Mediaan": formatted,
                "Eenheid": str(row.unit),
                "Spelers": int(row.players),
            }
        )
    return pd.DataFrame(rows)
