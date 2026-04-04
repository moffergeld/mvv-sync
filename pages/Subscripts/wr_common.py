# pages/Subscripts/wr_common.py (aangevuld met MVV styling functies)
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# -----------------------------
# Config / constants
# -----------------------------
CHART_H = 420

ASRM_PARAMS = [
    ("Muscle soreness", "muscle_soreness"),
    ("Fatigue", "fatigue"),
    ("Sleep quality", "sleep_quality"),
    ("Stress", "stress"),
    ("Mood", "mood"),
]

RPE_PARAMS = [
    ("RPE (weighted avg)", "avg_rpe"),
    ("RPE Load (sum dur*rpe)", "rpe_load"),
    ("Total duration (min)", "duration_min"),
]

ZONE_GREEN_MAX = 4.5
ZONE_ORANGE_MAX = 7.5

ASRM_RED_THRESHOLD = 7.5
RPE_RED_THRESHOLD = 7.5

# MVV Design System Colors
MVV_COLORS = {
    'primary': '#C8102E',
    'light': '#E8213F',
    'dark': '#8B0A1F',
    'background': '#0D0E13',
    'card': 'rgba(255, 255, 255, 0.04)',
    'text_primary': '#F0F0F0',
    'text_muted': 'rgba(240, 240, 240, 0.45)',
    'grid': 'rgba(255, 255, 255, 0.09)',
    'zone_green': 'rgba(0, 200, 0, 0.12)',
    'zone_orange': 'rgba(255, 165, 0, 0.14)',
    'zone_red': 'rgba(255, 0, 0, 0.14)'
}

def create_mvv_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str = "", 
                        show_zones: bool = False, y_range: tuple = (0, 10)) -> go.Figure:
    """Create a styled bar chart according to MVV design system"""
    
    fig = go.Figure()
    
    # Add bars with error bars if available
    if 'std' in df.columns:
        fig.add_trace(go.Bar(
            x=df[x_col],
            y=df[y_col],
            error_y=dict(
                type='data',
                array=df['std'] if 'std' in df.columns else None,
                color=MVV_COLORS['light'],
                thickness=2,
                width=6
            ),
            marker=dict(
                color=MVV_COLORS['primary'],
                line=dict(color=MVV_COLORS['light'], width=2)
            ),
            opacity=0.9
        ))
    else:
        fig.add_trace(go.Bar(
            x=df[x_col],
            y=df[y_col],
            marker=dict(
                color=MVV_COLORS['primary'],
                line=dict(color=MVV_COLORS['light'], width=2)
            ),
            opacity=0.9
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family="DM Sans", size=16, color=MVV_COLORS['text_primary']),
            x=0.5
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=MVV_COLORS['background'],
        font=dict(family="DM Sans", size=12, color=MVV_COLORS['text_primary']),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(color=MVV_COLORS['text_primary']),
            title_font=dict(color=MVV_COLORS['text_primary'])
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor=MVV_COLORS['grid'],
            tickfont=dict(color=MVV_COLORS['text_primary']),
            title_font=dict(color=MVV_COLORS['text_primary']),
            range=y_range
        ),
        bargap=0.3,
        margin=dict(l=50, r=30, t=50, b=50),
        showlegend=False
    )
    
    # Add zones if requested
    if show_zones:
        zones = [
            (0, ZONE_GREEN_MAX, MVV_COLORS['zone_green']),
            (ZONE_GREEN_MAX, ZONE_ORANGE_MAX, MVV_COLORS['zone_orange']),
            (ZONE_ORANGE_MAX, y_range[1], MVV_COLORS['zone_red']),
        ]
        for y0, y1, color in zones:
            fig.add_shape(
                type="rect",
                xref="paper",
                yref="y",
                x0=0,
                x1=1,
                y0=y0,
                y1=y1,
                fillcolor=color,
                line=dict(width=0),
                layer="below",
            )
    
    return fig

def create_mvv_line_chart(df: pd.DataFrame, x_col: str, y_cols: list, title: str = "") -> go.Figure:
    """Create a styled line chart according to MVV design system"""
    
    fig = go.Figure()
    
    colors = [MVV_COLORS['primary'], MVV_COLORS['light'], MVV_COLORS['dark']]
    
    for i, col in enumerate(y_cols):
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[col],
            mode='lines+markers',
            name=col,
            line=dict(color=colors[i % len(colors)], width=3),
            marker=dict(
                size=8,
                line=dict(width=2, color=MVV_COLORS['text_primary'])
            ),
            hovertemplate=f'<b>{col}</b><br>Datum: %{{x}}<br>Waarde: %{{y}}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family="DM Sans", size=16, color=MVV_COLORS['text_primary']),
            x=0.5
        ),
        paper_bgcolor=MVV_COLORS['background'],
        plot_bgcolor='rgba(255, 255, 255, 0.02)',
        font=dict(family="DM Sans", color=MVV_COLORS['text_primary']),
        xaxis=dict(
            showgrid=True,
            gridcolor=MVV_COLORS['grid'],
            tickfont=dict(color=MVV_COLORS['text_primary']),
            title_font=dict(color=MVV_COLORS['text_primary'])
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=MVV_COLORS['grid'],
            tickfont=dict(color=MVV_COLORS['text_primary']),
            title_font=dict(color=MVV_COLORS['text_primary'])
        ),
        legend=dict(
            font=dict(color=MVV_COLORS['text_primary']),
            bgcolor='rgba(0,0,0,0)'
        )
    )
    
    return fig

def _df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows or [])

# ... rest van de bestaande code blijft hetzelfde ...

def plot_week_player_mean_std_bars(
    df_stats: pd.DataFrame,
    player_name_col: str,
    mean_col: str = "mean",
    std_col: str = "std",
    y_title: str = "",
    zone_0_10: bool = False,
) -> None:
    if df_stats.empty:
        st.info("Geen data voor deze week/selectie.")
        return

    # Sort by mean value descending
    df_sorted = df_stats.sort_values(mean_col, ascending=False)
    
    fig = create_mvv_bar_chart(
        df=df_sorted,
        x_col=player_name_col,
        y_col=mean_col,
        title=y_title,
        show_zones=zone_0_10,
        y_range=(0, 10) if zone_0_10 else None
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_day_bars(df: pd.DataFrame, x_col: str, y_col: str, y_title: str, zone_0_10: bool) -> None:
    if df.empty:
        st.info("Geen data voor deze selectie.")
        return

    # Sort by value descending
    df_sorted = df.sort_values(y_col, ascending=False)
    
    fig = create_mvv_bar_chart(
        df=df_sorted,
        x_col=x_col,
        y_col=y_col,
        title=y_title,
        show_zones=zone_0_10,
        y_range=(0, 10) if zone_0_10 else None
    )
    
    st.plotly_chart(fig, use_container_width=True)
