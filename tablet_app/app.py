from __future__ import annotations

import base64
import html
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

import extra_streamlit_components as stx
import streamlit as st
import streamlit.components.v1 as components
from supabase import create_client


THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
PAGES_DIR = ROOT_DIR / "pages"
TABLET_ASSETS_DIR = THIS_DIR / "assets"
INJURY_BODY_SELECTOR_COMPONENT = components.declare_component(
    "mvv_injury_body_selector",
    path=str(ROOT_DIR / "components" / "injury_body_selector"),
)

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(PAGES_DIR) not in sys.path:
    sys.path.insert(0, str(PAGES_DIR))

from Subscripts.player_tab_forms import (  # noqa: E402
    _legend_asrm,
    _legend_rpe,
    load_asrm,
    load_rpe,
)


APP_TITLE = "MVV Tablet Check-in"
CLUB_NAME = "MVV Maastricht"
ACCESS_COOKIE_NAME = "mvv_tablet_access"
ACCESS_COOKIE_SECONDS = 60 * 60 * 24 * 30
TABLET_PLAYER_CACHE_TTL_SECONDS = 120
TABLET_COMPLETION_CACHE_TTL_SECONDS = 45
TABLET_FORM_CACHE_TTL_SECONDS = 300
RPE_DURATION_OPTIONS = list(range(30, 111, 5))
RPE_BULK_DURATION_DEFAULT = 60
INJURY_LOCATION_OPTIONS = [
    "None",
    "Foot",
    "Ankle",
    "Lower leg",
    "Knee",
    "Upper leg",
    "Hip",
    "Groin",
    "Glute",
    "Lower back",
    "Abdomen",
    "Chest",
    "Shoulder",
    "Upper arm",
    "Elbow",
    "Forearm",
    "Wrist",
    "Hand",
    "Neck",
    "Head",
    "Other",
]
INJURY_LOCATION_LABELS = {
    "None": "Geen",
    "Foot": "Voet",
    "Ankle": "Enkel",
    "Lower leg": "Onderbeen",
    "Knee": "Knie",
    "Upper leg": "Bovenbeen",
    "Hip": "Heup",
    "Groin": "Lies",
    "Glute": "Bil",
    "Lower back": "Onderrug",
    "Abdomen": "Buik",
    "Chest": "Borst",
    "Shoulder": "Schouder",
    "Upper arm": "Bovenarm",
    "Elbow": "Elleboog",
    "Forearm": "Onderarm",
    "Wrist": "Pols",
    "Hand": "Hand",
    "Neck": "Nek",
    "Head": "Hoofd",
    "Other": "Overig",
}
INJURY_BODY_IMAGE_MARKERS = [
    {"id": "front_head", "value": "Head", "top": 10.6, "left": 29.1, "width": 8.8, "height": 12.8},
    {"id": "back_head", "value": "Head", "top": 10.9, "left": 70.7, "width": 8.8, "height": 12.8},
    {"id": "front_neck", "value": "Neck", "top": 19.2, "left": 29.1, "width": 4.2, "height": 4.0},
    {"id": "back_neck", "value": "Neck", "top": 19.6, "left": 70.7, "width": 4.2, "height": 4.0},
    {"id": "front_shoulder_left", "value": "Shoulder", "top": 23.5, "left": 23.3, "width": 9.0, "height": 4.9},
    {"id": "front_shoulder_right", "value": "Shoulder", "top": 23.5, "left": 34.9, "width": 9.0, "height": 4.9},
    {"id": "back_shoulder_left", "value": "Shoulder", "top": 23.5, "left": 64.6, "width": 9.0, "height": 4.9},
    {"id": "back_shoulder_right", "value": "Shoulder", "top": 23.5, "left": 76.8, "width": 9.0, "height": 4.9},
    {"id": "front_chest", "value": "Chest", "top": 32.2, "left": 29.1, "width": 15.2, "height": 10.8},
    {"id": "front_upperarm_left", "value": "Upper arm", "top": 40.3, "left": 18.1, "width": 5.8, "height": 13.0},
    {"id": "front_upperarm_right", "value": "Upper arm", "top": 40.3, "left": 40.1, "width": 5.8, "height": 13.0},
    {"id": "back_upperarm_left", "value": "Upper arm", "top": 40.2, "left": 58.6, "width": 5.8, "height": 13.0},
    {"id": "back_upperarm_right", "value": "Upper arm", "top": 40.2, "left": 82.8, "width": 5.8, "height": 13.0},
    {"id": "front_elbow_left", "value": "Elbow", "top": 36.2, "left": 16.3, "width": 4.2, "height": 4.2},
    {"id": "front_elbow_right", "value": "Elbow", "top": 36.2, "left": 41.7, "width": 4.2, "height": 4.2},
    {"id": "back_elbow_left", "value": "Elbow", "top": 36.1, "left": 57.0, "width": 4.2, "height": 4.2},
    {"id": "back_elbow_right", "value": "Elbow", "top": 36.1, "left": 84.4, "width": 4.2, "height": 4.2},
    {"id": "front_forearm_left", "value": "Forearm", "top": 48.4, "left": 16.3, "width": 5.2, "height": 13.8},
    {"id": "front_forearm_right", "value": "Forearm", "top": 48.4, "left": 41.6, "width": 5.2, "height": 13.8},
    {"id": "back_forearm_left", "value": "Forearm", "top": 48.4, "left": 56.9, "width": 5.2, "height": 13.8},
    {"id": "back_forearm_right", "value": "Forearm", "top": 48.4, "left": 84.4, "width": 5.2, "height": 13.8},
    {"id": "front_wrist_left", "value": "Wrist", "top": 59.5, "left": 16.0, "width": 3.2, "height": 3.2},
    {"id": "front_wrist_right", "value": "Wrist", "top": 59.5, "left": 41.9, "width": 3.2, "height": 3.2},
    {"id": "back_wrist_left", "value": "Wrist", "top": 59.6, "left": 56.6, "width": 3.2, "height": 3.2},
    {"id": "back_wrist_right", "value": "Wrist", "top": 59.6, "left": 84.8, "width": 3.2, "height": 3.2},
    {"id": "front_hand_left", "value": "Hand", "top": 63.3, "left": 12.9, "width": 4.8, "height": 6.0},
    {"id": "front_hand_right", "value": "Hand", "top": 63.3, "left": 43.8, "width": 4.8, "height": 6.0},
    {"id": "back_hand_left", "value": "Hand", "top": 63.3, "left": 54.1, "width": 4.8, "height": 6.0},
    {"id": "back_hand_right", "value": "Hand", "top": 63.3, "left": 86.8, "width": 4.8, "height": 6.0},
    {"id": "front_abdomen", "value": "Abdomen", "top": 44.8, "left": 29.1, "width": 12.2, "height": 9.6},
    {"id": "front_groin", "value": "Groin", "top": 54.0, "left": 29.1, "width": 6.8, "height": 4.6},
    {"id": "back_lowerback", "value": "Lower back", "top": 44.5, "left": 70.7, "width": 11.2, "height": 11.6},
    {"id": "front_hip", "value": "Hip", "top": 60.0, "left": 29.1, "width": 12.8, "height": 5.8},
    {"id": "back_glute", "value": "Glute", "top": 58.3, "left": 70.7, "width": 12.8, "height": 7.4},
    {"id": "back_hip", "value": "Hip", "top": 63.8, "left": 70.7, "width": 12.8, "height": 5.8},
    {"id": "front_upperleg", "value": "Upper leg", "top": 69.5, "left": 29.1, "width": 12.0, "height": 15.6},
    {"id": "back_upperleg", "value": "Upper leg", "top": 69.4, "left": 70.7, "width": 12.0, "height": 15.6},
    {"id": "front_knee", "value": "Knee", "top": 81.0, "left": 29.1, "width": 7.8, "height": 6.4},
    {"id": "back_knee", "value": "Knee", "top": 81.0, "left": 70.7, "width": 7.8, "height": 6.4},
    {"id": "front_lowerleg", "value": "Lower leg", "top": 88.5, "left": 29.1, "width": 8.8, "height": 13.8},
    {"id": "back_lowerleg", "value": "Lower leg", "top": 88.5, "left": 70.7, "width": 8.8, "height": 13.8},
    {"id": "front_ankle", "value": "Ankle", "top": 95.0, "left": 29.1, "width": 5.0, "height": 3.4},
    {"id": "back_ankle", "value": "Ankle", "top": 95.0, "left": 70.7, "width": 5.0, "height": 3.4},
    {"id": "front_foot", "value": "Foot", "top": 98.0, "left": 29.1, "width": 11.0, "height": 3.8},
    {"id": "back_foot", "value": "Foot", "top": 98.0, "left": 70.7, "width": 11.0, "height": 3.8},
]


st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;600;700;800;900&display=swap');

      :root {
        --mvv-red: #c8102e;
        --mvv-dark-red: #8f0b20;
        --mvv-deep: #14070a;
        --mvv-cream: #fff7ef;
        --mvv-soft: #f7e9e7;
        --mvv-page: #E7E5E4;
        --mvv-gold: #d6a94a;
        --mvv-border: rgba(200, 16, 46, 0.20);
      }

      #MainMenu,
      header,
      footer,
      [data-testid="stSidebar"],
      [data-testid="collapsedControl"] {
        display: none !important;
      }

      .stApp {
        background:
          radial-gradient(circle at top left, rgba(200, 16, 46, 0.22), transparent 34rem),
          linear-gradient(135deg, #efedec 0%, #e9e7e6 48%, var(--mvv-page) 100%);
      }

      .block-container {
        max-width: 1240px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        font-family: 'Inter', sans-serif;
      }

      h1, h2, h3 {
        letter-spacing: 0.01em;
      }

      div.stButton > button {
        min-height: 5.25rem;
        border-radius: 18px;
        border: 1px solid var(--mvv-border);
        background: linear-gradient(145deg, #ffffff 0%, #fff4f1 100%);
        box-shadow: 0 12px 24px rgba(78, 8, 18, 0.10);
        color: var(--mvv-deep);
        font-size: 1rem;
        font-weight: 900;
        line-height: 1.35;
        white-space: pre-line;
        transition: transform 0.12s ease, box-shadow 0.12s ease, border-color 0.12s ease;
      }

      div.stButton > button:hover {
        transform: translateY(-2px);
        border-color: rgba(200, 16, 46, 0.55);
        box-shadow: 0 16px 32px rgba(78, 8, 18, 0.16);
      }

      div.stButton > button:active {
        transform: translateY(0px);
      }

      div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(200, 16, 46, 0.14);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: 0 10px 26px rgba(78, 8, 18, 0.08);
      }

      div[data-testid="stMetricValue"],
      div[data-testid="stMetricValue"] > div,
      div[data-testid="stMetricValue"] p,
      div[data-testid="stMetricValue"] span,
      [data-testid="stMetric"] [data-testid="stMetricValue"],
      [data-testid="stMetric"] [data-testid="stMetricValue"] * {
        color: var(--mvv-red) !important;
        font-size: 2.15rem;
        font-weight: 900 !important;
      }

      div[data-testid="stMetricLabel"] p {
        font-weight: 800;
        color: rgba(20, 7, 10, 0.72);
      }

      .tablet-hero {
        position: relative;
        overflow: hidden;
        display: flex;
        gap: 1rem;
        align-items: center;
        padding: 1.35rem 1.45rem;
        border: 1px solid rgba(255, 255, 255, 0.42);
        border-radius: 24px;
        background:
          linear-gradient(135deg, rgba(200, 16, 46, 0.96) 0%, rgba(143, 11, 32, 0.96) 54%, rgba(20, 7, 10, 0.96) 100%);
        color: white;
        box-shadow: 0 18px 46px rgba(78, 8, 18, 0.24);
        margin-bottom: 1.1rem;
      }

      .tablet-hero::after {
        content: '';
        position: absolute;
        right: -5rem;
        top: -6rem;
        width: 18rem;
        height: 18rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.10);
      }

      .mvv-logo-wrap {
        z-index: 1;
        width: 88px;
        height: 88px;
        min-width: 88px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 24px;
        background: rgba(255, 255, 255, 0.92);
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.4), 0 12px 24px rgba(0,0,0,0.16);
      }

      .mvv-logo-wrap img {
        max-width: 76px;
        max-height: 76px;
        object-fit: contain;
      }

      .mvv-logo-fallback {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.65rem;
        color: var(--mvv-red);
        letter-spacing: 0.02em;
      }

      .tablet-hero-content {
        z-index: 1;
      }

      .tablet-hero-kicker {
        margin: 0 0 0.2rem 0;
        text-transform: uppercase;
        font-size: 0.78rem;
        font-weight: 900;
        letter-spacing: 0.16em;
        color: rgba(255, 255, 255, 0.78) !important;
      }

      .tablet-hero-title {
        margin: 0;
        font-family: 'Bebas Neue', sans-serif;
        font-size: clamp(2.4rem, 5vw, 4.2rem);
        line-height: 0.95;
        letter-spacing: 0.02em;
        color: #ffffff !important;
      }

      .tablet-hero-subtitle {
        margin: 0.45rem 0 0 0;
        font-size: 1.02rem;
        font-weight: 600;
        color: rgba(255, 255, 255, 0.88) !important;
      }

      .mvv-section-card {
        padding: 1rem 1.1rem;
        border-radius: 20px;
        border: 1px solid rgba(200, 16, 46, 0.16);
        background: rgba(255, 255, 255, 0.84);
        box-shadow: 0 12px 30px rgba(78, 8, 18, 0.08);
        margin: 0.6rem 0 1rem 0;
      }


      .mvv-kpi-card {
        position: relative;
        overflow: hidden;
        padding: 1rem 1.15rem 0.95rem 1.15rem;
        border-radius: 24px;
        border: 0;
        box-shadow: 0 16px 34px rgba(78, 8, 18, 0.16);
        min-height: 122px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        margin-bottom: 0.6rem;
      }

      .mvv-kpi-card::after {
        content: "";
        position: absolute;
        top: -34px;
        right: -20px;
        width: 112px;
        height: 112px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.1);
      }

      .mvv-kpi-card-wellness {
        background: linear-gradient(135deg, #16572d 0%, #1f8a3b 45%, #2a9d51 100%);
      }

      .mvv-kpi-card-rpe {
        background: linear-gradient(135deg, #6f1225 0%, #98172b 48%, #c8102e 100%);
      }

      .mvv-kpi-head {
        position: relative;
        z-index: 1;
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 0.75rem;
      }

      .mvv-kpi-label {
        font-size: 0.9rem;
        font-weight: 900;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #ffffff !important;
      }

      .mvv-kpi-note {
        font-size: 0.72rem;
        font-weight: 800;
        line-height: 1.2;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        text-align: right;
        color: rgba(255, 255, 255, 0.84) !important;
      }

      .mvv-kpi-value {
        position: relative;
        z-index: 1;
        margin-top: 0.45rem;
        font-size: 2.35rem;
        line-height: 1;
        font-weight: 900;
        color: #ffffff !important;
      }

      .mvv-kpi-progress {
        position: relative;
        z-index: 1;
        margin-top: 0.8rem;
        height: 10px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.22);
        overflow: hidden;
      }

      .mvv-kpi-progress span {
        display: block;
        height: 100%;
        border-radius: inherit;
        background: linear-gradient(90deg, rgba(255, 255, 255, 0.72) 0%, #ffffff 100%);
      }

      .mvv-kpi-hint {
        margin: 0.1rem 0 0.55rem 0;
        text-align: center;
        font-size: 0.88rem;
        font-weight: 800;
        color: rgba(20, 7, 10, 0.68) !important;
      }

      .mvv-mini-stat {
        min-height: 110px;
        padding: 0.95rem 1rem;
        border-radius: 18px;
        border: 1px solid rgba(200, 16, 46, 0.14);
        background: linear-gradient(160deg, rgba(255, 255, 255, 0.96) 0%, rgba(255, 245, 242, 0.92) 100%);
        box-shadow: 0 10px 24px rgba(78, 8, 18, 0.06);
        display: flex;
        flex-direction: column;
        justify-content: center;
        margin-bottom: 0.6rem;
      }

      .mvv-mini-stat-label {
        font-size: 0.82rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: rgba(20, 7, 10, 0.62) !important;
      }

      .mvv-mini-stat-value {
        margin-top: 0.28rem;
        font-size: 1.7rem;
        line-height: 1;
        font-weight: 900;
        color: var(--mvv-deep) !important;
      }

      .mvv-mini-stat-note {
        margin-top: 0.3rem;
        font-size: 0.9rem;
        font-weight: 600;
        color: rgba(20, 7, 10, 0.72) !important;
      }

      .mvv-form-kicker {
        margin: 0;
        text-transform: uppercase;
        font-size: 0.76rem;
        font-weight: 900;
        letter-spacing: 0.14em;
        color: var(--mvv-red) !important;
      }

      .mvv-form-title {
        margin: 0.25rem 0 0 0;
        font-family: 'Bebas Neue', sans-serif;
        font-size: 2.05rem;
        line-height: 0.98;
        letter-spacing: 0.02em;
        color: var(--mvv-deep) !important;
      }

      .mvv-form-subtitle {
        margin: 0.32rem 0 0.8rem 0;
        font-size: 0.98rem;
        font-weight: 600;
        color: rgba(20, 7, 10, 0.78) !important;
      }

      .mvv-session-title {
        margin: 0 0 0.14rem 0;
        font-size: 1.18rem;
        font-weight: 900;
        color: var(--mvv-deep) !important;
      }

      .mvv-session-note {
        margin: 0 0 0.75rem 0;
        font-size: 0.92rem;
        font-weight: 600;
        color: rgba(20, 7, 10, 0.72) !important;
      }

      .mvv-duration-title {
        margin: 0.3rem 0 0.5rem 0;
        font-size: 0.9rem;
        font-weight: 800;
        color: rgba(20, 7, 10, 0.78) !important;
      }

      .mvv-bulk-player-name {
        margin: 0.2rem 0 0.12rem 0;
        font-size: 1.2rem;
        line-height: 1.1;
        font-weight: 900;
        color: var(--mvv-deep) !important;
      }

      .mvv-bulk-player-note {
        margin: 0 0 0.45rem 0;
        font-size: 0.92rem;
        font-weight: 600;
        color: rgba(20, 7, 10, 0.72) !important;
      }

      .mvv-bulk-shared-title {
        margin: 0.1rem 0 0.45rem 0;
        font-size: 1rem;
        font-weight: 900;
        color: var(--mvv-deep) !important;
      }

      .mvv-load-pill {
        margin: 0.6rem 0 0.35rem 0;
        padding: 0.72rem 0.85rem;
        border-radius: 16px;
        border: 1px solid rgba(200, 16, 46, 0.14);
        background: rgba(255, 248, 246, 0.95);
        font-size: 0.95rem;
        font-weight: 700;
        color: var(--mvv-deep) !important;
      }

      .mvv-load-pill strong {
        color: var(--mvv-red) !important;
        font-size: 1.12rem;
      }

      .mvv-muted-box {
        margin: 0.35rem 0 0.2rem 0;
        padding: 0.85rem 0.95rem;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.74);
        border: 1px dashed rgba(200, 16, 46, 0.22);
        color: rgba(20, 7, 10, 0.72) !important;
        font-size: 0.94rem;
        font-weight: 600;
      }

      .tablet-hero a,
      .tablet-hero a:visited,
      .tablet-hero a:hover,
      .tablet-hero a:active {
        color: #ffffff !important;
        text-decoration: none !important;
      }

      .tablet-hero svg,
      .tablet-hero [data-testid="stHeaderActionElements"] {
        display: none !important;
      }

      .mvv-note {
        padding: 0.9rem 1rem;
        border-radius: 16px;
        border-left: 6px solid var(--mvv-red);
        background: rgba(255, 255, 255, 0.80);
        font-weight: 800;
      }

      [data-testid="stTextInput"] input,
      [data-testid="stNumberInput"] input,
      textarea {
        border-radius: 14px !important;
      }

      .stApp [data-testid="stTextInput"] input,
      .stApp [data-testid="stNumberInput"] input,
      .stApp [data-baseweb="select"] > div,
      .stApp textarea {
        background: #ffffff !important;
        color: var(--mvv-deep) !important;
      }

      .stApp [data-baseweb="select"] * {
        color: var(--mvv-deep) !important;
      }

      .stApp [data-testid="stTextInput"] input,
      .stApp [data-testid="stNumberInput"] input,
      .stApp [data-baseweb="select"] > div,
      .stApp textarea {
        border: 1px solid rgba(200, 16, 46, 0.18) !important;
        box-shadow: none !important;
      }

      .stApp [data-testid="stNumberInput"] input,
      .stApp textarea {
        min-height: 54px;
        font-size: 1rem !important;
        font-weight: 700 !important;
      }

      .stApp [data-testid="stSlider"] label,
      .stApp [data-testid="stNumberInput"] label,
      .stApp [data-testid="stTextArea"] label,
      .stApp [data-testid="stSelectbox"] label {
        font-weight: 800 !important;
      }

      .stApp [data-testid="stSlider"] {
        padding: 0.2rem 0 0.8rem;
      }

      .stApp [data-testid="stSlider"] [data-baseweb="slider"] {
        padding-top: 0.5rem;
      }

      .stApp [data-testid="stSlider"] [data-baseweb="slider"] > div {
        height: 0.8rem !important;
      }

      .stApp [data-testid="stSlider"] [data-baseweb="slider"] > div > div {
        border-radius: 999px !important;
      }

      .stApp [data-testid="stSlider"] [data-baseweb="slider"] > div > div:first-child {
        background: rgba(200, 16, 46, 0.16) !important;
      }

      .stApp [data-testid="stSlider"] [data-baseweb="slider"] > div > div:nth-child(2) {
        background: linear-gradient(90deg, #ff7a80 0%, var(--mvv-red) 100%) !important;
      }

      .stApp [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
        width: 1.5rem !important;
        height: 1.5rem !important;
        border: 4px solid #ffffff !important;
        background: var(--mvv-red) !important;
        box-shadow:
          0 0 0 5px rgba(200, 16, 46, 0.18),
          0 10px 24px rgba(78, 8, 18, 0.22) !important;
      }

      .stApp [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"]:focus,
      .stApp [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"]:hover {
        box-shadow:
          0 0 0 7px rgba(200, 16, 46, 0.20),
          0 12px 28px rgba(78, 8, 18, 0.24) !important;
      }

      .stApp [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] > div {
        color: var(--mvv-red) !important;
        font-size: 1rem !important;
        font-weight: 900 !important;
      }

      .mvv-toggle-choice-title {
        margin: 0 0 0.45rem 0;
        font-size: 0.9rem;
        font-weight: 900;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: rgba(20, 7, 10, 0.76);
      }

      .mvv-inline-control-spacer {
        height: 1.45rem;
      }

      .mvv-injury-map-note {
        margin: 0.15rem 0 0.7rem 0;
        font-size: 0.95rem;
        font-weight: 700;
        color: rgba(20, 7, 10, 0.72) !important;
      }

      .mvv-body-image-card {
        margin: 0.2rem auto 0.85rem auto;
        padding: 0.9rem 0.8rem 1rem 0.8rem;
        border-radius: 26px;
        border: 1px solid rgba(200, 16, 46, 0.12);
        background: rgba(255, 255, 255, 0.72);
        box-shadow: 0 12px 28px rgba(78, 8, 18, 0.06);
      }

      .mvv-body-image-stage {
        position: relative;
        width: min(100%, 22rem);
        aspect-ratio: 1 / 1;
        margin: 0 auto;
        overflow: visible;
      }

      .mvv-body-image {
        width: 100%;
        height: 100%;
        object-fit: contain;
        display: block;
        user-select: none;
        pointer-events: none;
      }

      .mvv-body-marker {
        position: absolute;
        transform: translate(-50%, -50%);
        border-radius: 999px;
        border: 2px solid rgba(200, 16, 46, 0.16);
        background: rgba(200, 16, 46, 0.025);
        box-shadow: 0 4px 10px rgba(78, 8, 18, 0.04);
        transition: transform 0.12s ease, border-color 0.12s ease, background 0.12s ease, box-shadow 0.12s ease;
      }

      .mvv-body-marker:hover {
        transform: translate(-50%, -50%) scale(1.04);
        border-color: rgba(200, 16, 46, 0.34);
        background: rgba(200, 16, 46, 0.07);
      }

      .mvv-body-marker-active {
        border-color: var(--mvv-red) !important;
        background: rgba(200, 16, 46, 0.16) !important;
        box-shadow:
          0 0 0 5px rgba(200, 16, 46, 0.12),
          0 10px 22px rgba(78, 8, 18, 0.12) !important;
      }

      .mvv-body-image-actions {
        margin-top: 0.95rem;
        display: flex;
        justify-content: center;
        gap: 0.8rem;
        flex-wrap: wrap;
      }

      .mvv-body-image-action {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 9.75rem;
        min-height: 3rem;
        padding: 0.55rem 1rem;
        border-radius: 18px;
        border: 1px solid rgba(200, 16, 46, 0.18);
        background: linear-gradient(145deg, #ffffff 0%, #fff4f1 100%);
        box-shadow: 0 10px 22px rgba(78, 8, 18, 0.06);
        color: var(--mvv-deep) !important;
        font-size: 0.92rem;
        font-weight: 800;
        text-decoration: none !important;
      }

      .mvv-body-image-action:hover {
        border-color: rgba(200, 16, 46, 0.42);
      }

      .mvv-body-image-action-active {
        border-color: var(--mvv-red) !important;
        background: rgba(255, 122, 128, 0.14) !important;
        color: var(--mvv-red) !important;
      }

      .mvv-body-image-stage a,
      .mvv-body-image-actions a {
        text-decoration: none !important;
      }

      .mvv-body-image-shell {
        width: min(100%, 24rem);
        margin: 0 auto;
      }

      .stApp [data-testid="stRadio"] {
        margin: 0.35rem 0 1rem;
      }

      .stApp [data-testid="stRadio"] div[role="radiogroup"] {
        display: flex;
        gap: 0.45rem;
        width: fit-content;
      }

      .stApp [data-testid="stRadio"] div[role="radiogroup"] label {
        flex: 1 1 0;
        min-width: 8.75rem;
        min-height: 88px;
        margin: 0 !important;
        padding: 0.95rem 1.45rem;
        border-radius: 24px;
        border: 1px solid rgba(200, 16, 46, 0.18);
        background: rgba(255,255,255,0.90);
        box-shadow: 0 12px 26px rgba(78, 8, 18, 0.08);
        font-weight: 900;
        font-size: 1.08rem !important;
        display: flex !important;
        align-items: center;
        justify-content: flex-start;
        gap: 0.7rem;
        transition: border-color 0.2s ease, box-shadow 0.2s ease, background 0.2s ease, transform 0.12s ease;
      }

      .stApp [data-testid="stRadio"] div[role="radiogroup"] label:hover {
        transform: translateY(-1px);
        border-color: rgba(200, 16, 46, 0.28) !important;
        box-shadow: 0 14px 28px rgba(78, 8, 18, 0.12);
      }

      .stApp [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
        border-color: rgba(200, 16, 46, 0.38) !important;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(255, 122, 128, 0.16) 100%) !important;
        box-shadow: 0 16px 32px rgba(78, 8, 18, 0.14);
      }

      .stApp [data-testid="stRadio"] div[role="radiogroup"] label > div:first-of-type {
        transform: scale(1.3);
        transform-origin: center;
      }

      .stApp [data-testid="stRadio"] div[role="radiogroup"] label p,
      .stApp [data-testid="stRadio"] div[role="radiogroup"] label span {
        margin: 0 !important;
        white-space: nowrap !important;
        font-size: 1.08rem !important;
        font-weight: 900 !important;
        color: var(--mvv-deep) !important;
      }

      .stApp [data-testid="stRadio"] div[role="radiogroup"] label > div:last-of-type {
        flex: 1 1 auto;
        min-width: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
      }

      [class*="st-key-tablet_rpe_s1_dur_choice_"] [data-testid="stRadio"],
      [class*="st-key-tablet_rpe_s2_dur_choice_"] [data-testid="stRadio"],
      [class*="st-key-tablet_bulk_rpe_dur_"] [data-testid="stRadio"] {
        margin: 0.15rem 0 1rem;
      }

      [class*="st-key-tablet_rpe_s1_dur_choice_"] [data-testid="stRadio"] div[role="radiogroup"],
      [class*="st-key-tablet_rpe_s2_dur_choice_"] [data-testid="stRadio"] div[role="radiogroup"],
      [class*="st-key-tablet_bulk_rpe_dur_"] [data-testid="stRadio"] div[role="radiogroup"] {
        width: 100%;
        gap: 0.42rem;
        flex-wrap: wrap;
      }

      [class*="st-key-tablet_rpe_s1_dur_choice_"] [data-testid="stRadio"] div[role="radiogroup"] label,
      [class*="st-key-tablet_rpe_s2_dur_choice_"] [data-testid="stRadio"] div[role="radiogroup"] label,
      [class*="st-key-tablet_bulk_rpe_dur_"] [data-testid="stRadio"] div[role="radiogroup"] label {
        flex: 0 0 calc(25% - 0.35rem);
        min-width: 4.25rem;
        min-height: 3.25rem;
        padding: 0.45rem 0.5rem;
        border-radius: 16px;
        justify-content: center;
        gap: 0.38rem;
      }

      [class*="st-key-tablet_rpe_s1_dur_choice_"] [data-testid="stRadio"] div[role="radiogroup"] label > div:last-of-type,
      [class*="st-key-tablet_rpe_s2_dur_choice_"] [data-testid="stRadio"] div[role="radiogroup"] label > div:last-of-type,
      [class*="st-key-tablet_bulk_rpe_dur_"] [data-testid="stRadio"] div[role="radiogroup"] label > div:last-of-type {
        justify-content: center;
      }

      [class*="st-key-tablet_rpe_s1_dur_choice_"] [data-testid="stRadio"] div[role="radiogroup"] label p,
      [class*="st-key-tablet_rpe_s2_dur_choice_"] [data-testid="stRadio"] div[role="radiogroup"] label p,
      [class*="st-key-tablet_rpe_s1_dur_choice_"] [data-testid="stRadio"] div[role="radiogroup"] label span,
      [class*="st-key-tablet_rpe_s2_dur_choice_"] [data-testid="stRadio"] div[role="radiogroup"] label span,
      [class*="st-key-tablet_bulk_rpe_dur_"] [data-testid="stRadio"] div[role="radiogroup"] label p,
      [class*="st-key-tablet_bulk_rpe_dur_"] [data-testid="stRadio"] div[role="radiogroup"] label span {
        font-size: 0.98rem !important;
      }

      @media (max-width: 1100px) {
        [class*="st-key-tablet_rpe_s1_dur_choice_"] [data-testid="stRadio"] div[role="radiogroup"] label,
        [class*="st-key-tablet_rpe_s2_dur_choice_"] [data-testid="stRadio"] div[role="radiogroup"] label,
        [class*="st-key-tablet_bulk_rpe_dur_"] [data-testid="stRadio"] div[role="radiogroup"] label {
          flex-basis: calc(33.333% - 0.35rem);
        }
      }


      /* Force readable text even when Streamlit/account dark mode is enabled */
      .stApp,
      .stApp label,
      .stApp .stMarkdown,
      .stApp .stMarkdown p,
      .stApp .stMarkdown li,
      .stApp .stMarkdown span,
      .stApp .stMarkdown strong,
      .stApp [data-testid="stMarkdownContainer"] p,
      .stApp [data-testid="stMarkdownContainer"] li,
      .stApp [data-testid="stMetricLabel"] p,
      .stApp [data-testid="stTextInputRootElement"] label,
      .stApp [data-testid="stNumberInput"] label,
      .stApp [data-testid="stSelectbox"] label,
      .stApp [data-testid="stTextArea"] label,
      .stApp [data-testid="stSlider"] label,
      .stApp [data-testid="stRadio"] label,
      .stApp [data-testid="stToggle"] label,
      .stApp [data-testid="stWidgetLabel"],
      .stApp [data-testid="stCaptionContainer"] {
        color: var(--mvv-deep) !important;
      }

      .stApp [data-testid="stMetricLabel"] {
        opacity: 1 !important;
      }

      .stApp input::placeholder,
      .stApp textarea::placeholder {
        color: rgba(20, 7, 10, 0.62) !important;
      }

      .mvv-section-card,
      .mvv-note {
        color: var(--mvv-deep) !important;
      }

      .tablet-hero,
      .tablet-hero *,
      .mvv-logo-wrap,
      .mvv-logo-wrap * {
        color: white !important;
      }

      .tablet-hero-content,
      .tablet-hero-content *,
      .tablet-hero-title,
      .tablet-hero-subtitle,
      .tablet-hero-kicker {
        color: #ffffff !important;
      }

      .mvv-logo-fallback {
        color: var(--mvv-red) !important;
      }

      div.stFormSubmitButton > button {
        background: #ffffff !important;
        color: var(--mvv-deep) !important;
        border: 1px solid rgba(200, 16, 46, 0.20) !important;
        box-shadow: 0 10px 24px rgba(78, 8, 18, 0.06) !important;
      }

      div.stFormSubmitButton > button:hover {
        background: #ffffff !important;
        color: var(--mvv-deep) !important;
        border: 1px solid rgba(200, 16, 46, 0.32) !important;
      }

      div.stFormSubmitButton > button * {
        color: var(--mvv-deep) !important;
      }

      [class*="st-key-tablet_nav_"] button {
        min-height: 136px !important;
        border-radius: 24px !important;
        padding: 1.1rem 1.35rem !important;
        background: rgba(255, 255, 255, 0.92) !important;
        border: 2px solid rgba(200, 16, 46, 0.14) !important;
        box-shadow: 0 12px 30px rgba(78, 8, 18, 0.08) !important;
        position: relative !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        font-size: 0 !important;
        color: transparent !important;
      }

      [class*="st-key-tablet_nav_"] button p,
      [class*="st-key-tablet_nav_"] button span,
      [class*="st-key-tablet_nav_"] button [data-testid="stMarkdownContainer"] {
        font-size: 0 !important;
        color: transparent !important;
        margin: 0 !important;
      }

      [class*="st-key-tablet_nav_"] button::before {
        position: absolute;
        top: 22px;
        left: 0;
        right: 0;
        font-size: 0.95rem;
        line-height: 1.1;
        font-weight: 700;
        color: var(--mvv-deep);
        text-align: center;
      }

      [class*="st-key-tablet_nav_"] button::after {
        position: absolute;
        top: 54px;
        left: 0;
        right: 0;
        font-size: 2.25rem;
        line-height: 1.05;
        font-weight: 900;
        text-align: center;
      }

      [class*="st-key-tablet_nav_"] button:hover {
        border-color: var(--mvv-red) !important;
        background: rgba(255, 255, 255, 0.98) !important;
      }

      [class*="st-key-tablet_nav_"][class*="_active"] button {
        border-color: var(--mvv-red) !important;
        box-shadow: 0 0 0 4px rgba(200, 16, 46, 0.10), 0 12px 30px rgba(78, 8, 18, 0.08) !important;
      }

      .st-key-tablet_nav_wellness_ok_active button::before,
      .st-key-tablet_nav_wellness_ok_inactive button::before,
      .st-key-tablet_nav_wellness_open_active button::before,
      .st-key-tablet_nav_wellness_open_inactive button::before {
        content: "Wellness";
      }

      .st-key-tablet_nav_rpe_ok_active button::before,
      .st-key-tablet_nav_rpe_ok_inactive button::before,
      .st-key-tablet_nav_rpe_open_active button::before,
      .st-key-tablet_nav_rpe_open_inactive button::before {
        content: "RPE";
      }

      .st-key-tablet_nav_wellness_ok_active button::after,
      .st-key-tablet_nav_wellness_ok_inactive button::after,
      .st-key-tablet_nav_rpe_ok_active button::after,
      .st-key-tablet_nav_rpe_ok_inactive button::after {
        content: "OK";
        color: #1f8a3b;
      }

      .st-key-tablet_nav_wellness_open_active button::after,
      .st-key-tablet_nav_wellness_open_inactive button::after,
      .st-key-tablet_nav_rpe_open_active button::after,
      .st-key-tablet_nav_rpe_open_inactive button::after {
        content: "Open";
        color: var(--mvv-red);
      }

      .mvv-player-card {
        min-height: 124px;
        padding: 1rem 1rem 0.95rem 1rem;
        border-radius: 22px;
        border: 1px solid rgba(200, 16, 46, 0.18);
        background: linear-gradient(145deg, #ffffff 0%, #fff7f5 100%);
        box-shadow: 0 12px 24px rgba(78, 8, 18, 0.08);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        margin-bottom: 0.2rem;
      }

      .mvv-player-name {
        font-size: 1.24rem;
        line-height: 1.15;
        font-weight: 900;
        color: var(--mvv-deep) !important;
        margin-bottom: 0.45rem;
      }

      .mvv-player-status {
        font-size: 0.98rem;
        line-height: 1.3;
        font-weight: 700;
        color: var(--mvv-deep) !important;
      }

      .mvv-status-open {
        color: var(--mvv-red) !important;
        font-weight: 900;
      }

      .mvv-status-partial {
        color: #a96f14 !important;
        font-weight: 900;
      }

      .mvv-status-ok {
        color: #1f8a3b !important;
        font-weight: 900;
      }

      .mvv-player-next {
        margin-top: 0.35rem;
        font-size: 0.98rem;
        font-weight: 800;
      }

      .mvv-player-next-open {
        color: var(--mvv-red) !important;
      }

      .mvv-player-next-rpe {
        color: #a96f14 !important;
      }

      .mvv-player-next-ok {
        color: #1f8a3b !important;
      }

      [class*="st-key-tablet_pick_"],
      [class*="st-key-tablet_injury_pick_"] {
        margin-top: -124px;
        margin-bottom: 0.35rem;
        position: relative;
        z-index: 2;
      }

      [class*="st-key-tablet_pick_"] button,
      [class*="st-key-tablet_injury_pick_"] button {
        min-height: 124px !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: transparent !important;
        font-size: 0 !important;
      }

      [class*="st-key-tablet_pick_"] button:hover,
      [class*="st-key-tablet_pick_"] button:active,
      [class*="st-key-tablet_injury_pick_"] button:hover,
      [class*="st-key-tablet_injury_pick_"] button:active {
        transform: none !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
      }

      [class*="st-key-tablet_pick_"] button p,
      [class*="st-key-tablet_pick_"] button span,
      [class*="st-key-tablet_pick_"] button [data-testid="stMarkdownContainer"],
      [class*="st-key-tablet_injury_pick_"] button p,
      [class*="st-key-tablet_injury_pick_"] button span,
      [class*="st-key-tablet_injury_pick_"] button [data-testid="stMarkdownContainer"] {
        opacity: 0 !important;
        font-size: 0 !important;
        margin: 0 !important;
      }

      [class*="st-key-tablet_bulk_back"] button,
      [class*="st-key-tablet_injury_back"] button {
        min-height: 3.35rem !important;
        border-radius: 18px !important;
        font-size: 0.98rem !important;
      }

      [class*="st-key-tablet_day_rpe_mode_"] [data-testid="stRadio"] {
        margin: 0.15rem 0 1rem;
      }

      [class*="st-key-tablet_day_rpe_mode_"] [data-testid="stRadio"] div[role="radiogroup"] {
        width: 100%;
        gap: 0.45rem;
      }

      [class*="st-key-tablet_day_rpe_mode_"] [data-testid="stRadio"] div[role="radiogroup"] label {
        min-width: 8.2rem;
        min-height: 4.1rem;
        padding: 0.65rem 1rem;
        border-radius: 20px;
        gap: 0.5rem;
      }

      [class*="st-key-tablet_day_rpe_mode_"] [data-testid="stRadio"] div[role="radiogroup"] label > div:last-of-type {
        justify-content: center;
      }

      [class*="st-key-tablet_open_bulk_rpe"] button,
      [class*="st-key-tablet_open_injury"] button {
        min-height: 4.1rem !important;
        border-radius: 20px !important;
        font-size: 0.98rem !important;
      }
      @media (max-width: 768px) {
        .block-container { padding-left: 0.75rem; padding-right: 0.75rem; }
        .tablet-hero { border-radius: 20px; padding: 1rem; }
        .mvv-logo-wrap { width: 70px; height: 70px; min-width: 70px; border-radius: 18px; }
        div.stButton > button { min-height: 4.8rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def amsterdam_today():
    return datetime.now(ZoneInfo("Europe/Amsterdam")).date()


def cookie_mgr():
    if "_tablet_cookie_mgr" not in st.session_state:
        st.session_state["_tablet_cookie_mgr"] = stx.CookieManager(key="tablet_cookie_mgr")
    return st.session_state["_tablet_cookie_mgr"]


def logo_html() -> str:
    """Use an MVV logo file when present; otherwise show a clean MVV fallback mark."""
    candidates = [
        ROOT_DIR / "Assets" / "Afbeeldingen" / "Team_Logos" / "MVV Maastricht.png",
        ROOT_DIR / "Assets" / "Afbeeldingen" / "Team_Logos" / "MVV Maastricht.jpg",
        ROOT_DIR / "assets" / "mvv-logo.png",
        ROOT_DIR / "assets" / "mvv_logo.png",
        ROOT_DIR / "assets" / "mvv.png",
        THIS_DIR / "assets" / "mvv-logo.png",
        THIS_DIR / "assets" / "mvv_logo.png",
        THIS_DIR / "assets" / "mvv.png",
    ]
    for path in candidates:
        if path.exists():
            encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
            return f'<img src="data:image/png;base64,{encoded}" alt="MVV Maastricht logo" />'
    return '<div class="mvv-logo-fallback">MVV</div>'


def injury_body_image_src() -> str:
    image_path = TABLET_ASSETS_DIR / "injury-body.png"
    if not image_path.exists():
        return ""
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def render_hero(title: str, subtitle: str, kicker: str = CLUB_NAME) -> None:
    st.markdown(
        f"""
        <div class="tablet-hero">
          <div class="mvv-logo-wrap">{logo_html()}</div>
          <div class="tablet-hero-content">
            <div class="tablet-hero-kicker">{html.escape(kicker)}</div>
            <div class="tablet-hero-title">{html.escape(title)}</div>
            <div class="tablet-hero-subtitle">{html.escape(subtitle)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(label: str, completed: int, total: int, tone: str) -> None:
    progress = 0 if total <= 0 else max(0, min(100, round((completed / total) * 100)))
    st.markdown(
        f"""
        <div class="mvv-kpi-card mvv-kpi-card-{html.escape(tone)}">
          <div class="mvv-kpi-head">
            <div class="mvv-kpi-label">{html.escape(str(label))}</div>
            <div class="mvv-kpi-note">Ingevuld<br>vandaag</div>
          </div>
          <div class="mvv-kpi-value">{completed}/{total}</div>
          <div class="mvv-kpi-progress"><span style="width: {progress}%;"></span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_mini_stat_card(label: str, value: str, note: str = "") -> None:
    note_html = (
        f'<div class="mvv-mini-stat-note">{html.escape(str(note))}</div>'
        if str(note).strip()
        else ""
    )
    st.markdown(
        f'<div class="mvv-mini-stat"><div class="mvv-mini-stat-label">{html.escape(str(label))}</div><div class="mvv-mini-stat-value">{html.escape(str(value))}</div>{note_html}</div>',
        unsafe_allow_html=True,
    )


def render_player_pick_card(player_name: str, wellness_state: str, rpe_state: str, next_step: str) -> None:
    wellness_class = "mvv-status-ok" if str(wellness_state).upper() == "OK" else "mvv-status-open"
    rpe_state_text = str(rpe_state).strip()
    if rpe_state_text.upper() == "OK":
        rpe_class = "mvv-status-ok"
    elif "/" in rpe_state_text:
        rpe_class = "mvv-status-partial"
    else:
        rpe_class = "mvv-status-open"
    next_step_lower = str(next_step).lower()
    if next_step_lower.startswith("controleer"):
        next_class = "mvv-player-next-ok"
    elif next_step_lower.startswith("open rpe"):
        next_class = "mvv-player-next-rpe"
    else:
        next_class = "mvv-player-next-open"
    st.markdown(
        (
            f'<div class="mvv-player-card">'
            f'<div class="mvv-player-name">{html.escape(player_name)}</div>'
            f'<div class="mvv-player-status">Wellness: <span class="{wellness_class}">{html.escape(wellness_state)}</span> | RPE: <span class="{rpe_class}">{html.escape(rpe_state)}</span></div>'
            f'<div class="mvv-player-next {next_class}">{html.escape(next_step)}</div>'
            f'</div>'
        ),
        unsafe_allow_html=True,
    )


def render_injury_pick_card(player_name: str, selected: bool = False) -> None:
    next_class = "mvv-player-next-ok" if selected else "mvv-player-next-open"
    next_text = "Geselecteerd" if selected else "Open blessureformulier"
    subtitle = "Blessuremelding"
    st.markdown(
        (
            f'<div class="mvv-player-card">'
            f'<div class="mvv-player-name">{html.escape(player_name)}</div>'
            f'<div class="mvv-player-status">{html.escape(subtitle)}</div>'
            f'<div class="mvv-player-next {next_class}">{html.escape(next_text)}</div>'
            f'</div>'
        ),
        unsafe_allow_html=True,
    )


def injury_location_label(value: str) -> str:
    return INJURY_LOCATION_LABELS.get(str(value), str(value or "Geen"))


def render_injury_body_selector(selected_location: str) -> str:
    st.markdown('<div class="mvv-toggle-choice-title">Locatie kiezen</div>', unsafe_allow_html=True)
    body_src = injury_body_image_src()
    if not body_src:
        st.warning("Lichaamsafbeelding ontbreekt.")
        return str(selected_location or "None")

    component_value = INJURY_BODY_SELECTOR_COMPONENT(
        imageSrc=body_src,
        markers=INJURY_BODY_IMAGE_MARKERS,
        labels=INJURY_LOCATION_LABELS,
        value=str(selected_location or "None"),
        key="tablet_injury_body_selector",
        default=str(selected_location or "None"),
    )
    if str(component_value or "").strip() in INJURY_LOCATION_OPTIONS:
        return str(component_value)
    return str(selected_location or "None")

def render_form_nav_cards(has_wellness: bool, has_rpe: bool, active_form: str) -> str:
    wellness_status = "ok" if has_wellness else "open"
    rpe_status = "ok" if has_rpe else "open"

    cols = st.columns(2)
    with cols[0]:
        state = "active" if active_form == "Wellness" else "inactive"
        key = f"tablet_nav_wellness_{wellness_status}_{state}"
        if st.button("wellness", key=key, use_container_width=True):
            st.session_state["tablet_active_form"] = "Wellness"
            st.rerun()
    with cols[1]:
        state = "active" if active_form == "RPE" else "inactive"
        key = f"tablet_nav_rpe_{rpe_status}_{state}"
        if st.button("rpe", key=key, use_container_width=True):
            st.session_state["tablet_active_form"] = "RPE"
            st.rerun()

    return st.session_state.get("tablet_active_form", active_form)

def get_tablet_code() -> str:
    for key in ("TABLET_SHARED_CODE", "TABLET_CODE", "KIOSK_CODE"):
        value = str(st.secrets.get(key, "") or "").strip()
        if value:
            return value
    return ""


def get_service_key() -> str:
    for key in ("SUPABASE_SECRET_KEY", "SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_SERVICE_KEY", "SUPABASE_SERVICE_ROLE"):
        value = str(st.secrets.get(key, "") or "").strip()
        if value:
            return value
    return ""


def get_tablet_sb():
    url = str(st.secrets.get("SUPABASE_URL", "") or "").strip()
    service_key = get_service_key()
    if not url or not service_key:
        return None

    if "_tablet_sb_client" not in st.session_state or st.session_state.get("_tablet_sb_client") is None:
        st.session_state["_tablet_sb_client"] = create_client(url, service_key)

    return st.session_state["_tablet_sb_client"]




def get_tablet_created_by(player_id: str = "") -> str:
    """
    Tablet/kiosk mode heeft geen ingelogde Supabase-user, terwijl de database
    created_by verplicht maakt. Zet bij voorkeur TABLET_CREATED_BY_USER_ID in
    Streamlit secrets met de UUID van een admin/system-user. Zonder die secret
    gebruikt de app player_id als UUID-fallback.
    """
    for key in (
        "TABLET_CREATED_BY_USER_ID",
        "TABLET_CREATED_BY",
        "CHECKIN_CREATED_BY_USER_ID",
        "DEFAULT_CREATED_BY_USER_ID",
    ):
        value = str(st.secrets.get(key, "") or "").strip()
        if value:
            return value
    return str(player_id or "").strip()


def _session_cache_get(bucket_name: str, key: str, ttl_seconds: int):
    bucket = st.session_state.get(bucket_name) or {}
    cached = bucket.get(key)
    if not cached:
        return None

    fetched_at = float(cached.get("fetched_at", 0) or 0)
    if fetched_at and (time.time() - fetched_at) <= ttl_seconds:
        return cached.get("value")

    updated_bucket = dict(bucket)
    updated_bucket.pop(key, None)
    if updated_bucket:
        st.session_state[bucket_name] = updated_bucket
    else:
        st.session_state.pop(bucket_name, None)
    return None


def _session_cache_set(bucket_name: str, key: str, value) -> None:
    bucket = dict(st.session_state.get(bucket_name) or {})
    bucket[key] = {"fetched_at": time.time(), "value": value}
    st.session_state[bucket_name] = bucket


def _session_cache_clear(bucket_name: str, key: str | None = None) -> None:
    if key is None:
        st.session_state.pop(bucket_name, None)
        return

    bucket = dict(st.session_state.get(bucket_name) or {})
    bucket.pop(key, None)
    if bucket:
        st.session_state[bucket_name] = bucket
    else:
        st.session_state.pop(bucket_name, None)


def _player_day_cache_key(player_id: str, entry_date_iso: str) -> str:
    return f"{entry_date_iso}:{player_id}"


def _second_rpe_mode_key(entry_date_iso: str) -> str:
    return f"tablet_day_rpe_mode_{entry_date_iso.replace('-', '_')}"


def second_rpe_enabled_for_date(entry_date_iso: str) -> bool:
    return bool(st.session_state.get(_second_rpe_mode_key(entry_date_iso), False))


def _execute_upsert_with_fallback(sb, table_name: str, payload: Dict[str, Any], keys: Dict[str, Any]):
    try:
        return sb.table(table_name).upsert(payload, on_conflict=",".join(keys.keys())).execute()
    except Exception:
        existing = sb.table(table_name).select("id")
        for key, value in keys.items():
            existing = existing.eq(key, value)
        rows = existing.limit(1).execute().data or []
        if rows:
            row_id = rows[0].get("id")
            update_payload = dict(payload)
            update_payload.pop("id", None)
            update_payload.pop("created_by", None)
            return sb.table(table_name).update(update_payload).eq("id", row_id).execute()
        return sb.table(table_name).insert(payload).execute()


def save_asrm_tablet(sb, player_id, entry_date, muscle_soreness, fatigue, sleep_quality, stress, mood) -> None:
    entry_date_iso = entry_date.isoformat() if hasattr(entry_date, "isoformat") else str(entry_date)
    now_iso = datetime.now(ZoneInfo("UTC")).isoformat()
    payload = {
        "player_id": str(player_id),
        "entry_date": entry_date_iso,
        "muscle_soreness": int(muscle_soreness),
        "fatigue": int(fatigue),
        "sleep_quality": int(sleep_quality),
        "stress": int(stress),
        "mood": int(mood),
        "created_by": get_tablet_created_by(str(player_id)),
        "updated_at": now_iso,
    }
    _execute_upsert_with_fallback(
        sb,
        "asrm_entries",
        payload,
        {"player_id": str(player_id), "entry_date": entry_date_iso},
    )


def _load_existing_rpe_header(sb, player_id: str, entry_date_iso: str, existing_rpe_entry_id: str | None = None) -> Dict[str, Any]:
    query = sb.table("rpe_entries").select("*")
    if str(existing_rpe_entry_id or "").strip():
        query = query.eq("id", str(existing_rpe_entry_id).strip())
    else:
        query = query.eq("player_id", str(player_id)).eq("entry_date", entry_date_iso)
    rows = query.limit(1).execute().data or []
    return dict(rows[0]) if rows else {}


def _upsert_rpe_header_tablet(
    sb,
    player_id: str,
    entry_date,
    *,
    injury: bool | None = None,
    injury_type: str | None = None,
    injury_pain: int | None = None,
    notes: str | None = None,
    existing_rpe_entry_id: str | None = None,
) -> Dict[str, Any]:
    entry_date_iso = entry_date.isoformat() if hasattr(entry_date, "isoformat") else str(entry_date)
    now_iso = datetime.now(ZoneInfo("UTC")).isoformat()
    existing_header = _load_existing_rpe_header(sb, player_id, entry_date_iso, existing_rpe_entry_id)

    resolved_injury = bool(existing_header.get("injury", False)) if injury is None else bool(injury)
    resolved_injury_type = existing_header.get("injury_type") if injury_type is None else injury_type
    resolved_injury_pain = existing_header.get("injury_pain") if injury_pain is None else injury_pain
    resolved_notes = existing_header.get("notes") if notes is None else (str(notes or "").strip() or None)

    if not resolved_injury:
        resolved_injury_type = None
        resolved_injury_pain = None

    header_payload = {
        "player_id": str(player_id),
        "entry_date": entry_date_iso,
        "injury": resolved_injury,
        "injury_type": resolved_injury_type,
        "injury_pain": resolved_injury_pain,
        "notes": resolved_notes,
        "created_by": get_tablet_created_by(str(player_id)),
        "updated_at": now_iso,
    }

    rpe_entry_id = str(existing_header.get("id") or existing_rpe_entry_id or "").strip() or None

    if rpe_entry_id:
        update_payload = dict(header_payload)
        update_payload.pop("created_by", None)
        sb.table("rpe_entries").update(update_payload).eq("id", rpe_entry_id).execute()
    else:
        inserted = sb.table("rpe_entries").insert(header_payload).execute().data or []
        rpe_entry_id = str(inserted[0].get("id") or "").strip() if inserted else None

    if not rpe_entry_id:
        raise RuntimeError("RPE-header kon niet worden opgeslagen.")

    header_snapshot = dict(header_payload)
    header_snapshot["id"] = rpe_entry_id
    return header_snapshot


def save_rpe_tablet(
    sb,
    player_id: str,
    entry_date,
    sessions: List[Dict[str, int]],
    existing_rpe_entry_id: str | None = None,
) -> str:
    header_snapshot = _upsert_rpe_header_tablet(
        sb,
        player_id=player_id,
        entry_date=entry_date,
        existing_rpe_entry_id=existing_rpe_entry_id,
    )
    rpe_entry_id = str(header_snapshot.get("id") or "").strip()

    if not sessions:
        sb.table("rpe_sessions").delete().eq("rpe_entry_id", rpe_entry_id).execute()
        return str(rpe_entry_id)

    session_payloads: List[Dict[str, Any]] = []
    keep_session_indexes: set[int] = set()
    for session in sessions:
        duration_min = int(session.get("duration_min", 0) or 0)
        rpe_value = int(session.get("rpe", 0) or 0)
        session_index = int(session.get("session_index", 1) or 1)
        keep_session_indexes.add(session_index)
        session_payload = {
            "rpe_entry_id": rpe_entry_id,
            "session_index": session_index,
            "duration_min": duration_min,
            "rpe": rpe_value,
        }
        session_payloads.append(session_payload)

    sb.table("rpe_sessions").upsert(
        session_payloads,
        on_conflict="rpe_entry_id,session_index",
    ).execute()

    for session_index in (1, 2):
        if session_index not in keep_session_indexes:
            sb.table("rpe_sessions").delete().eq("rpe_entry_id", rpe_entry_id).eq("session_index", session_index).execute()

    return str(rpe_entry_id)


def save_injury_tablet(
    sb,
    player_id: str,
    entry_date,
    injury_type: str | None,
    injury_pain: int | None,
    notes: str,
    existing_rpe_entry_id: str | None = None,
) -> Dict[str, Any]:
    return _upsert_rpe_header_tablet(
        sb,
        player_id=player_id,
        entry_date=entry_date,
        injury=True,
        injury_type=injury_type,
        injury_pain=injury_pain,
        notes=notes,
        existing_rpe_entry_id=existing_rpe_entry_id,
    )

def grant_tablet_access() -> None:
    cm = cookie_mgr()
    cm.set(ACCESS_COOKIE_NAME, "1", max_age=ACCESS_COOKIE_SECONDS, key="tablet_access_set")
    st.session_state["tablet_unlocked"] = True
    time.sleep(0.10)
    st.rerun()


def lock_tablet() -> None:
    cm = cookie_mgr()
    cm.set(ACCESS_COOKIE_NAME, "", max_age=1, key="tablet_access_clear")
    for key in (
        "tablet_unlocked",
        "tablet_player_id",
        "tablet_player_name",
        "tablet_active_form",
        "tablet_flash",
        "tablet_bulk_rpe_mode",
        "tablet_injury_mode",
        "tablet_injury_player_id",
        "tablet_injury_loc",
        "tablet_injury_pain",
        "tablet_injury_notes",
    ):
        st.session_state.pop(key, None)
    time.sleep(0.10)
    st.rerun()


def ensure_tablet_access() -> None:
    shared_code = get_tablet_code()
    if not shared_code:
        st.error("Tabletcode ontbreekt in Streamlit secrets. Voeg TABLET_SHARED_CODE toe.")
        st.stop()

    cm = cookie_mgr()
    _ = cm.get(ACCESS_COOKIE_NAME)

    if st.session_state.get("tablet_unlocked"):
        return

    query_code = str(st.query_params.get("code", "") or "").strip()
    if query_code and query_code == shared_code:
        grant_tablet_access()

    cookie_value = str(cm.get(ACCESS_COOKIE_NAME) or "").strip()
    if cookie_value == "1":
        st.session_state["tablet_unlocked"] = True
        return

    render_hero("Tablet toegang", "Voer de teamcode in om de check-in pagina te openen.")

    with st.form("tablet_unlock_form", clear_on_submit=False):
        code_value = st.text_input("Tabletcode", type="password")
        submitted = st.form_submit_button("Open tablet", use_container_width=True)

    if submitted:
        if code_value.strip() == shared_code:
            grant_tablet_access()
        else:
            st.error("Code klopt niet.")

    st.stop()


@st.cache_data(show_spinner=False, ttl=120)
def fetch_active_players(_sb) -> List[Dict[str, str]]:
    try:
        rows = (
            _sb.table("players")
            .select("player_id,full_name")
            .eq("is_active", True)
            .order("full_name")
            .execute()
            .data
            or []
        )
    except Exception:
        rows = []

    out: List[Dict[str, str]] = []
    for row in rows:
        player_id = row.get("player_id")
        full_name = str(row.get("full_name") or "").strip()
        if player_id and full_name:
            out.append({"player_id": str(player_id), "full_name": full_name})
    return out


def get_cached_active_players(sb) -> List[Dict[str, str]]:
    cached = _session_cache_get("_tablet_active_players_cache", "all", TABLET_PLAYER_CACHE_TTL_SECONDS)
    if cached is not None:
        return list(cached)

    players = fetch_active_players(sb)
    _session_cache_set("_tablet_active_players_cache", "all", players)
    return players


@st.cache_data(show_spinner=False, ttl=30)
def fetch_daily_completion(_sb, entry_date_iso: str) -> Dict[str, List[str]]:
    try:
        asrm_rows = (
            _sb.table("asrm_entries")
            .select("player_id")
            .eq("entry_date", entry_date_iso)
            .execute()
            .data
            or []
        )
    except Exception:
        asrm_rows = []

    try:
        rpe_rows = (
            _sb.table("rpe_entries")
            .select("id,player_id")
            .eq("entry_date", entry_date_iso)
            .execute()
            .data
            or []
        )
    except Exception:
        rpe_rows = []

    asrm_ids = sorted({str(row.get("player_id")) for row in asrm_rows if row.get("player_id")})
    entry_map = {
        str(row.get("id")): str(row.get("player_id"))
        for row in rpe_rows
        if row.get("id") and row.get("player_id")
    }

    session_rows: List[Dict[str, Any]] = []
    if entry_map:
        try:
            session_rows = (
                _sb.table("rpe_sessions")
                .select("rpe_entry_id,session_index")
                .in_("rpe_entry_id", list(entry_map.keys()))
                .execute()
                .data
                or []
            )
        except Exception:
            session_rows = []

    sessions_by_entry: Dict[str, set[int]] = {}
    for row in session_rows:
        entry_id = str(row.get("rpe_entry_id") or "").strip()
        if not entry_id:
            continue
        sessions_by_entry.setdefault(entry_id, set()).add(int(row.get("session_index", 0) or 0))

    rpe1_ids: set[str] = set()
    rpe2_ids: set[str] = set()
    for entry_id, player_id in entry_map.items():
        session_indexes = sessions_by_entry.get(entry_id, set())
        if session_indexes:
            rpe1_ids.add(player_id)
        if 2 in session_indexes:
            rpe2_ids.add(player_id)

    return {
        "asrm_ids": asrm_ids,
        "rpe_ids": sorted(rpe1_ids),
        "rpe2_ids": sorted(rpe2_ids),
    }


def get_cached_daily_completion(sb, entry_date_iso: str) -> Dict[str, List[str]]:
    cached = _session_cache_get(
        "_tablet_daily_completion_cache",
        entry_date_iso,
        TABLET_COMPLETION_CACHE_TTL_SECONDS,
    )
    if cached is not None:
        return {
            "asrm_ids": list(cached.get("asrm_ids", [])),
            "rpe_ids": list(cached.get("rpe_ids", [])),
            "rpe2_ids": list(cached.get("rpe2_ids", [])),
        }

    completion = fetch_daily_completion(sb, entry_date_iso)
    _session_cache_set("_tablet_daily_completion_cache", entry_date_iso, completion)
    return completion


def update_cached_daily_completion(
    sb,
    player_id: str,
    entry_date_iso: str,
    *,
    has_asrm: bool | None = None,
    has_rpe: bool | None = None,
    has_rpe2: bool | None = None,
) -> None:
    cached = _session_cache_get(
        "_tablet_daily_completion_cache",
        entry_date_iso,
        TABLET_COMPLETION_CACHE_TTL_SECONDS,
    )
    if cached is None:
        try:
            cached = fetch_daily_completion(sb, entry_date_iso)
        except Exception:
            cached = {"asrm_ids": [], "rpe_ids": [], "rpe2_ids": []}
    asrm_ids = set(cached.get("asrm_ids", []))
    rpe_ids = set(cached.get("rpe_ids", []))
    rpe2_ids = set(cached.get("rpe2_ids", []))

    if has_asrm is True:
        asrm_ids.add(str(player_id))
    elif has_asrm is False:
        asrm_ids.discard(str(player_id))

    if has_rpe is True:
        rpe_ids.add(str(player_id))
    elif has_rpe is False:
        rpe_ids.discard(str(player_id))

    if has_rpe2 is True:
        rpe2_ids.add(str(player_id))
    elif has_rpe2 is False:
        rpe2_ids.discard(str(player_id))

    _session_cache_set(
        "_tablet_daily_completion_cache",
        entry_date_iso,
        {
            "asrm_ids": sorted(asrm_ids),
            "rpe_ids": sorted(rpe_ids),
            "rpe2_ids": sorted(rpe2_ids),
        },
    )


def get_cached_asrm_detail(sb, player_id: str, entry_date) -> Dict[str, Any]:
    entry_date_iso = entry_date.isoformat() if hasattr(entry_date, "isoformat") else str(entry_date)
    cache_key = _player_day_cache_key(str(player_id), entry_date_iso)
    cached = _session_cache_get("_tablet_asrm_detail_cache", cache_key, TABLET_FORM_CACHE_TTL_SECONDS)
    if cached is not None:
        return dict(cached)

    existing_asrm = load_asrm(sb, player_id, entry_date) or {}
    _session_cache_set("_tablet_asrm_detail_cache", cache_key, existing_asrm)
    return existing_asrm


def set_cached_asrm_detail(player_id: str, entry_date, payload: Dict[str, Any]) -> None:
    entry_date_iso = entry_date.isoformat() if hasattr(entry_date, "isoformat") else str(entry_date)
    cache_key = _player_day_cache_key(str(player_id), entry_date_iso)
    _session_cache_set("_tablet_asrm_detail_cache", cache_key, payload)


def get_cached_rpe_detail(sb, player_id: str, entry_date) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    entry_date_iso = entry_date.isoformat() if hasattr(entry_date, "isoformat") else str(entry_date)
    cache_key = _player_day_cache_key(str(player_id), entry_date_iso)
    cached = _session_cache_get("_tablet_rpe_detail_cache", cache_key, TABLET_FORM_CACHE_TTL_SECONDS)
    if cached is not None:
        return dict(cached.get("header", {})), list(cached.get("sessions", []))

    rpe_header, rpe_sessions = load_rpe(sb, player_id, entry_date)
    snapshot = {"header": rpe_header or {}, "sessions": rpe_sessions or []}
    _session_cache_set("_tablet_rpe_detail_cache", cache_key, snapshot)
    return dict(snapshot["header"]), list(snapshot["sessions"])


def set_cached_rpe_detail(player_id: str, entry_date, header: Dict[str, Any], sessions: List[Dict[str, Any]]) -> None:
    entry_date_iso = entry_date.isoformat() if hasattr(entry_date, "isoformat") else str(entry_date)
    cache_key = _player_day_cache_key(str(player_id), entry_date_iso)
    _session_cache_set(
        "_tablet_rpe_detail_cache",
        cache_key,
        {"header": header, "sessions": sessions},
    )


def clear_daily_cache(entry_date_iso: str | None = None) -> None:
    fetch_daily_completion.clear()
    if entry_date_iso:
        _session_cache_clear("_tablet_daily_completion_cache", entry_date_iso)
    else:
        _session_cache_clear("_tablet_daily_completion_cache")


def clear_selected_player_state(player_id: str | None = None) -> None:
    keys_to_clear = [
        "tablet_player_id",
        "tablet_player_name",
        "tablet_player_has_wellness",
        "tablet_player_has_rpe",
        "tablet_active_form",
    ]

    if player_id:
        keys_to_clear.extend(
            [
                f"tablet_asrm_ms_{player_id}",
                f"tablet_asrm_fat_{player_id}",
                f"tablet_asrm_sleep_{player_id}",
                f"tablet_asrm_stress_{player_id}",
                f"tablet_asrm_mood_{player_id}",
                f"tablet_rpe_stage_{player_id}",
                f"tablet_rpe_s1_dur_{player_id}",
                f"tablet_rpe_s1_dur_choice_{player_id}",
                f"tablet_rpe_s1_rpe_{player_id}",
                f"tablet_rpe_s2_dur_{player_id}",
                f"tablet_rpe_s2_dur_choice_{player_id}",
                f"tablet_rpe_s2_rpe_{player_id}",
            ]
        )

    for key in keys_to_clear:
        st.session_state.pop(key, None)


def clear_injury_report_state() -> None:
    for key in (
        "tablet_injury_mode",
        "tablet_injury_player_id",
        "tablet_injury_loc",
        "tablet_injury_pain",
        "tablet_injury_notes",
    ):
        st.session_state.pop(key, None)


def show_flash() -> None:
    flash = st.session_state.pop("tablet_flash", None)
    if flash:
        st.success(str(flash))


def render_top_actions(show_back: bool = False) -> None:
    # Geen navigatieknoppen bovenaan de tabletpagina's.
    return

def render_player_picker(sb) -> None:
    entry_date = amsterdam_today()
    entry_date_iso = entry_date.isoformat()
    players = get_cached_active_players(sb)
    if not players:
        st.warning("Geen actieve spelers gevonden.")
        return

    completion = get_cached_daily_completion(sb, entry_date_iso)
    asrm_ids = set(completion.get("asrm_ids", []))
    rpe1_ids = set(completion.get("rpe_ids", []))
    rpe2_ids = set(completion.get("rpe2_ids", []))

    render_top_actions(show_back=False)
    show_flash()

    render_hero(
        APP_TITLE,
        f"Selecteer een speler voor de invoer van vandaag ({entry_date.strftime('%d-%m-%Y')}).",
    )

    second_rpe_key = _second_rpe_mode_key(entry_date_iso)
    if second_rpe_key not in st.session_state:
        st.session_state[second_rpe_key] = False
    top_cols = st.columns([1.25, 1, 1], gap="small")
    with top_cols[0]:
        st.markdown('<div class="mvv-toggle-choice-title">RPE modus</div>', unsafe_allow_html=True)
        second_rpe_enabled = st.radio(
            "RPE modus",
            options=[False, True],
            index=1 if bool(st.session_state[second_rpe_key]) else 0,
            format_func=lambda value: "2 RPE" if value else "1 RPE",
            horizontal=True,
            label_visibility="collapsed",
            key=second_rpe_key,
        )

    with top_cols[1]:
        st.markdown('<div class="mvv-inline-control-spacer"></div>', unsafe_allow_html=True)
        if st.button("Groeps-RPE invullen", use_container_width=True, key="tablet_open_bulk_rpe"):
            clear_selected_player_state()
            clear_injury_report_state()
            st.session_state["tablet_bulk_rpe_mode"] = True
            st.rerun()
    with top_cols[2]:
        st.markdown('<div class="mvv-inline-control-spacer"></div>', unsafe_allow_html=True)
        if st.button("Injury melden", use_container_width=True, key="tablet_open_injury"):
            clear_selected_player_state()
            clear_injury_report_state()
            st.session_state["tablet_bulk_rpe_mode"] = False
            st.session_state["tablet_injury_mode"] = True
            st.rerun()

    total_players = len(players)
    wellness_completed = sum(1 for player in players if player["player_id"] in asrm_ids)
    completed_rpe_ids = rpe2_ids if second_rpe_enabled else rpe1_ids
    rpe_completed = sum(1 for player in players if player["player_id"] in completed_rpe_ids)

    stat_1, stat_2 = st.columns(2)
    with stat_1:
        render_kpi_card("Wellness", wellness_completed, total_players, tone="wellness")
    with stat_2:
        render_kpi_card("RPE", rpe_completed, total_players, tone="rpe")

    cols = st.columns(3)
    for idx, player in enumerate(players):
        player_id = player["player_id"]
        player_name = player["full_name"]
        wellness_done = player_id in asrm_ids
        rpe1_done = player_id in rpe1_ids
        rpe2_done = player_id in rpe2_ids
        wellness_state = "OK" if wellness_done else "OPEN"
        if second_rpe_enabled:
            if rpe2_done:
                rpe_state = "OK"
            elif rpe1_done:
                rpe_state = "1/2"
            else:
                rpe_state = "OPEN"
        else:
            rpe_state = "OK" if rpe1_done else "OPEN"

        rpe_done = rpe2_done if second_rpe_enabled else rpe1_done
        if not wellness_done:
            next_step = "Open wellness"
        elif second_rpe_enabled and not rpe1_done:
            next_step = "Open RPE 1"
        elif second_rpe_enabled and not rpe2_done:
            next_step = "Open RPE 2"
        elif not second_rpe_enabled and not rpe1_done:
            next_step = "Open RPE"
        else:
            next_step = "Controleer invoer"
        with cols[idx % 3]:
            render_player_pick_card(player_name, wellness_state, rpe_state, next_step)
            if st.button("select_player", use_container_width=True, key=f"tablet_pick_{player_id}"):
                st.session_state["tablet_bulk_rpe_mode"] = False
                st.session_state["tablet_injury_mode"] = False
                st.session_state["tablet_player_id"] = player_id
                st.session_state["tablet_player_name"] = player_name
                st.session_state["tablet_player_has_wellness"] = wellness_done
                st.session_state["tablet_player_has_rpe"] = rpe_done
                # Belangrijk: als wellness vandaag bestaat, start direct op RPE.
                st.session_state["tablet_active_form"] = "RPE" if wellness_done else "Wellness"
                st.rerun()


def render_bulk_rpe_page(sb) -> None:
    entry_date = amsterdam_today()
    entry_date_iso = entry_date.isoformat()
    players = get_cached_active_players(sb)
    if not players:
        st.warning("Geen actieve spelers gevonden.")
        return

    completion = get_cached_daily_completion(sb, entry_date_iso)
    asrm_ids = set(completion.get("asrm_ids", []))
    rpe1_ids = set(completion.get("rpe_ids", []))
    ready_players = [
        player
        for player in players
        if str(player.get("player_id")) in asrm_ids and str(player.get("player_id")) not in rpe1_ids
    ]
    render_top_actions(show_back=False)
    show_flash()

    render_hero(
        "RPE groepsinvoer",
        f"Vul meerdere spelers tegelijk in ({entry_date.strftime('%d-%m-%Y')}).",
        kicker=f"{CLUB_NAME} · snelle RPE",
    )

    if st.button("Terug naar spelersoverzicht", use_container_width=True, key="tablet_bulk_back"):
        st.session_state["tablet_bulk_rpe_mode"] = False
        st.rerun()

    if not ready_players:
        st.info("Geen spelers om RPE in te vullen.")
        return

    shared_duration_key = "tablet_bulk_rpe_dur_shared"
    with st.form("tablet_bulk_rpe_form", clear_on_submit=False):
        st.markdown('<div class="mvv-bulk-shared-title">Minuten voor alle spelers</div>', unsafe_allow_html=True)
        shared_duration = st.radio(
            "Minuten voor alle spelers",
            options=RPE_DURATION_OPTIONS,
            index=RPE_DURATION_OPTIONS.index(RPE_BULK_DURATION_DEFAULT),
            format_func=lambda value: str(value),
            horizontal=True,
            label_visibility="collapsed",
            key=shared_duration_key,
        )

        for player in ready_players:
            player_id = str(player["player_id"])
            player_name = str(player["full_name"])
            row_name_col, row_slider_col = st.columns([1.3, 3.2], gap="large")
            with row_name_col:
                st.markdown(
                    f'<div class="mvv-bulk-player-name">{html.escape(player_name)}</div>'
                    '<div class="mvv-bulk-player-note">RPE invoer</div>',
                    unsafe_allow_html=True,
                )
            with row_slider_col:
                st.slider(
                    f"RPE (1-10) {player_name}",
                    1,
                    10,
                    value=5,
                    key=f"tablet_bulk_rpe_rpe_{player_id}",
                )
            st.divider()

        bulk_submit = st.form_submit_button("RPE voor alle zichtbare spelers opslaan", use_container_width=True)

    if bulk_submit:
        try:
            saved_count = 0
            failed_players: List[str] = []
            shared_duration_value = int(shared_duration)
            for player in ready_players:
                player_id = str(player["player_id"])
                player_name = str(player["full_name"])
                rpe_value = int(st.session_state.get(f"tablet_bulk_rpe_rpe_{player_id}", 5) or 5)
                try:
                    sessions_payload = [
                        {
                            "session_index": 1,
                            "duration_min": shared_duration_value,
                            "rpe": rpe_value,
                        }
                    ]
                    existing_header, _ = get_cached_rpe_detail(sb, player_id, entry_date)
                    saved_rpe_entry_id = save_rpe_tablet(
                        sb,
                        player_id=player_id,
                        entry_date=entry_date,
                        sessions=sessions_payload,
                        existing_rpe_entry_id=str(existing_header.get("id") or "").strip() or None,
                    )
                    set_cached_rpe_detail(
                        player_id,
                        entry_date,
                        {
                            "id": saved_rpe_entry_id,
                            "player_id": player_id,
                            "entry_date": entry_date_iso,
                            "injury": bool(existing_header.get("injury", False)),
                            "injury_type": existing_header.get("injury_type"),
                            "injury_pain": existing_header.get("injury_pain"),
                            "notes": str(existing_header.get("notes") or ""),
                        },
                        sessions_payload,
                    )
                    update_cached_daily_completion(sb, player_id, entry_date_iso, has_rpe=True)
                    saved_count += 1
                except Exception as player_exc:
                    failed_players.append(f"{player_name} ({player_exc})")

            if failed_players:
                if saved_count:
                    st.warning(
                        "RPE opgeslagen voor "
                        f"{saved_count} spelers, maar niet voor: {', '.join(failed_players)}"
                    )
                else:
                    st.error(f"Opslaan faalde voor: {', '.join(failed_players)}")
            else:
                st.session_state["tablet_flash"] = f"RPE opgeslagen voor {saved_count} spelers."
                st.rerun()
        except Exception as exc:
            st.error(f"Opslaan faalde: {exc}")


def _session_value(rpe_sessions: List[Dict[str, Any]], idx: int, key: str, default: int) -> int:
    hit = next((row for row in rpe_sessions if int(row.get("session_index", 0) or 0) == idx), None)
    if not hit:
        return default
    value = hit.get(key)
    return int(value) if value is not None else default


def _existing_session_payload(rpe_sessions: List[Dict[str, Any]], idx: int) -> Dict[str, int] | None:
    duration_min = _session_value(rpe_sessions, idx, "duration_min", 0)
    if int(duration_min) <= 0:
        return None
    return {
        "session_index": int(idx),
        "duration_min": int(duration_min),
        "rpe": _session_value(rpe_sessions, idx, "rpe", 5),
    }


def _normalize_duration_choice(value: int) -> int:
    try:
        duration = int(value)
    except (TypeError, ValueError):
        duration = 0
    if duration in RPE_DURATION_OPTIONS:
        return duration
    if duration <= 0:
        return RPE_DURATION_OPTIONS[0]
    return min(RPE_DURATION_OPTIONS, key=lambda option: abs(option - duration))


def render_injury_report_page(sb) -> None:
    entry_date = amsterdam_today()
    entry_date_iso = entry_date.isoformat()
    players = get_cached_active_players(sb)
    if not players:
        st.warning("Geen actieve spelers gevonden.")
        return

    render_top_actions(show_back=False)
    show_flash()

    player_lookup = {str(player["player_id"]): str(player["full_name"]) for player in players}
    selected_player_id = str(st.session_state.get("tablet_injury_player_id") or "").strip()
    if selected_player_id not in player_lookup:
        selected_player_id = ""

    if not selected_player_id:
        render_hero(
            "Injury melden",
            f"Selecteer een speler en meld de blessure van vandaag ({entry_date.strftime('%d-%m-%Y')}).",
            kicker=f"{CLUB_NAME} - blessuremelding",
        )

        if st.button("Terug naar spelersoverzicht", use_container_width=True, key="tablet_injury_back"):
            clear_injury_report_state()
            st.rerun()

        st.markdown('<div class="mvv-toggle-choice-title">Speler kiezen</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        for idx, player in enumerate(players):
            player_id = str(player["player_id"])
            player_name = str(player["full_name"])
            with cols[idx % 3]:
                render_injury_pick_card(player_name, selected=False)
                if st.button("select_injury_player", use_container_width=True, key=f"tablet_injury_pick_{player_id}"):
                    st.session_state["tablet_injury_player_id"] = player_id
                    st.session_state.pop("tablet_injury_loc", None)
                    st.session_state.pop("tablet_injury_pain", None)
                    st.session_state.pop("tablet_injury_notes", None)
                    st.rerun()
        return

    player_name = player_lookup[selected_player_id]
    render_hero(
        player_name,
        f"Blessuremelding voor vandaag ({entry_date.strftime('%d-%m-%Y')}).",
        kicker=f"{CLUB_NAME} - blessuremelding",
    )

    nav_cols = st.columns(2)
    with nav_cols[0]:
        if st.button("Terug naar spelersoverzicht", use_container_width=True, key="tablet_injury_back"):
            clear_injury_report_state()
            st.rerun()
    with nav_cols[1]:
        if st.button("Andere speler kiezen", use_container_width=True, key="tablet_injury_reset_player"):
            st.session_state.pop("tablet_injury_player_id", None)
            st.session_state.pop("tablet_injury_loc", None)
            st.session_state.pop("tablet_injury_pain", None)
            st.session_state.pop("tablet_injury_notes", None)
            st.rerun()

    rpe_header, rpe_sessions = get_cached_rpe_detail(sb, selected_player_id, entry_date)
    existing_loc = str(rpe_header.get("injury_type") or "None").strip() or "None"
    if existing_loc not in INJURY_LOCATION_OPTIONS:
        existing_loc = "Other"
    existing_pain = int(rpe_header.get("injury_pain", 0) or 0)
    existing_notes = str(rpe_header.get("notes") or "")
    selected_injury_loc = str(st.session_state.get("tablet_injury_loc") or existing_loc or "None")
    if selected_injury_loc not in INJURY_LOCATION_OPTIONS:
        selected_injury_loc = "Other"

    st.markdown(
        f"""
        <div class="mvv-section-card">
          <div class="mvv-form-kicker">Blessure vandaag</div>
          <div class="mvv-form-title">{html.escape(player_name)}</div>
          <div class="mvv-form-subtitle">Vul locatie, pijn en een korte opmerking in.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_injury_loc = render_injury_body_selector(selected_injury_loc)
    st.session_state["tablet_injury_loc"] = selected_injury_loc
    st.markdown(
        f'<div class="mvv-load-pill">Gekozen locatie: <strong>{html.escape(injury_location_label(selected_injury_loc))}</strong></div>',
        unsafe_allow_html=True,
    )

    with st.form("tablet_injury_form", clear_on_submit=False):
        injury_pain = st.slider(
            "Pijn (0-10)",
            0,
            10,
            value=existing_pain,
            key="tablet_injury_pain",
        )
        notes = st.text_area(
            "Opmerking",
            value=existing_notes,
            key="tablet_injury_notes",
            height=140,
        )
        injury_submit = st.form_submit_button("Injury opslaan", use_container_width=True)

    if injury_submit:
        if selected_injury_loc == "None":
            st.error("Kies een blessurelocatie.")
            return

        try:
            saved_header = save_injury_tablet(
                sb,
                player_id=selected_player_id,
                entry_date=entry_date,
                injury_type=selected_injury_loc,
                injury_pain=int(injury_pain),
                notes=notes,
                existing_rpe_entry_id=str(rpe_header.get("id") or "").strip() or None,
            )
            set_cached_rpe_detail(
                selected_player_id,
                entry_date,
                {
                    "id": str(saved_header.get("id") or ""),
                    "player_id": selected_player_id,
                    "entry_date": entry_date_iso,
                    "injury": True,
                    "injury_type": selected_injury_loc,
                    "injury_pain": int(injury_pain),
                    "notes": str(notes or ""),
                },
                rpe_sessions,
            )
            st.session_state["tablet_flash"] = f"Injury opgeslagen voor {player_name}."
            clear_injury_report_state()
            st.rerun()
        except Exception as exc:
            st.error(f"Opslaan faalde: {exc}")


def render_player_forms(sb, player_id: str, player_name: str) -> None:
    entry_date = amsterdam_today()
    entry_date_iso = entry_date.isoformat()
    second_rpe_enabled = second_rpe_enabled_for_date(entry_date_iso)
    if "tablet_player_has_wellness" not in st.session_state or "tablet_player_has_rpe" not in st.session_state:
        completion = get_cached_daily_completion(sb, entry_date_iso)
        asrm_ids = set(completion.get("asrm_ids", []))
        rpe1_ids = set(completion.get("rpe_ids", []))
        rpe2_ids = set(completion.get("rpe2_ids", []))
        st.session_state["tablet_player_has_wellness"] = str(player_id) in asrm_ids
        st.session_state["tablet_player_has_rpe"] = str(player_id) in (rpe2_ids if second_rpe_enabled else rpe1_ids)

    has_wellness = bool(st.session_state.get("tablet_player_has_wellness", False))
    has_rpe = bool(st.session_state.get("tablet_player_has_rpe", False))

    if "tablet_active_form" not in st.session_state:
        st.session_state["tablet_active_form"] = "RPE" if has_wellness else "Wellness"

    render_top_actions(show_back=True)
    show_flash()

    render_hero(
        player_name,
        f"Invoer voor vandaag ({entry_date.strftime('%d-%m-%Y')}).",
        kicker=f"{CLUB_NAME} · speler check-in",
    )

    default_form = st.session_state.get("tablet_active_form", "RPE" if has_wellness else "Wellness")
    if default_form not in ["Wellness", "RPE"]:
        default_form = "Wellness"

    active_form = render_form_nav_cards(has_wellness, has_rpe, default_form)
    st.session_state["tablet_active_form"] = active_form

    if active_form == "Wellness":
        existing_asrm = get_cached_asrm_detail(sb, player_id, entry_date) if has_wellness else {}
        with st.form(f"tablet_asrm_form_{player_id}", clear_on_submit=False):
            _legend_asrm()

            ms = st.slider(
                "Muscle soreness (1-10)",
                1,
                10,
                value=int(existing_asrm.get("muscle_soreness", 5)),
                key=f"tablet_asrm_ms_{player_id}",
            )
            fat = st.slider(
                "Fatigue (1-10)",
                1,
                10,
                value=int(existing_asrm.get("fatigue", 5)),
                key=f"tablet_asrm_fat_{player_id}",
            )
            sleep = st.slider(
                "Sleep quality (1-10)",
                1,
                10,
                value=int(existing_asrm.get("sleep_quality", 5)),
                key=f"tablet_asrm_sleep_{player_id}",
            )
            stress = st.slider(
                "Stress (1-10)",
                1,
                10,
                value=int(existing_asrm.get("stress", 5)),
                key=f"tablet_asrm_stress_{player_id}",
            )
            mood = st.slider(
                "Mood (1-10)",
                1,
                10,
                value=int(existing_asrm.get("mood", 5)),
                key=f"tablet_asrm_mood_{player_id}",
            )

            asrm_submit = st.form_submit_button("Wellness opslaan", use_container_width=True)

        if asrm_submit:
            try:
                save_asrm_tablet(sb, player_id, entry_date, ms, fat, sleep, stress, mood)
                st.session_state["tablet_player_has_wellness"] = True
                set_cached_asrm_detail(
                    player_id,
                    entry_date,
                    {
                        "player_id": str(player_id),
                        "entry_date": entry_date_iso,
                        "muscle_soreness": int(ms),
                        "fatigue": int(fat),
                        "sleep_quality": int(sleep),
                        "stress": int(stress),
                        "mood": int(mood),
                    },
                )
                update_cached_daily_completion(sb, player_id, entry_date_iso, has_asrm=True)
                st.session_state["tablet_flash"] = f"Wellness opgeslagen voor {player_name}."
                clear_selected_player_state(player_id)
                st.rerun()
            except Exception as exc:
                st.error(f"Opslaan faalde: {exc}")

    if active_form == "RPE":
        rpe_header, rpe_sessions = get_cached_rpe_detail(sb, player_id, entry_date)
        has_rpe1 = any(int(row.get("session_index", 0) or 0) == 1 for row in rpe_sessions)
        has_rpe2 = any(int(row.get("session_index", 0) or 0) == 2 for row in rpe_sessions)
        st.session_state["tablet_player_has_rpe"] = has_rpe2 if second_rpe_enabled else has_rpe1

        s1_default_dur = _session_value(rpe_sessions, 1, "duration_min", 0)
        s1_default_rpe = _session_value(rpe_sessions, 1, "rpe", 5)
        s2_default_dur = _session_value(rpe_sessions, 2, "duration_min", 0)
        s2_default_rpe = _session_value(rpe_sessions, 2, "rpe", 5)
        stage_key = f"tablet_rpe_stage_{player_id}"

        default_stage = 2 if second_rpe_enabled and has_rpe1 and not has_rpe2 else 1
        if stage_key not in st.session_state:
            st.session_state[stage_key] = default_stage
        current_rpe_stage = int(st.session_state.get(stage_key, default_stage) or default_stage)
        if not second_rpe_enabled or not has_rpe1:
            current_rpe_stage = 1
            st.session_state[stage_key] = 1
        elif second_rpe_enabled and has_rpe1 and not has_rpe2:
            current_rpe_stage = 2
            st.session_state[stage_key] = 2
        elif current_rpe_stage not in (1, 2):
            current_rpe_stage = 1
            st.session_state[stage_key] = 1

        current_duration_default = s2_default_dur if current_rpe_stage == 2 else s1_default_dur
        current_rpe_default = s2_default_rpe if current_rpe_stage == 2 else s1_default_rpe
        current_duration_choice = _normalize_duration_choice(current_duration_default)
        duration_key = f"tablet_rpe_s{current_rpe_stage}_dur_choice_{player_id}"
        rpe_key = f"tablet_rpe_s{current_rpe_stage}_rpe_{player_id}"
        session_title = f"RPE {current_rpe_stage}"

        with st.form(f"tablet_rpe_form_{player_id}", clear_on_submit=False):
            _legend_rpe()

            st.markdown(
                f"""
                <div class="mvv-session-title">{html.escape(session_title)}</div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown('<div class="mvv-duration-title">Duration (min)</div>', unsafe_allow_html=True)
            current_dur = st.radio(
                "Duration (min)",
                options=RPE_DURATION_OPTIONS,
                index=RPE_DURATION_OPTIONS.index(current_duration_choice),
                format_func=lambda value: str(value),
                horizontal=True,
                label_visibility="collapsed",
                key=duration_key,
            )
            current_rpe_value = st.slider(
                "RPE (1-10)",
                1,
                10,
                value=current_rpe_default,
                key=rpe_key,
            )

            rpe_submit = st.form_submit_button(f"{session_title} opslaan", use_container_width=True)

        if rpe_submit:
            try:
                sessions_payload: List[Dict[str, int]] = []
                current_session_payload = {
                    "session_index": int(current_rpe_stage),
                    "duration_min": int(current_dur),
                    "rpe": int(current_rpe_value),
                }

                if current_rpe_stage == 1:
                    sessions_payload.append(current_session_payload)
                    existing_session_2 = _existing_session_payload(rpe_sessions, 2)
                    if existing_session_2:
                        sessions_payload.append(existing_session_2)
                else:
                    existing_session_1 = _existing_session_payload(rpe_sessions, 1)
                    if existing_session_1:
                        sessions_payload.append(existing_session_1)
                    sessions_payload.append(current_session_payload)

                sessions_payload = sorted(sessions_payload, key=lambda row: int(row["session_index"]))

                saved_rpe_entry_id = save_rpe_tablet(
                    sb,
                    player_id=player_id,
                    entry_date=entry_date,
                    sessions=sessions_payload,
                    existing_rpe_entry_id=str(rpe_header.get("id") or "").strip() or None,
                )
                current_header = {
                    "id": saved_rpe_entry_id,
                    "player_id": str(player_id),
                    "entry_date": entry_date_iso,
                    "injury": bool(rpe_header.get("injury", False)),
                    "injury_type": rpe_header.get("injury_type"),
                    "injury_pain": rpe_header.get("injury_pain"),
                    "notes": str(rpe_header.get("notes") or ""),
                }
                set_cached_rpe_detail(
                    player_id,
                    entry_date,
                    current_header,
                    sessions_payload,
                )
                update_cached_daily_completion(sb, player_id, entry_date_iso, has_rpe=True)

                if current_rpe_stage == 1 and second_rpe_enabled and not has_rpe2:
                    st.session_state["tablet_player_has_rpe"] = False
                    st.session_state[stage_key] = 2
                    st.session_state["tablet_flash"] = f"RPE 1 opgeslagen voor {player_name}. Vul nu RPE 2 in."
                    st.rerun()
                else:
                    if current_rpe_stage == 2:
                        update_cached_daily_completion(sb, player_id, entry_date_iso, has_rpe2=True)
                    st.session_state["tablet_player_has_rpe"] = True
                    st.session_state["tablet_flash"] = f"{session_title} opgeslagen voor {player_name}."
                    clear_selected_player_state(player_id)
                    st.rerun()
            except Exception as exc:
                st.error(f"Opslaan faalde: {exc}")
    if st.button("Klaar / volgende speler", use_container_width=True, key=f"tablet_done_{player_id}"):
        clear_selected_player_state(player_id)
        st.rerun()


def main() -> None:
    ensure_tablet_access()

    sb = get_tablet_sb()
    if sb is None:
        st.error(
            "Supabase service key ontbreekt. Voeg SUPABASE_SERVICE_ROLE_KEY toe in Streamlit secrets "
            "voor deze losse tablet-app."
        )
        st.stop()

    selected_player_id = str(st.session_state.get("tablet_player_id") or "").strip()
    selected_player_name = str(st.session_state.get("tablet_player_name") or "").strip()
    bulk_rpe_mode = bool(st.session_state.get("tablet_bulk_rpe_mode", False))
    injury_mode = bool(st.session_state.get("tablet_injury_mode", False))

    if selected_player_id and selected_player_name:
        render_player_forms(sb, selected_player_id, selected_player_name)
    elif injury_mode:
        render_injury_report_page(sb)
    elif bulk_rpe_mode:
        render_bulk_rpe_page(sb)
    else:
        render_player_picker(sb)


if __name__ == "__main__":
    main()
