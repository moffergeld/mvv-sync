from __future__ import annotations

import base64
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
ASSETS_DIR = ROOT_DIR / "Assets" / "Afbeeldingen"
TEAM_LOGO = ASSETS_DIR / "Team_Logos" / "MVV Maastricht.png"
TEAM_HERO_BG = ASSETS_DIR / "Backgrounds" / "team_page_hero.png"


def build_data_uri(path: Path) -> str:
    if not path.exists():
        return ""

    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "application/octet-stream")
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"
