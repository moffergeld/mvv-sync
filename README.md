# MVV Sync Reports

Deze applicatie bevat nu twee PDF-rapportstijlen die naast elkaar blijven bestaan:

- `legacy`: de bestaande ReportLab-generator, standaard actief
- `html`: een nieuwe HTML/CSS-generator op basis van Jinja2 en WeasyPrint

## Rapportstijl kiezen

In de volgende rapportpagina's is een nieuwe keuze `Rapportstijl` beschikbaar:

- `Week Report`
- `Player Report`

De standaardwaarde blijft `legacy`, zodat bestaande workflows en bestaande aanroepen ongewijzigd blijven werken.

## Router en generatoren

De centrale router staat in:

- `report_generator.py`

De router kiest op basis van `report_style` welke renderer wordt gebruikt:

- `legacy`:
  - `week_report_pdf.py`
  - `player_report_pdf.py`
- `html`:
  - `html_report_generator.py`
  - `templates/week_report.html`
  - `templates/player_report.html`
  - `static/css/report.css`

## Gebruik in code

Voorbeeld weekrapport:

```python
from report_generator import generate_week_report

pdf_bytes = generate_week_report(
    report_style="legacy",
    week_label="2026-W30",
    iso_label="ISO week 2026-W30",
    summary=summary,
    monitoring_summary=monitoring_summary,
    day_table=day_table,
    type_table=type_table,
    player_table=player_table,
    monitoring_day_table=monitoring_day_table,
    notes=notes,
)
```

Voorbeeld met de nieuwe HTML/CSS-stijl:

```python
pdf_bytes = generate_week_report(
    report_style="html",
    week_label="2026-W30",
    iso_label="ISO week 2026-W30",
    summary=summary,
    monitoring_summary=monitoring_summary,
    day_table=day_table,
    type_table=type_table,
    player_table=player_table,
    monitoring_day_table=monitoring_day_table,
    notes=notes,
)
```

Voor spelersrapporten werkt hetzelfde via:

- `generate_player_report(...)`

Wanneer `report_style` niet wordt meegegeven, blijft automatisch `legacy` actief.

## Vormgeving aanpassen

De nieuwe HTML-rapportstijl kan hier worden aangepast:

- Template-opbouw: `templates/week_report.html` en `templates/player_report.html`
- Gedeelde styling: `static/css/report.css`

In de CSS staan ook:

- `@page` voor A4 landscape, marges en paginanummers
- CSS-variabelen voor kleuren, radius, spacing en tekstkleuren
- regels om pagina-afbrekingen in kaarten, tabellen en grafieken te beperken

## Dependencies

Toegevoegd aan `requirements.txt`:

- `jinja2`
- `weasyprint`

Daarnaast blijft de bestaande dependency voor de legacy-PDF bestaan:

- `reportlab`

## WeasyPrint systeembibliotheken

Afhankelijk van de omgeving heeft WeasyPrint extra systeembibliotheken nodig.

Voor lokale Windows-omgevingen kan een extra GTK/Pango/Cairo-runtime nodig zijn.
Voor Linux-deployments is meestal een set renderbibliotheken nodig die door het platform beschikbaar moet zijn.

Als `report_style="html"` een dependencyfout geeft, blijft `report_style="legacy"` altijd als fallback beschikbaar.
