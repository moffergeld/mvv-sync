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

## Cloud-installatie en WeasyPrint systeembibliotheken

Streamlit Community Cloud gebruikt `environment.yml`. Deze richt zich op de
`base`-omgeving met Python 3.14, waar de cloud de Streamlit-server start, en
installeert WeasyPrint via `conda-forge`, inclusief de benodigde Pango-, HarfBuzz-
en Fontconfig-bibliotheken. De overige Python-dependencies staan expliciet in
het `pip`-gedeelte van dezelfde file. Zo zijn deze niet afhankelijk van de
werkdirectory waarmee de cloud-installer een geneste requirementsfile opent.
Houd deze lijst gelijk aan `requirements.txt`, behalve WeasyPrint dat Conda levert.
De lokale developmentcontainer gebruikt `requirements.txt` met Python 3.11.

De GitHub Actions-workflow `Check cloud runtime` installeert deze configuratie in
een Linux Conda-baseomgeving. De controle importeert alle app-dependencies,
waaronder `extra_streamlit_components`, en rendert een kleine PDF met WeasyPrint.
Dit controleert de daadwerkelijke installatie; een dependency-solve alleen
bewijst niet dat de modules beschikbaar zijn in de interpreter van de server.

Er staat geen `packages.txt` in de repository-root. Daardoor vraagt deze app geen
APT-installatie aan tijdens de cloud-build. Op 9 september 2026 mislukte die stap
door een verlopen `bullseye-security/InRelease` in de Streamlit-serveromgeving,
voordat de Python-dependencies werden geinstalleerd. De Conda-installatie levert
de PDF-bibliotheken zonder deze Debian-packagebron te gebruiken.

De Debian-pakketten voor de lokale developmentcontainer staan in
`.devcontainer/packages.txt`; de container installeert deze zelf op Debian Bookworm.
Voor lokale Windows-omgevingen kan een extra GTK/Pango-runtime nodig zijn.

Na publicatie van deze wijziging moet de cloud-build `environment.yml` verwerken.
Controleer daarna of het dashboard start en zowel een weekrapport als een
spelersrapport als PDF kan worden gedownload. Als de oude build zichtbaar blijft,
gebruik dan `Manage app` > `Reboot app`. Een geslaagde lokale controle bewijst nog
niet dat de cloud-deploy is hersteld.

Bronnen: [Streamlit dependencyselectie](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
en [WeasyPrint via Conda](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#conda).

## Secrets en Supabase

Gebruik voor lokale of cloud-configuratie altijd een niet-getrackte secretsfile:

- `.streamlit/secrets.toml`

Er staat nu een veilige voorbeeldfile in de repo:

- `.streamlit/secrets.toml.example`

Belangrijk:

- zet nooit Postgres-connection strings of service-role keys in tracked bestanden
- gebruik voor de hoofdapp minimaal `SUPABASE_URL` en `SUPABASE_ANON_KEY`
- gebruik voor de losse tablet-app daarnaast `SUPABASE_SERVICE_ROLE_KEY`, `TABLET_SHARED_CODE` en `TABLET_CREATED_BY_USER_ID`

Als er eerder credentials in de repo hebben gestaan, roteer die dan ook in Supabase.

## Supabase Preview

Deze repo bevat momenteel geen lokale `supabase/` projectmap of Supabase Branching-configuratie.
De GitHub-check `Supabase Preview` wordt dus niet vanuit deze repo zelf aangestuurd, maar vanuit een externe Supabase/GitHub-integratie.

Om onnodige preview-checks te voorkomen is de `keep-awake` workflow aangepast:

- geen lege commits meer naar `main`
- een periodiek bezoek aan de live Streamlit-apps

Optioneel kun je hiervoor GitHub repository variables instellen:

- `KEEP_AWAKE_URLS`
- `KEEP_AWAKE_URL`
- `KEEP_AWAKE_SECONDARY_URL`

Aanbevolen:

- zet in `KEEP_AWAKE_URLS` alle live app-URL's, bij voorkeur een per regel
- gebruik hiervoor de gewone app-URL, niet `/_stcore/health`
- voeg hier dus zowel de hoofdapp als de losse `tablet_app`-deploy toe als die apart online staat

Voorbeeld:

```text
https://mvv-dashboard.streamlit.app
https://jouw-tablet-app.streamlit.app
```

Compatibiliteit:

- `KEEP_AWAKE_URL` en `KEEP_AWAKE_SECONDARY_URL` blijven ondersteund
- als `KEEP_AWAKE_URLS` leeg is, valt de workflow terug op deze oudere variabelen
- als alles leeg is, gebruikt de workflow standaard:
  `https://mvv-dashboard.streamlit.app`

De workflow draait nu elke 4 uur en normaliseert oude health-URL's automatisch terug naar de gewone app-URL:

- `https://mvv-dashboard.streamlit.app`
- `https://mvv-dashboard.streamlit.app/_stcore/health`

Als `Supabase Preview` ondanks dit toch blijft falen, dan moet die integratie in Supabase of GitHub zelf worden aangepast of uitgezet voor deze repo.
