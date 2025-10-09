# Waarneming-app voor Vespa velutina

Streamlit-dashboard dat nesten van de Aziatische hoornaar van [waarneming.nl](https://waarneming.nl) ophaalt, combineert met lokale notities en toont op een kaart.

## Installatie

Installeer alle vereiste Python-pakketten (Streamlit, pandas, requests, pydeck, BeautifulSoup):

```bash
pip install -r requirements.txt
```

## Applicatie starten

Voer de Streamlit-app lokaal uit:

```bash
streamlit run streamlit_app.py
```

De webapp draait standaard op <http://localhost:8501>. Gebruik de sidebar om filters aan te passen: meerdere locaties toevoegen, datumrange, activiteit, en maximale recordlimiet.

## Belangrijkste functionaliteit

- **Scraper**: `waarneming_scraper.py` haalt observaties op rechtstreeks van waarneming.nl en verrijkt ieder record met exacte coördinaten.
- **Caching**: resultaten worden lokaal opgeslagen in `notes.db` (SQLite) zodat herladen sneller gaat; gebruik de knop *Verversen* om de cache te legen en opnieuw te scrapen.
- **Notities**: in de app kunnen statussen/opmerkingen per observatie bijgehouden worden; deze worden eveneens in `notes.db` bewaard.
- **Locatie-aliases**: standaard worden waarnemingen voor ’s-Hertogenbosch, Rosmalen en Empel gecombineerd; extra locaties kunnen via de sidebar worden toegevoegd.

## Ontwikkeltips

- De scraper gebruikt een gestandaardiseerde lijst van kolommen (`EXPECTED_COLUMNS`). Pas deze alleen aan als upload- of UI-code mee muteert.
- Wanneer je code wijzigt die onder `@st.cache_data` valt, vergeet niet via de *Verversen*-knop de cache te legen om nieuwe resultaten te zien.
