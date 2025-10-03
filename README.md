# 🎈 Blank app template

A simple Streamlit app template for you to modify!

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)

### Installing dependencies

Install the Python packages listed in `requirements.txt` — including `requests`, which the GBIF helper uses for HTTP calls:

```
pip install -r requirements.txt
```

### Using the GBIF helper module

The `gbif_api.py` module wraps the [GBIF Occurrence API](https://www.gbif.org/developer/occurrence) and supports filters such as dataset, taxon, and location. This example fetches the first five Eurasian blackbird observations recorded in the Netherlands:

```
python - <<'PY'
from gbif_api import search_occurrences

response = search_occurrences(taxon_key=5231190, country="NL", limit=5)
for record in response.get("results", []):
    species = record.get("species")
    lat = record.get("decimalLatitude")
    lon = record.get("decimalLongitude")
    print(species, lat, lon)
PY
```

### How to run the Streamlit app

```
streamlit run streamlit_app.py
```
