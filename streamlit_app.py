import os
import json
import time
import sqlite3
from datetime import datetime, date
from typing import Optional, Tuple, List

import pandas as pd
import requests
import streamlit as st
import pydeck as pdk

# ---------------------------------
# Config
# ---------------------------------
st.set_page_config(page_title="Aziatische hoornaar - Nesten (GBIF)", layout="wide")

ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "")  # in .streamlit/secrets.toml
DB_PATH = "notes.db"
VESPA_VELUTINA_TAXONKEY = 1311006  # GBIF taxonKey
DEFAULT_COUNTRY = "NL"

# ’s-Hertogenbosch benaderende bbox (ruim genomen) -> minLon, minLat, maxLon, maxLat
DEN_BOSCH_BBOX = (5.215, 51.63, 5.40, 51.76)

# ---------------------------------
# DB helpers
# ---------------------------------
def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            observation_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            comment TEXT,
            updated_at TEXT NOT NULL,
            updated_by TEXT
        )
    """)
    con.commit()
    con.close()

def upsert_note(observation_id: str, status: str, comment: str, user: str):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        INSERT INTO notes (observation_id, status, comment, updated_at, updated_by)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(observation_id) DO UPDATE SET
          status=excluded.status,
          comment=excluded.comment,
          updated_at=excluded.updated_at,
          updated_by=excluded.updated_by
    """, (observation_id, status, comment, datetime.utcnow().isoformat(), user))
    con.commit()
    con.close()

def fetch_notes() -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM notes", con)
    con.close()
    return df

# ---------------------------------
# GBIF fetch
# ---------------------------------
def _gbif_page(params: dict) -> dict:
    r = requests.get("https://api.gbif.org/v1/occurrence/search", params=params, timeout=25)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=True, ttl=3600)
def fetch_gbif_occurrences(
    taxon_key: int = VESPA_VELUTINA_TAXONKEY,
    country: Optional[str] = DEFAULT_COUNTRY,
    geometry_envelope: Optional[Tuple[float, float, float, float]] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    max_records: int = 1000
) -> pd.DataFrame:
    """
    Haalt occurrences op uit GBIF en normaliseert naar kolommen: id, lat, lon, date, location, reporter.
    Pagineert tot max_records.
    """
    records: List[dict] = []
    offset = 0
    page_size = 300  # GBIF limiet tot 300

    params = {
        "taxonKey": taxon_key,
        "hasCoordinate": "true",
        "limit": page_size,
        "offset": offset
    }
    if country:
        params["country"] = country
    if date_from or date_to:
        # Formaat: YYYY-MM-DD,YYYY-MM-DD (range, beide kanten optioneel)
        start_str = date_from.isoformat() if date_from else ""
        end_str = date_to.isoformat() if date_to else ""
        if start_str or end_str:
            params["eventDate"] = f"{start_str},{end_str}"

    if geometry_envelope:
        min_lon, min_lat, max_lon, max_lat = geometry_envelope
        params["geometry"] = bbox_to_wkt_polygon(min_lon, min_lat, max_lon, max_lat)

    while len(records) < max_records:
        params["offset"] = offset
        data = _gbif_page(params)
        page = data.get("results", [])
        if not page:
            break
        for occ in page:
            records.append({
                "id": str(occ.get("key")),
                "lat": occ.get("decimalLatitude"),
                "lon": occ.get("decimalLongitude"),
                "date": occ.get("eventDate"),
                "location": occ.get("locality") or occ.get("verbatimLocality"),
                "reporter": occ.get("recordedBy"),
                "countryCode": occ.get("countryCode"),
                "basisOfRecord": occ.get("basisOfRecord"),
                "datasetKey": occ.get("datasetKey"),
            })
            if len(records) >= max_records:
                break
        if len(page) < page_size:
            break
        offset += page_size

    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df

    # Normalisatie van types
    df["id"] = df["id"].astype(str)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    def to_date(val):
        if pd.isna(val):
            return None
        try:
            return pd.to_datetime(val).date()
        except Exception:
            return None
    df["date"] = df["date"].apply(to_date)

    # Alleen met geldige coords
    df = df[pd.notna(df["lat"]) & pd.notna(df["lon"])].reset_index(drop=True)
    return df


def bbox_to_wkt_polygon(min_lon, min_lat, max_lon, max_lat) -> str:
    # lon, lat volgorde; tegen de klok in; sluit de ring
    return (
        "POLYGON(("
        f"{min_lon} {min_lat},"
        f"{max_lon} {min_lat},"
        f"{max_lon} {max_lat},"
        f"{min_lon} {max_lat},"
        f"{min_lon} {min_lat}"
        "))"
    )

# ---------------------------------
# Auth
# ---------------------------------
def can_edit() -> bool:
    if not ADMIN_PASSWORD:
        return True
    if "is_admin" not in st.session_state:
        st.session_state["is_admin"] = False
    if st.session_state["is_admin"]:
        return True
    with st.sidebar:
        st.subheader("Beheer")
        pwd = st.text_input("Admin-wachtwoord", type="password")
        if st.button("Inloggen"):
            if pwd == ADMIN_PASSWORD:
                st.session_state["is_admin"] = True
                st.success("Ingelogd als beheerder.")
            else:
                st.error("Onjuist wachtwoord.")
    return st.session_state["is_admin"]

# ---------------------------------
# UI
# ---------------------------------
def main():
    init_db()

    st.title("Nesten Aziatische hoornaar – via GBIF")
    st.caption("Toont waarnemingen (occurrences) van Vespa velutina en bewaart je eigen statussen/opmerkingen lokaal (SQLite).")

    with st.sidebar:
        st.header("Filter")
        country = st.selectbox("Land (GBIF country code)", ["NL", "BE", "DE", "FR", "None"], index=0)
        country = None if country == "None" else country

        st.markdown("**Datumrange (eventDate)**")
        col_a, col_b = st.columns(2)
        with col_a:
            date_from = st.date_input("Vanaf", value=None)
        with col_b:
            date_to = st.date_input("Tot en met", value=None)

        st.markdown("**Gebied (ENVELOPE bbox)**")
        use_denbosch = st.checkbox("Gebruik preset ’s-Hertogenbosch", value=True)
        if use_denbosch:
            min_lon, min_lat, max_lon, max_lat = DEN_BOSCH_BBOX
        else:
            min_lon = st.number_input("minLon", value=5.215, format="%.6f")
            min_lat = st.number_input("minLat", value=51.630, format="%.6f")
            max_lon = st.number_input("maxLon", value=5.400, format="%.6f")
            max_lat = st.number_input("maxLat", value=51.760, format="%.6f")

        max_records = st.slider("Max records", 100, 5000, 1000, step=100)

        st.markdown("---")
        st.caption("Klik ‘Verversen’ om filters toe te passen.")
        refresh = st.button("Verversen", type="primary")

    # Data ophalen
    if "df" not in st.session_state or refresh or "last_params" not in st.session_state:
        df = fetch_gbif_occurrences(
            taxon_key=VESPA_VELUTINA_TAXONKEY,
            country=country,
            geometry_envelope=(min_lon, min_lat, max_lon, max_lat),
            date_from=date_from if isinstance(date_from, date) else None,
            date_to=date_to if isinstance(date_to, date) else None,
            max_records=max_records
        )
        st.session_state["df"] = df
        st.session_state["last_params"] = (country, (min_lon, min_lat, max_lon, max_lat), date_from, date_to, max_records)
    else:
        df = st.session_state["df"]

    # Merge met notities
    notes_df = fetch_notes()
    notes_df.rename(columns={"observation_id": "id"}, inplace=True)
    merged = df.merge(notes_df, on="id", how="left")

    # Snel-filters (client side)
    left, right = st.columns([3,2])
    with left:
        st.subheader("Kaart")
        if merged.empty:
            st.info("Geen resultaten voor deze filters.")
        else:
            # Kaartweergave (kleur op status)
            show_df = merged.copy()
            def status_display(row):
                return row["status"] if pd.notna(row.get("status")) else "(leeg)"
            show_df["status_display"] = show_df.apply(status_display, axis=1)
            show_df["color"] = show_df["status_display"].map({
                "Verwijderd": [30,150,30],
                "Onvindbaar": [200,100,0],
                "Anders": [100,100,200],
                "Open": [200,30,30],
                "(leeg)": [120,120,120]
            }).fillna([120,120,120])

            st.pydeck_chart(pdk.Deck(
                map_style=None,
                initial_view_state=pdk.ViewState(
                    latitude=float(show_df["lat"].mean()),
                    longitude=float(show_df["lon"].mean()),
                    zoom=11,
                    pitch=0
                ),
                layers=[
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=show_df,
                        get_position='[lon, lat]',
                        get_fill_color='color',
                        get_radius=35,
                        pickable=True
                    )
                ],
                tooltip={"text": "ID: {id}\nDatum: {date}\nStatus: {status_display}\nLocatie: {location}"}
            ))
    with right:
        st.subheader("Telling")
        total = len(merged)
        open_count = (merged["status"] == "Open").sum() if "status" in merged.columns else 0
        removed_count = (merged["status"] == "Verwijderd").sum() if "status" in merged.columns else 0
        st.metric("Totaal", total)
        st.metric("Open", int(open_count))
        st.metric("Verwijderd", int(removed_count))

    st.subheader("Meldingen")
    # Client-side tekstfilter
    text_filter = st.text_input("Zoek (id/loc/reporter)", "")
    list_df = merged.copy()
    if text_filter.strip():
        t = text_filter.lower()
        hay = pd.Series([""] * len(list_df))
        for col in ["id","location","reporter"]:
            if col in list_df.columns:
                hay = hay.str.cat(list_df[col].fillna("").astype(str).str.lower(), sep="|")
        list_df = list_df[hay.str.contains(t, na=False)]

    show_cols = [c for c in ["id","date","status","reporter","location","lat","lon","basisOfRecord","countryCode"] if c in list_df.columns]
    st.dataframe(list_df[show_cols].sort_values(by="date", ascending=False), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Bewerken")
    if can_edit():
        if list_df.empty:
            st.info("Geen meldingen om te bewerken met de huidige filter.")
        else:
            ids = list_df["id"].tolist()
            selected_id = st.selectbox("Kies melding (GBIF occurrence key)", options=ids)
            row = merged[merged["id"] == selected_id].iloc[0]

            st.write(
                f"**Datum:** {row.get('date')} &nbsp;&nbsp; "
                f"**Locatie:** {row.get('location','-')} &nbsp;&nbsp; "
                f"**Melder:** {row.get('reporter','-')}"
            )
            if pd.notna(row.get("lat")) and pd.notna(row.get("lon")):
                st.map(pd.DataFrame({"lat":[row["lat"]], "lon":[row["lon"]]}))

            current_status = row.get("status") if pd.notna(row.get("status")) else "Open"
            status = st.selectbox("Status", options=["Open","Verwijderd","Onvindbaar","Anders"],
                                  index=["Open","Verwijderd","Onvindbaar","Anders"].index(current_status)
                                  if current_status in ["Open","Verwijderd","Onvindbaar","Anders"] else 0)
            comment = st.text_area("Opmerking", value=row.get("comment") if pd.notna(row.get("comment")) else "")

            editor = st.text_input("Naam (voor log)", value=st.session_state.get("editor_name",""), placeholder="Bijv. Ward")
            if editor:
                st.session_state["editor_name"] = editor

            if st.button("Opslaan", type="primary", use_container_width=True):
                if not editor:
                    st.error("Vul je naam in voor de log.")
                else:
                    upsert_note(selected_id, status, comment, editor)
                    st.success("Opgeslagen.")
                    time.sleep(0.5)
                    st.rerun()
    else:
        st.info("Voer het admin-wachtwoord in de sidebar in om te bewerken.")

    with st.expander("Export notities (CSV)"):
        out = fetch_notes()
        st.download_button("Download notes.csv", out.to_csv(index=False).encode("utf-8"), "notes.csv", "text/csv")


if __name__ == "__main__":
    main()
