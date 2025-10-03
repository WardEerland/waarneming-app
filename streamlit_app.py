import time
import sqlite3
import re
import math
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
# Correct GBIF taxonKey for Vespa velutina (Yellow-legged hornet)
VESPA_VELUTINA_TAXONKEY = 1311477
DEFAULT_COUNTRY = "NL"

# ’s-Hertogenbosch benaderende bbox (ruim genomen) -> minLon, minLat, maxLon, maxLat
DEN_BOSCH_BBOX = (5.215, 51.63, 5.40, 51.76)
MAP_WIDGET_KEY = "occurrence_map"


def ensure_bbox_defaults():
    if "bbox_min_lon" not in st.session_state:
        st.session_state["bbox_min_lon"] = DEN_BOSCH_BBOX[0]
    if "bbox_min_lat" not in st.session_state:
        st.session_state["bbox_min_lat"] = DEN_BOSCH_BBOX[1]
    if "bbox_max_lon" not in st.session_state:
        st.session_state["bbox_max_lon"] = DEN_BOSCH_BBOX[2]
    if "bbox_max_lat" not in st.session_state:
        st.session_state["bbox_max_lat"] = DEN_BOSCH_BBOX[3]
    if "use_denbosch" not in st.session_state:
        st.session_state["use_denbosch"] = True


def bbox_from_session() -> Tuple[float, float, float, float]:
    return (
        float(st.session_state["bbox_min_lon"]),
        float(st.session_state["bbox_min_lat"]),
        float(st.session_state["bbox_max_lon"]),
        float(st.session_state["bbox_max_lat"]),
    )


def sync_bbox_from_map_event() -> None:
    """Update stored bbox if the user moved the map and bounds are available."""

    map_state = st.session_state.get(MAP_WIDGET_KEY)
    if not map_state:
        return

    viewport = map_state.get("viewport")
    if not isinstance(viewport, dict):
        return

    bounds = viewport.get("bounds")
    if not bounds or len(bounds) != 2:
        return

    (min_lon, min_lat), (max_lon, max_lat) = bounds
    if None in (min_lon, min_lat, max_lon, max_lat):
        return

    new_bbox = tuple(round(float(val), 6) for val in (min_lon, min_lat, max_lon, max_lat))
    current_bbox = bbox_from_session()

    if all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(new_bbox, current_bbox)):
        return

    (
        st.session_state["bbox_min_lon"],
        st.session_state["bbox_min_lat"],
        st.session_state["bbox_max_lon"],
        st.session_state["bbox_max_lat"],
    ) = new_bbox
    st.session_state["use_denbosch"] = False
    st.session_state["pending_refresh"] = True

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
    expected_columns = [
        "id",
        "lat",
        "lon",
        "date",
        "location",
        "reporter",
        "countryCode",
        "basisOfRecord",
        "datasetKey",
        "occurrenceID",
    ]

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
                "occurrenceID": occ.get("occurrenceID"),
            })
            if len(records) >= max_records:
                break
        if len(page) < page_size:
            break
        offset += page_size

    df = pd.DataFrame.from_records(records, columns=expected_columns)
    if df.empty:
        return pd.DataFrame(columns=expected_columns)

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

    if "occurrenceID" in df.columns:
        df["occurrenceID"] = df["occurrenceID"].apply(lambda val: str(val) if pd.notna(val) else None)

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


ACTIVITY_PATTERN = re.compile(r"<th>\s*(?:Activity|Activiteit)\s*</th>\s*<td>([^<]+)</td>", re.IGNORECASE)


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_observation_activity(occurrence_url: Optional[str]) -> Optional[str]:
    """Return the activity label from the linked Observation.org page if available."""

    if not occurrence_url or "observation.org/observation/" not in occurrence_url:
        return None

    headers = {"User-Agent": "waarneming-app/1.0 (+https://waarneming.nl)"}
    try:
        response = requests.get(occurrence_url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException:
        return None

    match = ACTIVITY_PATTERN.search(response.text)
    if match:
        return match.group(1).strip()
    return None


def ensure_activity_column(df: pd.DataFrame) -> pd.DataFrame:
    """Populate an ``activity`` column by scraping Observation.org when needed."""

    if df.empty:
        if "activity" not in df.columns:
            df = df.copy()
            df["activity"] = pd.NA
        return df

    if "occurrenceID" not in df.columns:
        df = df.copy()
        df["activity"] = pd.NA
        return df

    df = df.copy()
    if "activity" not in df.columns:
        df["activity"] = pd.NA

    mask = df["occurrenceID"].notna() & df["activity"].isna()
    if not mask.any():
        return df

    for idx, url in df.loc[mask, "occurrenceID"].items():
        activity = fetch_observation_activity(url)
        df.at[idx, "activity"] = activity if activity is not None else ""

    return df

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
    ensure_bbox_defaults()
    sync_bbox_from_map_event()

    st.title("Nesten Aziatische hoornaar – via GBIF")
    st.caption("Toont waarnemingen (occurrences) van Vespa velutina en bewaart je eigen statussen/opmerkingen lokaal (SQLite).")

    with st.sidebar:
        st.header("Filter")
        country = st.selectbox("Land (GBIF country code)", ["NL", "BE", "DE", "FR", "None"], index=0)
        country = None if country == "None" else country

        st.markdown("**Datumrange (eventDate)**")
        apply_date_filter = st.checkbox("Filter op datumrange", value=False)
        col_a, col_b = st.columns(2)
        with col_a:
            date_from = st.date_input("Vanaf", value=None, disabled=not apply_date_filter)
        with col_b:
            date_to = st.date_input("Tot en met", value=None, disabled=not apply_date_filter)

        st.markdown("**Gebied (ENVELOPE bbox)**")
        use_denbosch = st.checkbox(
            "Gebruik preset ’s-Hertogenbosch",
            value=st.session_state.get("use_denbosch", True),
            key="use_denbosch",
        )
        if use_denbosch:
            (
                st.session_state["bbox_min_lon"],
                st.session_state["bbox_min_lat"],
                st.session_state["bbox_max_lon"],
                st.session_state["bbox_max_lat"],
            ) = DEN_BOSCH_BBOX

        col_lon, col_lat = st.columns(2)
        with col_lon:
            st.number_input(
                "minLon",
                value=st.session_state["bbox_min_lon"],
                key="bbox_min_lon",
                format="%.6f",
                disabled=use_denbosch,
            )
            st.number_input(
                "maxLon",
                value=st.session_state["bbox_max_lon"],
                key="bbox_max_lon",
                format="%.6f",
                disabled=use_denbosch,
            )
        with col_lat:
            st.number_input(
                "minLat",
                value=st.session_state["bbox_min_lat"],
                key="bbox_min_lat",
                format="%.6f",
                disabled=use_denbosch,
            )
            st.number_input(
                "maxLat",
                value=st.session_state["bbox_max_lat"],
                key="bbox_max_lat",
                format="%.6f",
                disabled=use_denbosch,
            )

        st.markdown("**Activiteit**")
        only_nests = st.checkbox("Alleen nesten", value=False, help="Filtert waarnemingen waar Activity = 'nest' op Observation.org")

        max_records = st.slider("Max records", 100, 5000, 1000, step=100)

        st.markdown("---")
        st.caption("Klik ‘Verversen’ om filters toe te passen.")
        refresh = st.button("Verversen", type="primary")

    if st.session_state.pop("pending_refresh", False):
        refresh = True

    date_from_value = date_from if apply_date_filter and isinstance(date_from, date) else None
    date_to_value = date_to if apply_date_filter and isinstance(date_to, date) else None
    min_lon, min_lat, max_lon, max_lat = bbox_from_session()

    # Data ophalen
    if "df" not in st.session_state or refresh or "last_params" not in st.session_state:
        df = fetch_gbif_occurrences(
            taxon_key=VESPA_VELUTINA_TAXONKEY,
            country=country,
            geometry_envelope=(min_lon, min_lat, max_lon, max_lat),
            date_from=date_from_value,
            date_to=date_to_value,
            max_records=max_records
        )
        st.session_state["df"] = df
        st.session_state["last_params"] = (
            country,
            (min_lon, min_lat, max_lon, max_lat),
            date_from_value,
            date_to_value,
            max_records,
            apply_date_filter,
        )
    else:
        df = st.session_state["df"]

    if only_nests:
        base_df = st.session_state["df"]
        needs_activity = "activity" not in base_df.columns or base_df["activity"].isna().any()
        if needs_activity:
            with st.spinner("Activiteiten ophalen voor nestfilter..."):
                enriched = ensure_activity_column(base_df)
            st.session_state["df"] = enriched
        else:
            enriched = base_df
        df = enriched
        df = df[df["activity"].fillna("").str.lower().str.contains("nest")].reset_index(drop=True)
    else:
        df = df.copy()

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
            if "activity" not in show_df.columns:
                show_df["activity"] = ""
            show_df["date"] = show_df["date"].apply(
                lambda val: val.isoformat() if isinstance(val, date) else ""
            )
            def status_display(row):
                return row["status"] if pd.notna(row.get("status")) else "(leeg)"
            show_df["status_display"] = show_df.apply(status_display, axis=1)
            status_colors = {
                "Verwijderd": [30, 150, 30],
                "Onvindbaar": [200, 100, 0],
                "Anders": [100, 100, 200],
                "Open": [200, 30, 30],
                "(leeg)": [120, 120, 120],
            }
            default_color = status_colors["(leeg)"]
            show_df["color"] = show_df["status_display"].map(status_colors).apply(
                lambda value: value if isinstance(value, list) else default_color
            )

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
                tooltip={"text": "ID: {id}\nDatum: {date}\nActiviteit: {activity}\nStatus: {status_display}\nLocatie: {location}"}
            ), key=MAP_WIDGET_KEY)
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
    text_filter = st.text_input("Zoek (id/loc/reporter/activiteit)", "")
    list_df = merged.copy()
    if text_filter.strip():
        t = text_filter.lower()
        hay = pd.Series([""] * len(list_df))
        for col in ["id", "location", "reporter", "activity"]:
            if col in list_df.columns:
                hay = hay.str.cat(list_df[col].fillna("").astype(str).str.lower(), sep="|")
        list_df = list_df[hay.str.contains(t, na=False)]

    show_cols = [
        c
        for c in [
            "id",
            "date",
            "activity",
            "status",
            "reporter",
            "location",
            "lat",
            "lon",
            "basisOfRecord",
            "countryCode",
        ]
        if c in list_df.columns
    ]
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
