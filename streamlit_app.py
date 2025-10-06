import time
import sqlite3
import math
from datetime import datetime, date, timedelta
from typing import Optional, Tuple, List

import pandas as pd
import streamlit as st
import pydeck as pdk

from waarneming_scraper import (
    fetch_waarneming_occurrences,
    WaarnemingScraperError,
)

# ---------------------------------
# Config
# ---------------------------------
st.set_page_config(page_title="Aziatische hoornaar - Nesten (Waarneming.nl)", layout="wide")

ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "")  # in .streamlit/secrets.toml
DB_PATH = "notes.db"
WAARNEMING_SPECIES_ID = 8807
DEFAULT_LOCATION_QUERY = "s-Hertogenbosch"
DEFAULT_HISTORY_DAYS = 90

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


def bbox_from_session() -> Tuple[float, float, float, float]:
    return (
        float(st.session_state["bbox_min_lon"]),
        float(st.session_state["bbox_min_lat"]),
        float(st.session_state["bbox_max_lon"]),
        float(st.session_state["bbox_max_lat"]),
    )


def viewport_to_bbox(viewport: Optional[dict]) -> Optional[Tuple[float, float, float, float]]:
    if not viewport or not isinstance(viewport, dict):
        return None

    try:
        lon = float(viewport["longitude"])
        lat = float(viewport["latitude"])
    except (KeyError, TypeError, ValueError):
        return None

    zoom = float(viewport.get("zoom", 10.0))
    width = float(viewport.get("width") or viewport.get("viewport_width") or 900)
    height = float(viewport.get("height") or viewport.get("viewport_height") or 600)

    lat_rad = math.radians(lat)
    # Avoid divide-by-zero near the poles
    cos_lat = max(math.cos(lat_rad), 1e-6)

    meters_per_pixel = 156543.03392 * cos_lat / (2 ** zoom)
    half_width_m = meters_per_pixel * width / 2
    half_height_m = meters_per_pixel * height / 2

    earth_radius = 6_378_137.0  # meters
    lat_delta = (half_height_m / earth_radius) * (180 / math.pi)
    lon_delta = (half_width_m / (earth_radius * cos_lat)) * (180 / math.pi)

    min_lat = max(-90.0, lat - lat_delta)
    max_lat = min(90.0, lat + lat_delta)
    min_lon = lon - lon_delta
    max_lon = lon + lon_delta

    if min_lon < -180.0:
        min_lon += 360.0
    if max_lon > 180.0:
        max_lon -= 360.0

    return (
        round(min_lon, 6),
        round(min_lat, 6),
        round(max_lon, 6),
        round(max_lat, 6),
    )


def update_bbox_from_viewport(viewport: Optional[dict]) -> bool:
    bbox = viewport_to_bbox(viewport)
    if not bbox:
        return False

    current_bbox = bbox_from_session()
    if all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(bbox, current_bbox)):
        return False

    (
        st.session_state["bbox_min_lon"],
        st.session_state["bbox_min_lat"],
        st.session_state["bbox_max_lon"],
        st.session_state["bbox_max_lat"],
    ) = bbox
    return True


def current_map_viewport() -> Optional[dict]:
    state = st.session_state.get(MAP_WIDGET_KEY)
    if not isinstance(state, dict):
        return None

    for key in ("viewport", "view_state", "last_view_state"):
        candidate = state.get(key)
        if isinstance(candidate, dict):
            return candidate

    # Fallback: some Streamlit versions store the values directly on the dict.
    if {"latitude", "longitude"}.issubset(state.keys()):
        return state

    return None

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

    st.title("Nesten Aziatische hoornaar – Waarneming.nl")
    st.caption("Scrapet waarneming.nl voor Vespa velutina-nesten en bewaart je eigen statussen/opmerkingen lokaal (SQLite).")

    today = date.today()
    default_date_to = today
    default_date_from = max(today - timedelta(days=DEFAULT_HISTORY_DAYS), date(today.year, 1, 1))

    with st.sidebar:
        st.header("Filter")
        location_query = st.text_input(
            "Locatie (zoals in waarneming.nl zoekveld)",
            value=DEFAULT_LOCATION_QUERY,
            help="Bijvoorbeeld 's-Hertogenbosch of een gemeente/wijk."
        ).strip()
        if not location_query:
            location_query = DEFAULT_LOCATION_QUERY

        activity_options = {
            "Alle activiteiten": "",
            "Alleen nesten (activity=NEST)": "NEST",
        }
        activity_label = st.selectbox("Activiteit (website filter)", list(activity_options.keys()), index=1)
        activity_param = activity_options[activity_label]

        st.markdown("**Datumrange**")
        apply_date_filter = st.checkbox("Filter op datumrange", value=False)
        st.caption(
            "Standaard worden waarnemingen uit de afgelopen drie maanden (binnen het huidige jaar) opgehaald."
        )
        col_a, col_b = st.columns(2)
        with col_a:
            date_from = st.date_input(
                "Vanaf",
                value=default_date_from,
                disabled=not apply_date_filter
            )
        with col_b:
            date_to = st.date_input(
                "Tot en met",
                value=default_date_to,
                disabled=not apply_date_filter
            )

        st.markdown("**Gebied (kaart)**")
        curr_min_lon, curr_min_lat, curr_max_lon, curr_max_lat = bbox_from_session()
        st.caption(
            "Kaart start in ’s-Hertogenbosch’. Versleep of zoom de kaart en klik onder de kaart op ‘Filters toepassen op kaart’ om het gebied bij te werken."
        )
        st.text(
            f"minLon: {curr_min_lon:.4f}\nminLat: {curr_min_lat:.4f}\nmaxLon: {curr_max_lon:.4f}\nmaxLat: {curr_max_lat:.4f}"
        )
        reset_bbox = st.button("Reset naar preset ’s-Hertogenbosch’")

        max_records = st.slider("Max records", 50, 500, 200, step=50)

        st.markdown("---")
        st.caption("Klik ‘Verversen’ om filters toe te passen.")
        refresh = st.button("Verversen", type="primary")
        force_refresh = st.button(
            "Forceer verversen",
            help="Leegt de cache en haalt de waarneming-data opnieuw op."
        )

    if force_refresh:
        fetch_waarneming_occurrences.clear()
        st.session_state.pop("df", None)
        st.session_state.pop("last_params", None)
        refresh = True
        with st.sidebar:
            st.success("Cache geleegd. Data wordt opnieuw opgehaald…")

    if reset_bbox:
        (
            st.session_state["bbox_min_lon"],
            st.session_state["bbox_min_lat"],
            st.session_state["bbox_max_lon"],
            st.session_state["bbox_max_lat"],
        ) = DEN_BOSCH_BBOX
        st.session_state.pop("df", None)
        st.session_state.pop("last_params", None)
        refresh = True

    date_from_value = (
        date_from if apply_date_filter and isinstance(date_from, date) else default_date_from
    )
    date_to_value = (
        date_to if apply_date_filter and isinstance(date_to, date) else default_date_to
    )
    min_lon, min_lat, max_lon, max_lat = bbox_from_session()

    # Data ophalen
    if "df" not in st.session_state or refresh or "last_params" not in st.session_state:
        try:
            df = fetch_waarneming_occurrences(
                species_id=WAARNEMING_SPECIES_ID,
                location_query=location_query,
                date_from=pd.Timestamp(date_from_value) if date_from_value else None,
                date_to=pd.Timestamp(date_to_value) if date_to_value else None,
                activity=activity_param or None,
                max_records=max_records,
            )
        except WaarnemingScraperError as exc:
            st.error(f"Kon waarnemingen niet ophalen: {exc}")
            df = pd.DataFrame(columns=[
                "id",
                "lat",
                "lon",
                "date",
                "location",
                "reporter",
                "activity",
                "count",
                "details",
                "observation_url",
            ])

        if not df.empty:
            df = df.dropna(subset=["lat", "lon"])
            if min_lon <= max_lon:
                lon_mask = df["lon"].between(min_lon, max_lon)
            else:
                lon_mask = (df["lon"] >= min_lon) | (df["lon"] <= max_lon)
            lat_mask = df["lat"].between(min_lat, max_lat)
            df = df[lon_mask & lat_mask].reset_index(drop=True)

        st.session_state["df"] = df
        st.session_state["last_params"] = (
            location_query,
            activity_param,
            (min_lon, min_lat, max_lon, max_lat),
            date_from_value,
            date_to_value,
            max_records,
            apply_date_filter,
        )
    else:
        df = st.session_state["df"].copy()

    df = df.copy()

    # Merge met notities
    notes_df = fetch_notes()
    notes_df.rename(columns={"observation_id": "id"}, inplace=True)
    merged = df.merge(notes_df, on="id", how="left")

    # Snel-filters (client side)
    left, right = st.columns([3,2])
    map_df = None
    with left:
        st.subheader("Kaart")
        if merged.empty:
            st.info("Geen resultaten voor deze filters.")
        else:
            # Groepeer meldingen op identieke coördinaten zodat overlappende
            # punten zichtbaar blijven en een teller krijgen.
            map_source = merged.copy()
            if "activity" not in map_source.columns:
                map_source["activity"] = ""

            status_colors = {
                "Verwijderd": [30, 150, 30],
                "Onvindbaar": [200, 100, 0],
                "Anders": [100, 100, 200],
                "Open": [200, 30, 30],
                "(leeg)": [120, 120, 120],
            }
            default_color = status_colors["(leeg)"]

            def status_display(row: pd.Series) -> str:
                return row["status"] if pd.notna(row.get("status")) else "(leeg)"

            map_source["status_display"] = map_source.apply(status_display, axis=1)

            def summarise_group(group: pd.DataFrame) -> pd.Series:
                statuses = group["status_display"].dropna()
                status = statuses.mode().iloc[0] if not statuses.empty else "(leeg)"
                color = status_colors.get(status, default_color)

                date_values = [d for d in group["date"] if isinstance(d, date)]
                if date_values:
                    start = min(date_values).isoformat()
                    end = max(date_values).isoformat()
                    date_range = start if start == end else f"{start} – {end}"
                else:
                    date_range = "-"

                activities = sorted({a for a in group["activity"] if isinstance(a, str) and a})
                activity_summary = ", ".join(activities) if activities else "-"

                locations = group["location"].dropna()
                location_summary = locations.mode().iloc[0] if not locations.empty else "-"

                reporters = sorted({r for r in group["reporter"] if isinstance(r, str) and r})
                reporter_summary = ", ".join(reporters[:3]) if reporters else "-"
                if reporters and len(reporters) > 3:
                    reporter_summary += " …"

                ids = list(group["id"])
                sample_ids = ", ".join(ids[:3]) if ids else "-"
                if len(ids) > 3:
                    sample_ids += f" … (+{len(ids) - 3})"

                return pd.Series({
                    "count": len(group),
                    "status_display": status,
                    "color": color,
                    "date_range": date_range,
                    "activity_summary": activity_summary,
                    "location_summary": location_summary,
                    "reporter_summary": reporter_summary,
                    "sample_ids": sample_ids,
                })

            map_df = map_source.groupby(["lat", "lon"]).apply(summarise_group).reset_index()
            map_df["radius"] = map_df["count"].apply(
                lambda cnt: 35 + 18 * math.sqrt(max(cnt - 1, 0))
            )
            map_df["tooltip_html"] = map_df.apply(
                lambda row: (
                    f"<b>Aantal meldingen:</b> {row['count']}<br/>"
                    f"<b>Meest voorkomende status:</b> {row['status_display']}<br/>"
                    f"<b>Datumrange:</b> {row['date_range']}<br/>"
                    f"<b>Activiteiten:</b> {row['activity_summary']}<br/>"
                    f"<b>Locatie:</b> {row['location_summary']}<br/>"
                    f"<b>Melder(s):</b> {row['reporter_summary']}<br/>"
                    f"<b>Voorbeeld IDs:</b> {row['sample_ids']}"
                ),
                axis=1,
            )

            st.pydeck_chart(pdk.Deck(
                map_style=None,
                initial_view_state=pdk.ViewState(
                    latitude=float(map_df["lat"].mean()),
                    longitude=float(map_df["lon"].mean()),
                    zoom=11,
                    pitch=0
                ),
                layers=[
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=map_df,
                        get_position='[lon, lat]',
                        get_fill_color='color',
                        get_radius='radius',
                        pickable=True
                    )
                ],
                tooltip={"html": "{tooltip_html}", "style": {"color": "white"}}
            ), key=MAP_WIDGET_KEY)

            st.caption(
                f"Markers tonen {int(map_df['count'].sum())} meldingen samengevoegd tot "
                f"{len(map_df)} unieke coördinaten. Straal schaalt mee met het aantal meldingen."
            )

            apply_map_filter = st.button("Filters toepassen op kaart", key="apply_map")
            if apply_map_filter:
                viewport = current_map_viewport()
                if viewport and update_bbox_from_viewport(viewport):
                    st.session_state.pop("df", None)
                    st.session_state.pop("last_params", None)
                    st.rerun()
                else:
                    st.info("Geen wijziging in kaartuitsnede gedetecteerd.")
    with right:
        st.subheader("Telling")
        total = len(merged)
        open_count = (merged["status"] == "Open").sum() if "status" in merged.columns else 0
        removed_count = (merged["status"] == "Verwijderd").sum() if "status" in merged.columns else 0
        st.metric("Totaal", total)
        st.metric("Open", int(open_count))
        st.metric("Verwijderd", int(removed_count))
        unique_locations = len(map_df) if map_df is not None else 0
        st.metric("Unieke locaties op kaart", unique_locations)

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
            "details",
            "status",
            "reporter",
            "location",
            "lat",
            "lon",
            "basisOfRecord",
            "countryCode",
            "observation_url",
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
            selected_id = st.selectbox("Kies melding (waarneming-id)", options=ids)
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
