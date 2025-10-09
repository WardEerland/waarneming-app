import json
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
SCRAPE_CACHE_TTL_HOURS = 6

# ’s-Hertogenbosch benaderende bbox (ruim genomen) -> minLon, minLat, maxLon, maxLat
DEN_BOSCH_BBOX = (5.215, 51.63, 5.40, 51.76)
MAP_WIDGET_KEY = "occurrence_map"
FOCUS_LOCK_PATCH_KEY = "_focus_lock_patch_applied"


def disable_focus_lock_trap():
    """Streamlit wraps some widgets in a focus-lock container that may block mouse clicks in iframes."""
    if st.session_state.get(FOCUS_LOCK_PATCH_KEY):
        return
    st.session_state[FOCUS_LOCK_PATCH_KEY] = True
    st.markdown(
        """
        <style>
        [data-testid="stFocusLock"] { pointer-events: none; }
        [data-testid="stFocusLock"] * { pointer-events: auto; }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
    cur.execute("""
        CREATE TABLE IF NOT EXISTS scrape_cache (
            cache_key TEXT PRIMARY KEY,
            payload TEXT NOT NULL,
            created_at TEXT NOT NULL
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


def load_cached_scrape(cache_key: str, ttl_hours: int = SCRAPE_CACHE_TTL_HOURS) -> Optional[pd.DataFrame]:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "SELECT payload, created_at FROM scrape_cache WHERE cache_key = ?",
        (cache_key,),
    )
    row = cur.fetchone()
    con.close()
    if not row:
        return None

    payload, created_at = row
    if created_at:
        try:
            created_dt = datetime.fromisoformat(created_at)
        except ValueError:
            created_dt = None
        if ttl_hours > 0 and created_dt is not None:
            if created_dt < datetime.utcnow() - timedelta(hours=ttl_hours):
                return None

    try:
        records = json.loads(payload)
    except json.JSONDecodeError:
        return None

    if not isinstance(records, list):
        return None

    df = pd.DataFrame.from_records(records)
    if not df.empty and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    return df


def store_cached_scrape(cache_key: str, df: pd.DataFrame) -> None:
    records = df.to_dict(orient="records")
    payload = json.dumps(records, default=str)
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO scrape_cache (cache_key, payload, created_at)
        VALUES (?, ?, ?)
        ON CONFLICT(cache_key) DO UPDATE SET
            payload = excluded.payload,
            created_at = excluded.created_at
        """,
        (cache_key, payload, datetime.utcnow().isoformat()),
    )
    con.commit()
    con.close()


def delete_cached_scrape(cache_key: str) -> None:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("DELETE FROM scrape_cache WHERE cache_key = ?", (cache_key,))
    con.commit()
    con.close()

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
    disable_focus_lock_trap()

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

    cache_key = "|".join([
        location_query or "",
        activity_param or "",
        date_from_value.isoformat() if isinstance(date_from_value, date) else "",
        date_to_value.isoformat() if isinstance(date_to_value, date) else "",
        str(max_records),
    ])

    if force_refresh:
        fetch_waarneming_occurrences.clear()
        delete_cached_scrape(cache_key)
        st.session_state.pop("df", None)
        st.session_state.pop("last_params", None)
        refresh = True
        with st.sidebar:
            st.success("Cache geleegd. Data wordt opnieuw opgehaald…")

    # Data ophalen
    if "df" not in st.session_state or refresh or "last_params" not in st.session_state:
        raw_df = load_cached_scrape(cache_key)
        used_cache = raw_df is not None

        if raw_df is None:
            try:
                raw_df = fetch_waarneming_occurrences(
                    species_id=WAARNEMING_SPECIES_ID,
                    location_query=location_query,
                    date_from=pd.Timestamp(date_from_value) if date_from_value else None,
                    date_to=pd.Timestamp(date_to_value) if date_to_value else None,
                    activity=activity_param or None,
                    max_records=max_records,
                )
                store_cached_scrape(cache_key, raw_df)
            except WaarnemingScraperError as exc:
                st.error(f"Kon waarnemingen niet ophalen: {exc}")
                raw_df = pd.DataFrame(columns=[
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

        df = raw_df.copy()
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
            cache_key,
            used_cache,
        )
        st.session_state["fetch_source"] = "cache" if used_cache else "scrape"
    else:
        df = st.session_state["df"].copy()
        if "fetch_source" not in st.session_state:
            st.session_state["fetch_source"] = "cache"

    df = df.copy()

    # Merge met notities
    notes_df = fetch_notes()
    notes_df.rename(columns={"observation_id": "id"}, inplace=True)
    merged = df.merge(notes_df, on="id", how="left")

    fetch_source = st.session_state.get("fetch_source")
    if fetch_source == "cache":
        st.caption(f"Dataset geladen uit lokale cache (max {SCRAPE_CACHE_TTL_HOURS} uur oud).")
    elif fetch_source == "scrape":
        st.caption("Dataset opnieuw gescrapet van waarneming.nl.")

    # Snel-filters (client side)
    left, right = st.columns([3,2])
    map_df = None
    with left:
        st.subheader("Kaart")
        if merged.empty:
            st.info("Geen resultaten voor deze filters.")
        else:
            map_df = merged.copy()
            if "activity" not in map_df.columns:
                map_df["activity"] = ""

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

            map_df["status_display"] = map_df.apply(status_display, axis=1)
            map_df["color"] = map_df["status_display"].map(status_colors).apply(
                lambda val: val if isinstance(val, list) else default_color
            )
            map_df["radius"] = 35
            map_plot_df = map_df.copy()

            def safe_str(value, empty="-") -> str:
                if pd.isna(value):
                    return empty
                if isinstance(value, datetime):
                    return value.date().isoformat()
                if isinstance(value, date):
                    return value.isoformat()
                return str(value)

            for col in ["id", "activity", "location", "reporter", "status_display"]:
                map_plot_df[col] = map_plot_df[col].apply(lambda v: safe_str(v, ""))

            if "date" in map_plot_df.columns:
                map_plot_df["date"] = map_plot_df["date"].apply(safe_str)

            map_plot_df["tooltip_html"] = map_plot_df.apply(
                lambda row: (
                    f"<b>ID:</b> {row.get('id', '-') or '-'}<br/>"
                    f"<b>Status:</b> {row['status_display']}<br/>"
                    f"<b>Datum:</b> {row.get('date', '-') or '-'}<br/>"
                    f"<b>Activiteit:</b> {row.get('activity') or ''}<br/>"
                    f"<b>Locatie:</b> {row.get('location', '-') or '-'}<br/>"
                    f"<b>Melder:</b> {row.get('reporter', '-') or '-'}"
                ),
                axis=1,
            )

            deck = pdk.Deck(
                map_style=None,
                initial_view_state=pdk.ViewState(
                    latitude=float(map_df["lat"].mean()),
                    longitude=float(map_df["lon"].mean()),
                    zoom=11,
                    pitch=0,
                    controller=True,
                ),
                layers=[
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=map_plot_df,
                        get_position='[lon, lat]',
                        get_fill_color='color',
                        get_radius='radius',
                        pickable=True,
                        auto_highlight=True,
                        id="occurrence-layer",
                        parameters={"pickable": True}
                    )
                ],
                tooltip={"html": "{tooltip_html}", "style": {"color": "white"}}
            )
            map_event = st.pydeck_chart(
                deck,
                key=MAP_WIDGET_KEY,
                selection_mode="single-select",
                on_select="rerun",
                on_click="rerun",
            )

            st.caption("Elke marker toont één waarneming met de exacte coördinaten uit waarneming.nl.")

            with st.expander("Kaart-debug (tijdelijk)"):
                st.json(st.session_state.get(MAP_WIDGET_KEY, {}))

            def extract_id_from_payload(payload) -> Optional[str]:
                if not isinstance(payload, dict):
                    return None
                candidate_obj = payload.get("object")
                if isinstance(candidate_obj, dict) and candidate_obj.get("id") is not None:
                    return str(candidate_obj["id"])
                if payload.get("id") is not None:
                    return str(payload["id"])
                point_index = None
                for idx_key in ("index", "pointIndex"):
                    idx_val = payload.get(idx_key)
                    if isinstance(idx_val, int):
                        point_index = idx_val
                        break
                if isinstance(point_index, int) and 0 <= point_index < len(map_plot_df):
                    return str(map_plot_df.iloc[point_index]["id"])
                return None

            selection_from_event = False
            map_state = st.session_state.get(MAP_WIDGET_KEY)
            picked_id = None

            if map_event is not None:
                event_selections = getattr(map_event, "selection", None)
                event_selections = getattr(event_selections, "value", event_selections)
                if isinstance(event_selections, dict):
                    event_selections = [event_selections]
                if isinstance(event_selections, list):
                    for payload in event_selections:
                        picked_id = extract_id_from_payload(payload)
                        if picked_id:
                            selection_from_event = True
                            break
                if not picked_id:
                    event_click = getattr(map_event, "click_info", None)
                    event_click = getattr(event_click, "value", event_click)
                    if isinstance(event_click, dict):
                        event_click = [event_click]
                    if isinstance(event_click, list):
                        for payload in event_click:
                            picked_id = extract_id_from_payload(payload)
                            if picked_id:
                                selection_from_event = True
                                break

            # editor selection fallback per marker index (pydeck's click_info feature)
            if isinstance(map_state, dict):
                click_info = map_state.get("click_info")
                if isinstance(click_info, dict):
                    click_entries = [click_info]
                elif isinstance(click_info, list) and click_info:
                    click_entries = click_info
                else:
                    click_entries = []

                if click_entries:
                    first_click = click_entries[0]
                    if isinstance(first_click, dict):
                        obj = first_click.get("object")
                        if isinstance(obj, dict) and obj.get("id") is not None:
                            picked_id = str(obj["id"])
                        else:
                            point_index = None
                            for idx_key in ("index", "pointIndex"):
                                idx_val = first_click.get(idx_key)
                                if isinstance(idx_val, int):
                                    point_index = idx_val
                                    break
                            if isinstance(point_index, int) and 0 <= point_index < len(map_plot_df):
                                picked_id = str(map_plot_df.iloc[point_index]["id"])

            if not picked_id and isinstance(map_state, dict):
                selection_payload = None
                for key_name in ("selected_data", "picked_objects"):
                    candidate = map_state.get(key_name)
                    if isinstance(candidate, list) and candidate:
                        selection_payload = candidate[0]
                        break
                if isinstance(selection_payload, dict):
                    obj = selection_payload.get("object") or selection_payload
                    if isinstance(obj, dict) and "id" in obj:
                        picked_id = str(obj["id"])
            if picked_id and st.session_state.get("selected_edit_id") != picked_id:
                st.session_state["selected_edit_id"] = picked_id
                if not selection_from_event:
                    st.rerun()

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
        if map_df is not None:
            unique_locations = map_df[["lat", "lon"]].dropna().drop_duplicates().shape[0]
        else:
            unique_locations = 0
        st.metric("Unieke locaties op kaart", unique_locations)

    list_df = merged.copy()

    st.divider()
    st.subheader("Bewerken")
    if can_edit():
        if list_df.empty:
            st.info("Geen meldingen om te bewerken met de huidige filter.")
            selected_id = None
        else:
            ids = list_df["id"].astype(str).tolist()
            if ids:
                current_selected = st.session_state.get("selected_edit_id")
                if current_selected not in ids:
                    st.session_state["selected_edit_id"] = ids[0]
                selected_id = st.selectbox(
                    "Kies melding (waarneming-id)",
                    options=ids,
                    key="selected_edit_id"
                )
                row = merged[merged["id"].astype(str) == selected_id].iloc[0]
            else:
                selected_id = None
                row = None

            if selected_id and row is not None:
                st.write(
                    f"**Datum:** {row.get('date', '-') or '-'} &nbsp;&nbsp; "
                    f"**Locatie:** {row.get('location','-') or '-'} &nbsp;&nbsp; "
                    f"**Melder:** {row.get('reporter','-') or '-'}"
                )

                lat_val = row.get("lat")
                lon_val = row.get("lon")
                lat_str = f"{lat_val:.5f}" if pd.notna(lat_val) else "-"
                lon_str = f"{lon_val:.5f}" if pd.notna(lon_val) else "-"
                activity_str = row.get("activity") or "-"
                count_val = row.get("count")
                if pd.notna(count_val):
                    if isinstance(count_val, (int, float)) and float(count_val).is_integer():
                        count_str = str(int(count_val))
                    else:
                        count_str = str(count_val)
                else:
                    count_str = "-"
                detail_str = row.get("details") or "-"
                observation_url = row.get("observation_url")
                status_label = row.get("status") if pd.notna(row.get("status")) else "(leeg)"

                st.markdown(
                    "\n".join([
                        f"**ID:** {selected_id}",
                        f"**Activiteit:** {activity_str}",
                        f"**Aantal / details:** {count_str} | {detail_str}",
                        f"**Coördinaten:** {lat_str}, {lon_str}",
                        f"**Status (laatste):** {status_label}",
                    ])
                )
                if observation_url:
                    st.markdown(f"[Open waarneming in browser]({observation_url})")

                status_options = ["Open", "Verwijderd", "Onvindbaar", "Anders"]
                initial_status = row.get("status") if pd.notna(row.get("status")) else "Open"
                if initial_status not in status_options:
                    initial_status = "Open"
                initial_comment = row.get("comment") if pd.notna(row.get("comment")) else ""

                if st.session_state.get("edit_selected_id") != selected_id:
                    st.session_state["edit_selected_id"] = selected_id
                    st.session_state["edit_status"] = initial_status
                    st.session_state["edit_comment"] = initial_comment
                    if "edit_editor" not in st.session_state:
                        st.session_state["edit_editor"] = st.session_state.get("editor_name", "")

                status = st.selectbox(
                    "Status",
                    options=status_options,
                    key="edit_status",
                )
                comment = st.text_area(
                    "Opmerking",
                    key="edit_comment",
                )

                if "edit_editor" not in st.session_state:
                    st.session_state["edit_editor"] = st.session_state.get("editor_name", "")
                editor = st.text_input(
                    "Naam (voor log)",
                    placeholder="Bijv. Ward",
                    key="edit_editor",
                )
                if editor:
                    st.session_state["editor_name"] = editor

                if st.button("Opslaan", type="primary", width="stretch", key="save_note"):
                    if not editor:
                        st.error("Vul je naam in voor de log.")
                    else:
                        upsert_note(selected_id, status, comment, editor)
                        st.success("Opgeslagen.")
                        time.sleep(0.5)
                        st.rerun()
    else:
        st.info("Voer het admin-wachtwoord in de sidebar in om te bewerken.")

    with st.expander("Meldingen (tabeloverzicht)", expanded=False):
        st.subheader("Meldingen")
        # Client-side tekstfilter
        text_filter = st.text_input("Zoek (id/loc/reporter/activiteit)", "", key="table_filter")
        table_df = merged.copy()
        if text_filter.strip():
            t = text_filter.lower()
            hay = pd.Series([""] * len(table_df))
            for col in ["id", "location", "reporter", "activity"]:
                if col in table_df.columns:
                    hay = hay.str.cat(table_df[col].fillna("").astype(str).str.lower(), sep="|")
            table_df = table_df[hay.str.contains(t, na=False)]

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
            if c in table_df.columns
        ]
        st.dataframe(table_df[show_cols].sort_values(by="date", ascending=False), width="stretch", hide_index=True)

    with st.expander("Export notities (CSV)"):
        out = fetch_notes()
        st.download_button("Download notes.csv", out.to_csv(index=False).encode("utf-8"), "notes.csv", "text/csv")


if __name__ == "__main__":
    main()
