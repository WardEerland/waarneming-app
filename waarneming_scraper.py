"""Helpers voor het scrapen van waarneming.nl observaties."""
from __future__ import annotations

import time
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
import streamlit as st

BASE_DOMAIN = "https://waarneming.nl"
DEFAULT_USER_AGENT = "waarneming-app/1.0 (+https://example.com/contact)"
REQUEST_TIMEOUT = 20  # seconds
REQUEST_DELAY = 0.1  # seconds between detail fetches
LOCATION_ALIASES: Dict[str, List[str]] = {
    "s-hertogenbosch": ["s-Hertogenbosch", "Rosmalen", "Empel"],
}

EXPECTED_COLUMNS = [
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
]


class WaarnemingScraperError(RuntimeError):
    """Raised when scraping waarneming.nl fails."""


@dataclass
class _ScrapeConfig:
    """Configuratiecontainer voor een enkele scrape-run."""

    species_id: int
    location_query: Optional[str]
    date_from: Optional[pd.Timestamp]
    date_to: Optional[pd.Timestamp]
    activity: Optional[str]
    max_records: int
    max_pages: int


def _build_search_params(cfg: _ScrapeConfig, page: int) -> Dict[str, str]:
    """Stel de queryparameters samen voor de soortpagina."""

    params: Dict[str, str] = {
        "advanced": "on",
        "search": "",
        "user": "",
        "country_division": "",
        "sex": "",
        "month": "",
        "life_stage": "",
        "method": "",
    }

    if cfg.date_from is not None:
        params["date_after"] = cfg.date_from.date().isoformat()
    if cfg.date_to is not None:
        params["date_before"] = cfg.date_to.date().isoformat()
    if cfg.location_query:
        params["location"] = cfg.location_query
    if cfg.activity:
        params["activity"] = cfg.activity
    if page > 1:
        params["page"] = str(page)
    return params


def _extract_observation_id(href: str) -> Optional[str]:
    """Extraheer het observatie-ID uit een relatieve waarneming.nl URL."""

    if not href:
        return None
    parts = href.strip("/").split("/")
    if len(parts) < 2:
        return None
    if parts[0] != "observation":
        return None
    return parts[1]


def _clean_cell_text(cell) -> str:
    """Haal leesbare tekst uit een tabelcel en trim whitespace."""

    return cell.get_text(" ", strip=True) if cell else ""


def _parse_count(details: str) -> Optional[int]:
    """Probeer een telwaarde uit het detailsveld te halen."""

    for token in details.split():
        if token.isdigit():
            try:
                return int(token)
            except ValueError:
                return None
    return None


def _parse_activity(details: str) -> str:
    """Normaliseer het activiteitlabel dat in de details genoemd wordt."""

    text = details.lower()
    if "nest" in text:
        # Normalise naar een korte label
        return "nest"
    return ""


def _parse_date(value: str) -> Optional[pd.Timestamp]:
    """Zet een datumsnaar uit de tabel om naar een pandas Timestamp."""

    if not value:
        return None
    try:
        ts = pd.to_datetime(value, utc=False, errors="coerce")
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return ts


def _fetch_coordinates(session: requests.Session, observation_id: str, cache: Dict[str, Tuple[Optional[float], Optional[float]]]) -> Tuple[Optional[float], Optional[float]]:
    """Vraag de geojson-detailpagina op en haal lat/lon op met caching."""

    if observation_id in cache:
        return cache[observation_id]

    url = f"{BASE_DOMAIN}/observation/{observation_id}/?json=geojson"
    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        cache[observation_id] = (None, None)
        return cache[observation_id]
    except ValueError:
        cache[observation_id] = (None, None)
        return cache[observation_id]

    features = data.get("features") if isinstance(data, dict) else None
    if isinstance(features, list):
        for feature in features:
            geometry = feature.get("geometry") if isinstance(feature, dict) else None
            if not geometry:
                continue
            coords = geometry.get("coordinates")
            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                lon, lat = coords[0], coords[1]
                try:
                    lat_f = float(lat)
                    lon_f = float(lon)
                except (TypeError, ValueError):
                    continue
                cache[observation_id] = (lat_f, lon_f)
                time.sleep(REQUEST_DELAY)
                return cache[observation_id]

    cache[observation_id] = (None, None)
    time.sleep(REQUEST_DELAY)
    return cache[observation_id]


def _resolve_location_queries(location_query: Optional[str]) -> List[Optional[str]]:
    """Zet een gebruikersinvoer om naar een lijst losse locatiequeries."""

    if location_query is None:
        return [None]

    raw = location_query.strip()
    if not raw:
        return [None]

    lower_key = raw.lower()
    if lower_key in LOCATION_ALIASES:
        aliases = LOCATION_ALIASES[lower_key]
        return list(dict.fromkeys(alias.strip() for alias in aliases if alias.strip()))

    normalized = raw.replace("\n", ",")
    if "," in normalized:
        parts = [part.strip() for part in normalized.split(",")]
        expanded: List[str] = []
        for part in parts:
            if not part:
                continue
            alias_key = part.lower()
            if alias_key in LOCATION_ALIASES:
                expanded.extend(LOCATION_ALIASES[alias_key])
            else:
                expanded.append(part)
        if expanded:
            return list(dict.fromkeys(expanded))

    return [raw]


def _scrape(cfg: _ScrapeConfig, user_agent: str) -> pd.DataFrame:
    """Voer de daadwerkelijke scrape uit voor één locatiequery."""

    session = requests.Session()
    session.headers.update({"User-Agent": user_agent or DEFAULT_USER_AGENT})

    base_url = f"{BASE_DOMAIN}/species/{cfg.species_id}/observations/"
    records: List[Dict[str, object]] = []
    coords_cache: Dict[str, Tuple[Optional[float], Optional[float]]] = {}

    for page in range(1, cfg.max_pages + 1):
        if len(records) >= cfg.max_records:
            break

        params = _build_search_params(cfg, page)
        try:
            response = session.get(base_url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise WaarnemingScraperError(f"Kon pagina {page} niet ophalen: {exc}") from exc

        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.select_one(".table-container table")
        if table is None:
            if page == 1:
                raise WaarnemingScraperError("Kon de resultaatentabel niet vinden in de HTML.")
            break

        rows = table.select("tbody tr")
        if not rows:
            break

        for row in rows:
            if len(records) >= cfg.max_records:
                break
            cells = row.find_all("td")
            if len(cells) < 4:
                continue

            link = cells[0].find("a", href=True)
            if not link:
                continue
            obs_id = _extract_observation_id(link["href"])
            if not obs_id:
                continue

            date_text = link.get_text(strip=True)
            count_details = _clean_cell_text(cells[1])
            location_text = _clean_cell_text(cells[2])
            reporter_text = _clean_cell_text(cells[3])
            obs_url = urljoin(BASE_DOMAIN, link["href"])

            timestamp = _parse_date(date_text)
            lat, lon = _fetch_coordinates(session, obs_id, coords_cache)

            records.append({
                "id": obs_id,
                "lat": lat,
                "lon": lon,
                "date": timestamp.date() if timestamp is not None else None,
                "location": location_text or None,
                "reporter": reporter_text or None,
                "activity": _parse_activity(count_details),
                "count": _parse_count(count_details),
                "details": count_details or None,
                "observation_url": obs_url,
            })

        # Veiligheidsbreuk wanneer de huidige pagina minder rijen had dan verwacht.
        if len(rows) < 1:
            break

    if not records:
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    df = pd.DataFrame.from_records(records, columns=EXPECTED_COLUMNS)
    df["id"] = df["id"].astype(str)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    return df


@st.cache_data(show_spinner=True, ttl=900)
def fetch_waarneming_occurrences(
    *,
    species_id: int,
    location_query: str,
    date_from: Optional[pd.Timestamp],
    date_to: Optional[pd.Timestamp],
    activity: Optional[str],
    max_records: int,
    user_agent: str = DEFAULT_USER_AGENT,
) -> pd.DataFrame:
    """Scrape waarneming.nl en retourneer een DataFrame met observaties."""

    max_pages = max(1, math.ceil(max_records / 25))
    location_queries = _resolve_location_queries(location_query)
    collected_frames: List[pd.DataFrame] = []

    for loc_query in location_queries:
        cfg = _ScrapeConfig(
            species_id=species_id,
            location_query=loc_query,
            date_from=date_from,
            date_to=date_to,
            activity=activity,
            max_records=max_records,
            max_pages=max_pages,
        )
        df_part = _scrape(cfg, user_agent)
        if not df_part.empty:
            collected_frames.append(df_part)

    if not collected_frames:
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    combined = pd.concat(collected_frames, ignore_index=True)
    combined = combined.drop_duplicates(subset="id", keep="first")
    combined = combined.head(max_records)
    combined = combined.reindex(columns=EXPECTED_COLUMNS)
    return combined
