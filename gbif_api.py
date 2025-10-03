"""Helpers for querying the GBIF occurrence API.

This module wraps the GBIF Occurrence Search endpoint and exposes small helper
functions for building parameter dictionaries and fetching results with basic
error handling.
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, Optional

import requests

BASE_URL = "https://api.gbif.org/v1/occurrence/search"
REQUEST_TIMEOUT = 10  # seconds


class GBIFAPIError(RuntimeError):
    """Raised when the GBIF API request fails."""


def build_occurrence_params(
    *,
    dataset_key: Optional[str] = None,
    taxon_key: Optional[str] = None,
    country: Optional[str] = None,
    geometry: Optional[str] = None,
    has_coordinate: Optional[bool] = None,
    limit: int = 20,
    offset: int = 0,
    **extra: Any,
) -> Dict[str, Any]:
    """Return a parameters dictionary for the occurrence search endpoint.

    Parameters correspond to documented GBIF Occurrence Search filters.
    Additional keyword arguments are passed through so callers can benefit from
    new API parameters without updating this helper first.
    """

    params: Dict[str, Any] = {"limit": limit, "offset": offset}

    if dataset_key:
        params["datasetKey"] = dataset_key
    if taxon_key:
        params["taxonKey"] = taxon_key
    if country:
        params["country"] = country
    if geometry:
        params["geometry"] = geometry
    if has_coordinate is not None:
        params["hasCoordinate"] = str(has_coordinate).lower()

    params.update(extra)
    return params


def fetch_occurrences(params: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch a single page of occurrence search results.

    Returns the decoded JSON payload on success and raises ``GBIFAPIError``
    otherwise. The payload includes the ``results`` list, paging metadata, and
    other fields described in the GBIF API documentation.
    """

    try:
        response = requests.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - best effort
        raise GBIFAPIError(f"GBIF occurrence search failed: {exc}") from exc

    # Rely on requests to handle JSON decoding and propagate ValueError if the
    # response body is not valid JSON.
    return response.json()


def search_occurrences(
    *,
    dataset_key: Optional[str] = None,
    taxon_key: Optional[str] = None,
    country: Optional[str] = None,
    geometry: Optional[str] = None,
    has_coordinate: Optional[bool] = True,
    limit: int = 20,
    offset: int = 0,
    **extra: Any,
) -> Dict[str, Any]:
    """High-level helper that builds parameters and fetches a page of results."""

    params = build_occurrence_params(
        dataset_key=dataset_key,
        taxon_key=taxon_key,
        country=country,
        geometry=geometry,
        has_coordinate=has_coordinate,
        limit=limit,
        offset=offset,
        **extra,
    )
    return fetch_occurrences(params)


def iter_occurrences(
    *,
    page_size: int = 100,
    max_records: Optional[int] = None,
    **search_kwargs: Any,
) -> Iterator[Dict[str, Any]]:
    """Iterate over occurrence records with automatic paging.

    ``search_kwargs`` are forwarded to :func:`search_occurrences`. Set
    ``max_records`` to limit the total number of records yielded.
    """

    yielded = 0
    offset = search_kwargs.pop("offset", 0)

    while True:
        page = search_occurrences(limit=page_size, offset=offset, **search_kwargs)
        results = page.get("results", [])
        if not results:
            break

        for record in results:
            yield record
            yielded += 1
            if max_records is not None and yielded >= max_records:
                return

        offset += page_size
        end_of_records = page.get("endOfRecords")
        if end_of_records:
            break
