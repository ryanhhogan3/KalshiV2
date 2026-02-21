"""
Polymarket fetch + flatten logic.

Fetches all active events from the Gamma API and flattens them into
a list of normalised market dicts ready for DB upsert.

Mirrors the role that src/data/overview/all_markets.py plays for Kalshi.
"""

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Iterator, List, Optional

import requests

GAMMA_BASE = "https://gamma-api.polymarket.com"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _maybe_parse_json_string(val: Any) -> Any:
    """If val is a JSON-encoded string that looks like a list/dict, parse it."""
    if isinstance(val, str):
        s = val.strip()
        if (s.startswith("[") and s.endswith("]")) or (
            s.startswith("{") and s.endswith("}")
        ):
            try:
                return json.loads(s)
            except Exception:
                return val
    return val


def _get_json(
    session: requests.Session,
    url: str,
    params: Dict[str, Any],
    timeout: int = 30,
) -> Any:
    r = session.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def iter_active_event_batches(
    limit: int = 100,
    order: str = "volume24hr",
    ascending: str = "false",
    max_retries: int = 5,
) -> Iterator[List[Dict[str, Any]]]:
    """
    Yield pages of active Polymarket events from the Gamma API.
    Uses offset pagination: GET /events?active=true&closed=false&limit=...&offset=...

    Mirrors OverviewAllMarkets.iter_open_markets for Kalshi — yields one
    batch at a time so memory stays bounded.
    """
    session = requests.Session()
    backoff_base = float(os.environ.get("POLY_BACKOFF_BASE", "1.0"))
    offset = 0

    while True:
        base_params: Dict[str, Any] = {
            "active": "true",
            "closed": "false",
            "limit": limit,
            "offset": offset,
        }

        params = dict(base_params)
        if order:
            params["order"] = order
            params["ascending"] = ascending

        url = f"{GAMMA_BASE}/events"
        backoff = backoff_base
        last_err: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                batch = _get_json(session, url, params)
                break
            except requests.HTTPError as e:
                last_err = e
                status = getattr(e.response, "status_code", None)

                # 422 usually means an invalid order param — strip it and retry
                if status == 422 and ("order" in params or "ascending" in params):
                    params = dict(base_params)
                    try:
                        batch = _get_json(session, url, params)
                        break
                    except Exception as e2:
                        last_err = e2

                if status in (429, 500, 502, 503, 504):
                    print(f"[poly_fetch] HTTP {status}, sleeping {backoff:.1f}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(backoff)
                    backoff = min(backoff * 2.0, 32.0)
                    continue

                raise
            except Exception as e:
                last_err = e
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 32.0)
        else:
            raise RuntimeError(
                f"[poly_fetch] Request failed after {max_retries} retries: {last_err}"
            )

        if not batch:
            break
        if not isinstance(batch, list):
            raise RuntimeError(
                f"[poly_fetch] Unexpected /events response type: {type(batch)}; "
                f"payload={str(batch)[:500]}"
            )

        yield batch

        if len(batch) < limit:
            break

        offset += limit


def fetch_all_active_events(limit: int = 100) -> List[Dict[str, Any]]:
    """Materialise all active events into a single list (convenience wrapper)."""
    events: List[Dict[str, Any]] = []
    for batch in iter_active_event_batches(limit=limit):
        events.extend(batch)
    return events


# ---------------------------------------------------------------------------
# Flattening / normalisation
# ---------------------------------------------------------------------------

def _extract_prices(m: Dict[str, Any]):
    """
    Return (outcome_map, yes_price, no_price) from a raw Gamma market dict.

    outcome_map  – {outcome_label: float_price, ...} or None
    yes_price    – float 0-1 for the "Yes" leg (binary markets), or None
    no_price     – float 0-1 for the "No" leg, or None
    """
    outcomes = _maybe_parse_json_string(m.get("outcomes"))
    prices = _maybe_parse_json_string(m.get("outcomePrices"))

    outcome_map = None
    if (
        isinstance(outcomes, list)
        and isinstance(prices, list)
        and len(outcomes) == len(prices)
    ):
        try:
            outcome_map = {
                str(outcomes[i]): float(prices[i]) for i in range(len(outcomes))
            }
        except Exception:
            outcome_map = None

    yes_price: Optional[float] = None
    no_price: Optional[float] = None

    if isinstance(outcome_map, dict):
        # Try standard "Yes"/"No" keys (case-insensitive)
        for k, v in outcome_map.items():
            kl = k.lower()
            if kl == "yes":
                yes_price = v
            elif kl == "no":
                no_price = v

        # Fallback for binary markets with non-standard labels
        if yes_price is None and len(outcome_map) == 2:
            vals = list(outcome_map.values())
            yes_price, no_price = vals[0], vals[1]

    return outcome_map, yes_price, no_price, outcomes, prices


def flatten_markets_from_events(
    events: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Flatten a list of Gamma events into a flat list of market dicts.
    Each entry is ready to pass directly to normalize_poly_market().
    """
    flat: List[Dict[str, Any]] = []

    for ev in events:
        ev_id = ev.get("id")
        ev_slug = ev.get("slug")
        ev_title = ev.get("title")
        ev_category = ev.get("category")
        ev_start = ev.get("startDate")
        ev_end = ev.get("endDate")

        markets = ev.get("markets") or []
        if not isinstance(markets, list):
            continue

        for m in markets:
            outcome_map, yes_price, no_price, outcomes, prices = _extract_prices(m)

            flat.append(
                {
                    "market_id": m.get("id"),
                    "condition_id": m.get("conditionId"),
                    "slug": m.get("slug") or ev_slug,
                    "question": m.get("question"),
                    "category": m.get("category") or ev_category,
                    "active": m.get("active"),
                    "closed": m.get("closed"),
                    "enable_order_book": m.get("enableOrderBook"),
                    "start_date": m.get("startDate") or ev_start,
                    "end_date": m.get("endDate") or ev_end,
                    "liquidity": m.get("liquidity"),
                    "volume": m.get("volume"),
                    "volume_24hr": m.get("volume24hr"),
                    "outcome_yes_price": yes_price,
                    "outcome_no_price": no_price,
                    "outcome_map": outcome_map,
                    "outcomes": outcomes,
                    "outcome_prices": prices,
                    "event_id": ev_id,
                    "event_slug": ev_slug,
                    "event_title": ev_title,
                    "event_category": ev_category,
                    "event_start": ev_start,
                    "event_end": ev_end,
                    # Keep the raw Gamma market object as the safety net
                    "raw": m,
                }
            )

    return flat


def iter_flat_market_batches(limit: int = 100) -> Iterable[List[Dict[str, Any]]]:
    """
    Yield batches of flattened market dicts, one event-page at a time.
    This keeps memory bounded — each batch is one Gamma /events page,
    flattened into its constituent markets.
    """
    for event_batch in iter_active_event_batches(limit=limit):
        yield flatten_markets_from_events(event_batch)
