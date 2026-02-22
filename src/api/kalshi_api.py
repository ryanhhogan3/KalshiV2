import math
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from src.data.db_connect import connect


ALLOWED_ORIGINS = [
    "https://predictionshift.com",
    "http://localhost:5173",
    "http://localhost:3000",
    "https://api.predictionshift.com"
]


app = FastAPI(title="Kalshi Analytics API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
def _run_query(sql: str):
    """Execute a read-only SQL query using psycopg via db_connect.connect.

    Returns a list of row tuples.
    """

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            if cur.description:
                return cur.fetchall()
            return []

def _fetch_tradability_raw(scan=5000):
    scan = max(1, min(int(scan), 20000))
    sql = f"""
        SELECT market_ticker, title, volume, open_interest, yes_bid, yes_ask, updated_at
        FROM kalshi.open_markets
        WHERE status='active'
          AND yes_bid IS NOT NULL
          AND yes_ask IS NOT NULL
        ORDER BY updated_at DESC
        LIMIT {scan}
    """
    rows = _run_query(sql)
    cols = ["market_ticker","title","volume","open_interest","yes_bid","yes_ask","updated_at"]
    return [dict(zip(cols, r)) for r in rows]

def _query_tradability_score(limit=100, scan=5000, min_spread_ticks=1):
    limit = max(1, min(int(limit), 200))
    scan = max(1, min(int(scan), 20000))
    min_spread_ticks = max(1, int(min_spread_ticks))

    raw = _fetch_tradability_raw(scan)

    out = []
    for m in raw:
        vol = m.get("volume") or 0
        oi  = m.get("open_interest") or 0

        yes_bid = m.get("yes_bid")
        yes_ask = m.get("yes_ask")
        if yes_bid is None or yes_ask is None:
            continue

        spread = yes_ask - yes_bid
        if spread < min_spread_ticks:
            continue

        # score = (ln(1+vol) + 0.7*ln(1+oi)) / spread
        score = (math.log1p(vol) + 0.7 * math.log1p(oi)) / float(spread)

        out.append({
            "market_ticker": m["market_ticker"],
            "title": m.get("title"),
            "volume": float(vol),
            "open_interest": float(oi),
            "yes_bid": float(yes_bid),
            "yes_ask": float(yes_ask),
            "spread_ticks": float(spread),
            "tradability_score": round(score, 4),
            "updated_at": m["updated_at"].isoformat() if m.get("updated_at") else None,
        })

    out.sort(key=lambda x: x["tradability_score"], reverse=True)
    return out[:limit]

def _query_export_runs(limit=20):
    # LIMIT bind params can be finicky; safest is clamp + format int
    limit = max(1, min(int(limit), 100))
    sql = f"""
        SELECT run_id, started_at, finished_at, source, host, n_markets, note
        FROM kalshi.export_runs
        ORDER BY run_id DESC
        LIMIT {limit}
    """
    rows = _run_query(sql)
    cols = ["run_id","started_at","finished_at","source","host","n_markets","note"]
    return [dict(zip(cols, r)) for r in rows]

def _fetch_all_active_raw(scan=5000):
    limit = max(1, min(int(scan), 5000))

    sql = f"""
        SELECT market_ticker, title, yes_bid, yes_ask, volume, open_interest, updated_at
        FROM kalshi.open_markets
        WHERE status = 'active'
        ORDER BY updated_at DESC
        LIMIT {limit}
    """

    rows = _run_query(sql)
    cols = ["ticker", "title", "yes_bid", "yes_ask", "volume", "oi", "updated_at"]
    return [dict(zip(cols, r)) for r in rows]

def _query_opportunity_gap_python(limit=20, scan=1000):
    raw_data = _fetch_all_active_raw(scan)
    
    processed = []
    for m in raw_data:
        # Perform the manipulation in Python
        if m['yes_bid'] and m['yes_ask'] and m['yes_bid'] > 0:
            spread_pts = m['yes_ask'] - m['yes_bid']
            spread_pct = round((spread_pts / m['yes_bid']) * 100, 2)
            
            processed.append({
                "market_ticker": m['ticker'],
                "title": m['title'],
                "spread_points": spread_pts,
                "spread_percentage": spread_pct
            })
    
    # Sort by the calculated field in Python
    processed.sort(key=lambda x: x['spread_percentage'], reverse=True)
    return processed[:limit]

def _query_market_heat_python(limit=200, scan=1000):
    raw_data = _fetch_all_active_raw(scan)
    
    processed = []
    for m in raw_data:
        # Logic: Churn = Volume / Open Interest
        if m['oi'] and m['oi'] > 100 and m['volume'] is not None:
            churn = round(float(m['volume']) / m['oi'], 2)
            processed.append({
                "market_ticker": m['ticker'],
                "title": m['title'],
                "churn_rate": churn
            })
            
    processed.sort(key=lambda x: x['churn_rate'], reverse=True)
    return processed[:limit]


def _query_market_screener(
    limit=50,
    scan=2000,
    min_volume=None,
    min_open_interest=None,
    max_spread_ticks=None,
    sort_by="tradability_score",
):
    """Simple read-only market screener over recent active markets.

    All data comes from kalshi.open_markets via _fetch_all_active_raw.
    Filtering and ranking are done in Python to keep the DB query simple
    and avoid N+1 round-trips.
    """

    # Clamp bounds to keep the query efficient and predictable
    limit = max(1, min(int(limit), 200))
    scan = max(1, min(int(scan), 5000))

    # Normalise numeric filters (they may be None or stringy)
    def _to_float(v):
        if v is None:
            return None
        try:
            return float(v)
        except Exception:
            return None

    min_volume = _to_float(min_volume)
    min_open_interest = _to_float(min_open_interest)
    max_spread_ticks = _to_float(max_spread_ticks)

    raw = _fetch_all_active_raw(scan)

    out = []
    for m in raw:
        yes_bid = m.get("yes_bid")
        yes_ask = m.get("yes_ask")
        if yes_bid is None or yes_ask is None or yes_bid <= 0:
            continue

        vol = m.get("volume") or 0
        oi = m.get("oi") or 0

        spread = yes_ask - yes_bid
        if spread is None:
            continue

        # Apply simple filters
        if min_volume is not None and float(vol) < min_volume:
            continue
        if min_open_interest is not None and float(oi) < min_open_interest:
            continue
        if max_spread_ticks is not None and float(spread) > max_spread_ticks:
            continue

        # Derived metrics
        tradability_score = (math.log1p(float(vol)) + 0.7 * math.log1p(float(oi))) / float(spread or 1.0)
        churn = float(vol) / float(oi) if oi not in (None, 0) else None

        item = {
            "market_ticker": m["ticker"],
            "title": m.get("title"),
            "volume": float(vol),
            "open_interest": float(oi),
            "yes_bid": float(yes_bid),
            "yes_ask": float(yes_ask),
            "spread_ticks": float(spread),
            "tradability_score": round(tradability_score, 4),
            "churn_rate": round(churn, 4) if churn is not None else None,
            "updated_at": m["updated_at"].isoformat() if m.get("updated_at") else None,
        }

        out.append(item)

    # Sorting
    sort_by = (sort_by or "tradability_score").lower()
    # Default: higher is better
    reverse = True

    if sort_by == "spread_ticks":
        # Tighter spreads first
        key_fn = lambda x: x.get("spread_ticks", float("inf"))
        reverse = False
    elif sort_by == "volume":
        key_fn = lambda x: x.get("volume", 0.0)
    elif sort_by in ("open_interest", "oi"):
        key_fn = lambda x: x.get("open_interest", 0.0)
    elif sort_by in ("churn", "churn_rate"):
        key_fn = lambda x: x.get("churn_rate", 0.0)
    else:
        # Default to tradability score
        key_fn = lambda x: x.get("tradability_score", 0.0)

    out.sort(key=key_fn, reverse=reverse)
    return out[:limit]

def query_open_markets(limit=25):
    limit = max(1, min(int(limit), 100))
    
    # FILTER HERE: Add 'AND volume > 0' to your WHERE clause
    sql = f"""
        SELECT 
            market_ticker, title, status, volume, updated_at
        FROM kalshi.open_markets
        WHERE status = 'active' 
          AND volume > 0
        ORDER BY updated_at DESC
        LIMIT {limit}
    """
    
    rows = _run_query(sql)
    cols = ["market_ticker", "title", "status", "volume", "updated_at"]

    result: list[Dict[str, Any]] = []
    for r in rows:
        d = dict(zip(cols, r))
        if d["updated_at"]:
            d["updated_at"] = d["updated_at"].isoformat()
        d["volume"] = float(d["volume"]) if d["volume"] is not None else 0
        result.append(d)

    return result


def _query_top_events_by_open_interest(limit=50):
    """Return top events (series) ordered by total open interest at latest snapshot."""
    limit = max(1, min(int(limit), 200))

    sql = f"""
        WITH latest AS (
            SELECT MAX(snap_ts) AS snap_ts
            FROM kalshi.market_snapshot_series
        )
        SELECT
            s.series_ticker AS event_ticker,
            s.n_markets,
            s.total_volume,
            s.total_open_interest,
            s.avg_spread_ticks
        FROM kalshi.market_snapshot_series s
        JOIN latest l USING (snap_ts)
        ORDER BY s.total_open_interest DESC
        LIMIT {limit}
    """

    rows = _run_query(sql)
    cols = [
        "event_ticker",
        "n_markets",
        "total_volume",
        "total_open_interest",
        "avg_spread_ticks",
    ]
    out = []
    for r in rows:
        d = dict(zip(cols, r))
        # Ensure numerics are JSON-friendly
        for k in ("n_markets", "total_volume", "total_open_interest", "avg_spread_ticks"):
            if d.get(k) is not None:
                d[k] = float(d[k])
        out.append(d)
    return out


def _query_top_events_by_volume(limit=50):
    """Return top events (series) ordered by total volume at latest snapshot."""
    limit = max(1, min(int(limit), 200))

    sql = f"""
        WITH latest AS (
            SELECT MAX(snap_ts) AS snap_ts
            FROM kalshi.market_snapshot_series
        )
        SELECT
            s.series_ticker AS event_ticker,
            s.n_markets,
            s.total_volume,
            s.total_open_interest,
            s.avg_spread_ticks
        FROM kalshi.market_snapshot_series s
        JOIN latest l USING (snap_ts)
        ORDER BY s.total_volume DESC
        LIMIT {limit}
    """

    rows = _run_query(sql)
    cols = [
        "event_ticker",
        "n_markets",
        "total_volume",
        "total_open_interest",
        "avg_spread_ticks",
    ]
    out = []
    for r in rows:
        d = dict(zip(cols, r))
        # Ensure numerics are JSON-friendly
        for k in ("n_markets", "total_volume", "total_open_interest", "avg_spread_ticks"):
            if d.get(k) is not None:
                d[k] = float(d[k])
        out.append(d)
    return out


def _query_global_6h_deltas(limit=9):
    """Return recent 6h deltas of global volume, OI, pricing, and liquidity."""
    limit = max(1, min(int(limit), 200))

    sql = f"""
        WITH g AS (
            SELECT
                snap_ts,
                total_volume,
                total_open_interest,
                n_priced,
                avg_spread_ticks,
                n_wide_spread,
                LAG(total_volume)        OVER (ORDER BY snap_ts) AS prev_volume,
                LAG(total_open_interest) OVER (ORDER BY snap_ts) AS prev_oi,
                LAG(n_priced)            OVER (ORDER BY snap_ts) AS prev_priced,
                LAG(avg_spread_ticks)    OVER (ORDER BY snap_ts) AS prev_spread,
                LAG(n_wide_spread)       OVER (ORDER BY snap_ts) AS prev_wide
            FROM kalshi.market_snapshot_global
        )
        SELECT
            snap_ts,
            total_volume - prev_volume        AS d_volume_6h,
            total_open_interest - prev_oi     AS d_oi_6h,
            n_priced - prev_priced            AS d_priced_6h,
            avg_spread_ticks - prev_spread    AS d_spread_6h,
            n_wide_spread - prev_wide         AS d_wide_6h
        FROM g
        WHERE prev_volume IS NOT NULL
        ORDER BY snap_ts DESC
        LIMIT {limit}
    """

    rows = _run_query(sql)
    cols = [
        "snap_ts",
        "d_volume_6h",
        "d_oi_6h",
        "d_priced_6h",
        "d_spread_6h",
        "d_wide_6h",
    ]
    out = []
    for r in rows:
        d = dict(zip(cols, r))
        # Make numeric fields JSON-friendly
        for k in ("d_volume_6h", "d_oi_6h", "d_priced_6h", "d_spread_6h", "d_wide_6h"):
            if d.get(k) is not None:
                d[k] = float(d[k])
        out.append(d)
    return out


def _query_market_movers(limit=100, min_diff=25):
    """Return markets with large price moves between latest and ~24h-ago snapshots."""
    limit = max(1, min(int(limit), 500))
    min_diff = max(1, int(min_diff))

    sql = f"""
        WITH snaps AS (
            SELECT DISTINCT snap_ts FROM kalshi.market_snapshot_markets
        ),
        latest AS (
            SELECT MAX(snap_ts) AS snap_ts FROM snaps
        ),
        base AS (
            SELECT COALESCE(
                (
                    SELECT snap_ts
                    FROM snaps
                    WHERE snap_ts <= (SELECT snap_ts FROM latest) - (24||' hours')::interval
                    ORDER BY snap_ts DESC
                    LIMIT 1
                ),
                (
                    SELECT snap_ts
                    FROM snaps
                    WHERE snap_ts < (SELECT snap_ts FROM latest)
                    ORDER BY snap_ts DESC
                    LIMIT 1
                )
            ) AS snap_ts
        ),
        pairs AS (
            SELECT
                l.market_ticker,
                l.mid AS mid_now,
                p.mid AS mid_prev,
                ABS(l.mid - p.mid) AS price_diff
            FROM kalshi.market_snapshot_markets l
            JOIN base b ON TRUE
            JOIN kalshi.market_snapshot_markets p
                ON p.snap_ts = b.snap_ts
               AND p.market_ticker = l.market_ticker
            WHERE l.snap_ts = (SELECT snap_ts FROM latest)
        )
        SELECT
            market_ticker,
            mid_prev AS old_price,
            mid_now AS new_price,
            price_diff
        FROM pairs
        WHERE price_diff >= {min_diff}
        ORDER BY price_diff DESC
        LIMIT {limit}
    """

    rows = _run_query(sql)
    cols = ["market_ticker", "old_price", "new_price", "price_diff"]
    out = []
    for r in rows:
        d = dict(zip(cols, r))
        for k in ("old_price", "new_price", "price_diff"):
            if d.get(k) is not None:
                d[k] = float(d[k])
        out.append(d)
    return out


def _query_spread_blowouts(hours=24, limit=100):
    """Return markets with biggest liquidity deterioration (spread blowouts)."""
    # Clamp inputs to reasonable bounds
    hours = max(1, min(int(hours), 168))  # up to 7 days
    limit = max(1, min(int(limit), 500))

    sql = f"""
        WITH snaps AS (
            SELECT DISTINCT snap_ts FROM kalshi.market_snapshot_markets
        ),
        latest AS (
            SELECT MAX(snap_ts) AS snap_ts FROM snaps
        ),
        base AS (
            SELECT COALESCE(
                (
                    SELECT snap_ts
                    FROM snaps
                    WHERE snap_ts <= (SELECT snap_ts FROM latest) - INTERVAL '{hours} hours'
                    ORDER BY snap_ts DESC
                    LIMIT 1
                ),
                (
                    SELECT snap_ts
                    FROM snaps
                    WHERE snap_ts < (SELECT snap_ts FROM latest)
                    ORDER BY snap_ts DESC
                    LIMIT 1
                )
            ) AS snap_ts
        )
        SELECT
            l.market_ticker,
            l.series_ticker AS event_ticker,
            l.spread_ticks AS spread_now,
            p.spread_ticks AS spread_prev,
            (l.spread_ticks - p.spread_ticks) AS d_spread,
            l.mid AS mid_now,
            p.mid AS mid_prev,
            (l.mid - p.mid) AS d_mid
        FROM kalshi.market_snapshot_markets l
        JOIN base b ON TRUE
        JOIN kalshi.market_snapshot_markets p
          ON p.snap_ts = b.snap_ts
         AND p.market_ticker = l.market_ticker
        WHERE l.snap_ts = (SELECT snap_ts FROM latest)
        ORDER BY (l.spread_ticks - p.spread_ticks) DESC
        LIMIT {limit}
    """

    rows = _run_query(sql)
    cols = [
        "market_ticker",
        "event_ticker",
        "spread_now",
        "spread_prev",
        "d_spread",
        "mid_now",
        "mid_prev",
        "d_mid",
    ]
    out = []
    for r in rows:
        d = dict(zip(cols, r))
        for k in ("spread_now", "spread_prev", "d_spread", "mid_now", "mid_prev", "d_mid"):
            if d.get(k) is not None:
                d[k] = float(d[k])
        out.append(d)
    return out


def _query_expiring_markets(hours=48, limit=50):
    """Return top near-expiry markets by open interest at latest snapshot."""
    hours = max(1, min(int(hours), 336))  # up to 14 days
    limit = max(1, min(int(limit), 200))

    sql = f"""
        WITH latest AS (
            SELECT MAX(snap_ts) AS snap_ts
            FROM kalshi.market_snapshot_markets
        )
        SELECT
            market_ticker,
            series_ticker AS event_ticker,
            expiration_time,
            open_interest,
            volume,
            spread_ticks,
            mid
        FROM kalshi.market_snapshot_markets
        WHERE snap_ts = (SELECT snap_ts FROM latest)
          AND expiration_time IS NOT NULL
          AND expiration_time <= NOW() + INTERVAL '{hours} hours'
        ORDER BY open_interest DESC NULLS LAST
        LIMIT {limit}
    """

    rows = _run_query(sql)
    cols = [
        "market_ticker",
        "event_ticker",
        "expiration_time",
        "open_interest",
        "volume",
        "spread_ticks",
        "mid",
    ]
    out = []
    for r in rows:
        d = dict(zip(cols, r))
        if d.get("expiration_time") is not None:
            d["expiration_time"] = d["expiration_time"].isoformat()
        for k in ("open_interest", "volume", "spread_ticks", "mid"):
            if d.get(k) is not None:
                d[k] = float(d[k])
        out.append(d)
    return out


def _query_mid_moves(hours=24, limit=100):
    """Return markets with largest mid-price moves (and spread changes) over a lookback window."""
    hours = max(1, min(int(hours), 168))  # up to 7 days
    limit = max(1, min(int(limit), 500))

    sql = f"""
        WITH snaps AS (
            SELECT DISTINCT snap_ts FROM kalshi.market_snapshot_markets
        ),
        latest AS (
            SELECT MAX(snap_ts) AS snap_ts FROM snaps
        ),
        base AS (
            SELECT COALESCE(
                (
                    SELECT snap_ts
                    FROM snaps
                    WHERE snap_ts <= (SELECT snap_ts FROM latest) - INTERVAL '{hours} hours'
                    ORDER BY snap_ts DESC
                    LIMIT 1
                ),
                (
                    SELECT snap_ts
                    FROM snaps
                    WHERE snap_ts < (SELECT snap_ts FROM latest)
                    ORDER BY snap_ts DESC
                    LIMIT 1
                )
            ) AS snap_ts
        )
        SELECT
            l.market_ticker,
            om.title,
            l.series_ticker AS event_ticker,
            l.mid AS mid_now,
            p.mid AS mid_prev,
            (l.mid - p.mid) AS d_mid,
            l.spread_ticks AS spread_now,
            p.spread_ticks AS spread_prev,
            (l.spread_ticks - p.spread_ticks) AS d_spread
        FROM kalshi.market_snapshot_markets l
        JOIN base b ON TRUE
        JOIN kalshi.market_snapshot_markets p
          ON p.snap_ts = b.snap_ts
         AND p.market_ticker = l.market_ticker
        LEFT JOIN kalshi.open_markets om
          ON om.market_ticker = l.market_ticker
        WHERE l.snap_ts = (SELECT snap_ts FROM latest)
        ORDER BY ABS(l.mid - p.mid) DESC
        LIMIT {limit}
    """

    rows = _run_query(sql)
    cols = [
        "market_ticker",
        "title",
        "event_ticker",
        "mid_now",
        "mid_prev",
        "d_mid",
        "spread_now",
        "spread_prev",
        "d_spread",
    ]
    out = []
    for r in rows:
        d = dict(zip(cols, r))
        for k in ("mid_now", "mid_prev", "d_mid", "spread_now", "spread_prev", "d_spread"):
            if d.get(k) is not None:
                d[k] = float(d[k])
        out.append(d)
    return out


def _query_top_changes_24h(metric="volume", limit=50, min_prev_value=0.0):
    """Top markets by change from oldest->newest snapshot inside the latest 24h window."""
    limit = max(1, min(int(limit), 500))
    try:
        min_prev_value = float(min_prev_value)
    except Exception:
        min_prev_value = 0.0

    metric_map = {
        "volume": "volume",
        "open_interest": "open_interest",
        "oi": "open_interest",
        "mid": "mid",
        "spread_ticks": "spread_ticks",
        "spread": "spread_ticks",
    }
    metric_key = (metric or "volume").lower()
    metric_col = metric_map.get(metric_key)
    if metric_col is None:
        raise ValueError("Unsupported metric. Use one of: volume, open_interest, mid, spread_ticks")

    sql = f"""
        WITH latest AS (
            SELECT MAX(snap_ts) AS snap_ts
            FROM kalshi.market_snapshot_markets
        ),
        window_snaps AS (
            SELECT DISTINCT m.snap_ts
            FROM kalshi.market_snapshot_markets m
            JOIN latest l ON TRUE
            WHERE m.snap_ts >= l.snap_ts - INTERVAL '24 hours'
              AND m.snap_ts <= l.snap_ts
        ),
        base AS (
            SELECT MIN(snap_ts) AS snap_ts
            FROM window_snaps
        )
        SELECT
            l.market_ticker,
            om.title,
            l.series_ticker AS event_ticker,
            p.{metric_col} AS prev_value,
            l.{metric_col} AS current_value,
            (l.{metric_col} - p.{metric_col}) AS delta_value,
            CASE
              WHEN p.{metric_col} = 0 THEN NULL
              ELSE ((l.{metric_col} - p.{metric_col}) / p.{metric_col}) * 100.0
            END AS pct_change,
                        p.volume AS prev_volume,
                        l.volume AS current_volume,
                        (l.volume - p.volume) AS d_volume,
                        p.open_interest AS prev_open_interest,
                        l.open_interest AS current_open_interest,
                        (l.open_interest - p.open_interest) AS d_open_interest,
            (SELECT snap_ts FROM latest) AS latest_snap_ts,
                        (SELECT snap_ts FROM base) AS base_snap_ts,
                        EXTRACT(EPOCH FROM ((SELECT snap_ts FROM latest) - (SELECT snap_ts FROM base))) / 3600.0 AS window_hours
        FROM kalshi.market_snapshot_markets l
        JOIN base b ON TRUE
        JOIN kalshi.market_snapshot_markets p
          ON p.snap_ts = b.snap_ts
         AND p.market_ticker = l.market_ticker
        LEFT JOIN kalshi.open_markets om
          ON om.market_ticker = l.market_ticker
        WHERE l.snap_ts = (SELECT snap_ts FROM latest)
                    AND p.snap_ts = (SELECT snap_ts FROM base)
          AND p.{metric_col} IS NOT NULL
          AND l.{metric_col} IS NOT NULL
          AND ABS(p.{metric_col}) >= {min_prev_value}
        ORDER BY ABS(l.{metric_col} - p.{metric_col}) DESC
                             , ABS(COALESCE(l.volume - p.volume, 0)) + ABS(COALESCE(l.open_interest - p.open_interest, 0)) DESC
                             , l.volume DESC NULLS LAST
        LIMIT {limit}
    """

    rows = _run_query(sql)
    cols = [
        "market_ticker",
        "title",
        "event_ticker",
        "prev_value",
        "current_value",
        "delta_value",
        "pct_change",
        "prev_volume",
        "current_volume",
        "d_volume",
        "prev_open_interest",
        "current_open_interest",
        "d_open_interest",
        "latest_snap_ts",
        "base_snap_ts",
        "window_hours",
    ]

    out = []
    for r in rows:
        d = dict(zip(cols, r))
        for k in (
            "prev_value",
            "current_value",
            "delta_value",
            "pct_change",
            "prev_volume",
            "current_volume",
            "d_volume",
            "prev_open_interest",
            "current_open_interest",
            "d_open_interest",
            "window_hours",
        ):
            if d.get(k) is not None:
                d[k] = float(d[k])
        for ts_col in ("latest_snap_ts", "base_snap_ts"):
            if d.get(ts_col) is not None and hasattr(d[ts_col], "isoformat"):
                d[ts_col] = d[ts_col].isoformat()
        d["metric"] = metric_col
        out.append(d)
    return out


def _query_global_vol_index(points=50, min_open_interest=100):
    """Compute a simple global realized log-odds volatility index over recent snapshots.

    For each pair of consecutive snapshot timestamps, we:
      - join markets present in both snapshots,
      - filter to reasonably liquid markets by open interest and mid in (0.01, 0.99),
      - compute log-odds returns on the mid price,
      - aggregate an open-interest-weighted realized variance.

    We then annualise using the time delta between snapshots and scale by 100
    to form an index value. The result is a small time series of index points,
    one per snapshot pair, ordered from newest to oldest.
    """

    try:
        points = max(1, min(int(points), 200))
    except Exception:
        points = 50
    try:
        min_open_interest = max(0, int(min_open_interest))
    except Exception:
        min_open_interest = 100

    # We keep the heavy lifting inside SQL and only return one row per
    # snapshot pair, keeping this query reasonably light. We also clamp the
    # number of points and require a minimum open interest filter.
    sql = f"""
        WITH snaps AS (
            SELECT DISTINCT snap_ts
            FROM kalshi.market_snapshot_markets
            ORDER BY snap_ts DESC
            LIMIT {points + 1}
        ),
        ordered AS (
            SELECT snap_ts, ROW_NUMBER() OVER (ORDER BY snap_ts) AS rn
            FROM snaps
        ),
        pairs AS (
            SELECT
                cur.snap_ts AS snap_ts,
                prev.snap_ts AS prev_snap_ts,
                EXTRACT(EPOCH FROM (cur.snap_ts - prev.snap_ts)) / 3600.0 AS dt_hours
            FROM ordered cur
            JOIN ordered prev
              ON prev.rn = cur.rn - 1
        ),
        joined AS (
            SELECT
                p.snap_ts,
                p.dt_hours,
                m_now.mid AS mid_now,
                m_prev.mid AS mid_prev,
                m_now.open_interest
            FROM pairs p
            JOIN kalshi.market_snapshot_markets m_now
              ON m_now.snap_ts = p.snap_ts
            JOIN kalshi.market_snapshot_markets m_prev
              ON m_prev.snap_ts = p.prev_snap_ts
             AND m_prev.market_ticker = m_now.market_ticker
            WHERE m_now.open_interest >= {min_open_interest}
              AND m_now.mid > 0.01 AND m_now.mid < 0.99
              AND m_prev.mid > 0.01 AND m_prev.mid < 0.99
        ),
        returns AS (
            SELECT
                snap_ts,
                dt_hours,
                LN(mid_now / (1 - mid_now)) - LN(mid_prev / (1 - mid_prev)) AS r,
                open_interest
            FROM joined
            WHERE dt_hours > 0
        ),
        stats AS (
            SELECT
                snap_ts,
                dt_hours,
                SUM(open_interest * r * r) / NULLIF(SUM(open_interest), 0) AS rv
            FROM returns
            GROUP BY snap_ts, dt_hours
        )
        SELECT
            snap_ts,
            dt_hours,
            rv
        FROM stats
        ORDER BY snap_ts DESC
        LIMIT {points}
    """

    rows = _run_query(sql)

    # Annualisation: treat each rv as variance over dt_hours, scale to a
    # 1-year horizon (using 365 days) and take sqrt.
    HOURS_PER_YEAR = 24.0 * 365.0

    series = []
    for snap_ts, dt_hours, rv in rows:
        if rv is None or dt_hours is None or dt_hours <= 0:
            continue
        try:
            rv = float(rv)
            dt_hours = float(dt_hours)
        except Exception:
            continue

        # Guard against degenerate values
        if rv < 0:
            continue

        annualised_var = rv * (HOURS_PER_YEAR / dt_hours)
        vol_index = 100.0 * math.sqrt(annualised_var) if annualised_var > 0 else 0.0

        # Normalise timestamp to ISO string for JSON friendliness
        ts_val = snap_ts.isoformat() if hasattr(snap_ts, "isoformat") else str(snap_ts)

        series.append({
            "snap_ts": ts_val,
            "dt_hours": round(dt_hours, 6),
            "realised_variance": round(rv, 10) if rv is not None else None,
            "vol_index": round(vol_index, 4),
        })

    return {
        "points": len(series),
        "min_open_interest": min_open_interest,
        "series": series,
    }


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "up"}


@app.get("/export-runs")
@app.get("/export_runs", include_in_schema=False)
def export_runs(limit: int = Query(20, ge=1, le=100)):
    try:
        return _query_export_runs(limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"export-runs failed: {e}")


@app.get("/active-markets")
@app.get("/active_markets", include_in_schema=False)
def active_markets(limit: int = Query(20, ge=1, le=100)):
    try:
        return query_open_markets(limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"active-markets failed: {e}")


@app.get("/opportunity-gap")
@app.get("/opportunity_gap", include_in_schema=False)
def opportunity_gap(
    limit: int = Query(20, ge=1, le=200),
    scan: int = Query(1000, ge=1, le=20000),
):
    try:
        return _query_opportunity_gap_python(limit, scan)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"opportunity-gap failed: {e}")


@app.get("/market-heat")
@app.get("/market_heat", include_in_schema=False)
def market_heat(
    limit: int = Query(20, ge=1, le=200),
    scan: int = Query(1000, ge=1, le=20000),
):
    try:
        return _query_market_heat_python(limit, scan)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"market-heat failed: {e}")


@app.get("/markets/screener")
def market_screener(
    limit: int = Query(50, ge=1, le=200),
    scan: int = Query(2000, ge=1, le=5000),
    min_volume: float | None = Query(None),
    min_open_interest: float | None = Query(None),
    max_spread_ticks: float | None = Query(None),
    sort_by: str = Query("tradability_score"),
):
    try:
        return _query_market_screener(
            limit=limit,
            scan=scan,
            min_volume=min_volume,
            min_open_interest=min_open_interest,
            max_spread_ticks=max_spread_ticks,
            sort_by=sort_by,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"markets/screener failed: {e}")


@app.get("/tradeability-score")
@app.get("/tradability_score", include_in_schema=False)
def tradability_score(
    limit: int = Query(20, ge=1, le=200),
    scan: int = Query(5000, ge=1, le=20000),
    min_spread_ticks: int = Query(1, ge=1),
):
    try:
        return _query_tradability_score(limit, scan, min_spread_ticks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"tradeability-score failed: {e}")


@app.get("/top-events-open-interest")
@app.get("/top_events_open_interest", include_in_schema=False)
def top_events_open_interest(limit: int = Query(50, ge=1, le=200)):
    try:
        return _query_top_events_by_open_interest(limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"top-events-open-interest failed: {e}")


@app.get("/top-events-volume")
@app.get("/top_events_volume", include_in_schema=False)
def top_events_volume(limit: int = Query(50, ge=1, le=200)):
    try:
        return _query_top_events_by_volume(limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"top-events-volume failed: {e}")


@app.get("/global-6h-deltas")
@app.get("/global_6h_deltas", include_in_schema=False)
def global_6h_deltas(limit: int = Query(9, ge=1, le=200)):
    try:
        return _query_global_6h_deltas(limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"global-6h-deltas failed: {e}")


@app.get("/markets/spread-blowouts")
@app.get("/markets/spread_blowouts", include_in_schema=False)
def spread_blowouts(
    hours: int = Query(24, ge=1, le=168),
    limit: int = Query(100, ge=1, le=500),
):
    try:
        return _query_spread_blowouts(hours, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"markets/spread-blowouts failed: {e}")


@app.get("/markets/expiring-soon")
@app.get("/markets/expiring_soon", include_in_schema=False)
def expiring_soon(
    hours: int = Query(48, ge=1, le=336),
    limit: int = Query(50, ge=1, le=200),
):
    try:
        return _query_expiring_markets(hours, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"markets/expiring-soon failed: {e}")


@app.get("/markets/mid-moves")
@app.get("/markets/mid_moves", include_in_schema=False)
def mid_moves(
    hours: int = Query(24, ge=1, le=168),
    limit: int = Query(100, ge=1, le=500),
):
    try:
        return _query_mid_moves(hours, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"markets/mid-moves failed: {e}")


@app.get("/market-movers")
@app.get("/market_movers", include_in_schema=False)
def market_movers(
    limit: int = Query(100, ge=1, le=500),
    min_diff: int = Query(25, ge=1),
):
    try:
        return _query_market_movers(limit, min_diff)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"market-movers failed: {e}")


@app.get("/markets/top-changes-24h")
@app.get("/markets/top_changes_24h", include_in_schema=False)
def top_changes_24h(
    metric: str = Query("volume"),
    limit: int = Query(10, ge=1, le=500),
    min_prev_value: float = Query(0.0, ge=0),
):
    try:
        return _query_top_changes_24h(metric=metric, limit=limit, min_prev_value=min_prev_value)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"markets/top-changes-24h failed: {e}")


@app.get("/vol/index/global")
@app.get("/vol_index_global", include_in_schema=False)
def vol_index_global(
    points: int = Query(50, ge=1, le=200),
    min_open_interest: int = Query(100, ge=0),
):
    try:
        return _query_global_vol_index(points, min_open_interest)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"global-vol-index failed: {e}")

