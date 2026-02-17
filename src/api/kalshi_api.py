import math
import os, json
import pg8000.native

_creds = None
_conn = None

ALLOWED_ORIGINS = {
    "https://predictionshift.com",
    "http://localhost:5173",
    "http://localhost:3000",
}

def _get_origin(event):
    h = event.get("headers") or {}
    # API Gateway/Lambda URLs sometimes vary casing
    return h.get("origin") or h.get("Origin")

def _resp(code, body, event=None, cors=True):
    headers = {"content-type": "application/json"}

    if cors and event is not None:
        origin = _get_origin(event)
        if origin in ALLOWED_ORIGINS:
            headers.update({
                "access-control-allow-origin": origin,
                "access-control-allow-headers": "content-type",
                "access-control-allow-methods": "GET,OPTIONS",
                "vary": "Origin",
            })
        else:
            # No ACAO header if origin not allowed
            # (browser will block, which is what you want)
            pass

    return {"statusCode": code, "headers": headers, "body": json.dumps(body, default=str)}

def _get_creds():
    """Return DB credentials from environment variables.

    Expected env vars:
      DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASS
    """

    global _creds
    if _creds is None:
        _creds = {
            "host": os.environ.get("DB_HOST", "127.0.0.1"),
            "port": int(os.environ.get("DB_PORT", "5432")),
            "database": os.environ.get("DB_NAME", "shift"),
            "user": os.environ.get("DB_USER", "shift_user"),
            "password": os.environ["DB_PASS"],
        }
    return _creds

def _get_conn():
    global _conn
    c = _get_creds()
    
    # Check if connection is actually alive
    if _conn is not None:
        try:
            # Quick "ping" to verify the socket
            _conn.run("SELECT 1")
        except Exception:
            print("Connection dead. Cleaning up...")
            try:
                _conn.close()
            except:
                pass
            _conn = None

    if _conn is None:
        _conn = pg8000.native.Connection(
            user=c["user"],
            password=c["password"],
            host=c["host"],
            port=c["port"],
            database=c["database"],
            ssl_context=True,
            timeout=30, # pg8000 internal socket timeout
        )
    return _conn

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
    rows = _get_conn().run(sql)
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
    rows = _get_conn().run(sql)
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

    rows = _get_conn().run(sql)
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
    
    try:
        rows = _get_conn().run(sql)
        cols = ["market_ticker", "title", "status", "volume", "updated_at"]
        
        result = []
        for r in rows:
            d = dict(zip(cols, r))
            # Safe conversion for JSON serialization
            if d['updated_at']:
                d['updated_at'] = d['updated_at'].isoformat()
            # Ensure volume is a float/int for the frontend
            d['volume'] = float(d['volume']) if d['volume'] is not None else 0
            result.append(d)
                
        return result
    except Exception as e:
        print(f"Error: {e}")
        return []


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

    rows = _get_conn().run(sql)
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

    rows = _get_conn().run(sql)
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

    rows = _get_conn().run(sql)
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

    rows = _get_conn().run(sql)
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

    rows = _get_conn().run(sql)
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

    rows = _get_conn().run(sql)
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

    rows = _get_conn().run(sql)
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

    rows = _get_conn().run(sql)

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


def lambda_handler(event, context):
    path = event.get("rawPath") or event.get("path") or "/"
    method = event.get("requestContext", {}).get("http", {}).get("method") or event.get("httpMethod", "GET")

    # CORS preflight
    if method == "OPTIONS":
        return _resp(200, {"ok": True},event=event)
    
        # Root: list available endpoints and what they return
    if path == "/":
        return _resp(200, {
            "endpoints": {
                "/health": {
                    "description": "Simple health check for the API",
                    "returns": {"status": "up"}
                },
                "/export-runs": {
                    "aliases": ["/export_runs"],
                    "description": "Recent export run metadata ordered by most recent run_id",
                    "query_params": {
                        "limit": "Optional. Max number of runs to return (default 20, max 100)."
                    }
                },
                "/active-markets": {
                    "aliases": ["/active_markets"],
                    "description": "Active markets with non-zero volume ordered by most recent update",
                    "query_params": {
                        "limit": "Optional. Max number of markets to return (default 20, max 100)."
                    }
                },
                "/opportunity-gap": {
                    "aliases": ["/opportunity_gap"],
                    "description": "Active markets ranked by bid/ask spread percentage (Python computed).",
                    "query_params": {
                        "limit": "Optional. Max number of markets to return (default 20).",
                        "scan": "Optional. Number of recent active markets to scan before ranking (default 1000, max 5000)."
                    }
                },
                "/market-heat": {
                    "aliases": ["/market_heat"],
                    "description": "Active markets ranked by churn rate (volume / open interest).",
                    "query_params": {
                        "limit": "Optional. Max number of markets to return (default 20).",
                        "scan": "Optional. Number of recent active markets to scan before ranking (default 1000, max 5000)."
                    }
                },
                "/markets/screener": {
                    "aliases": ["/markets/screener"],
                    "description": "Simple read-only market screener over recent active markets.",
                    "query_params": {
                        "limit": "Optional. Max number of markets to return (default 50, max 200).",
                        "scan": "Optional. Number of recent active markets to scan before screening (default 2000, max 5000).",
                        "min_volume": "Optional. Minimum traded volume to include a market.",
                        "min_open_interest": "Optional. Minimum open interest to include a market.",
                        "max_spread_ticks": "Optional. Maximum bid/ask spread in ticks to include a market.",
                        "sort_by": "Optional. One of tradability_score (default), spread_ticks, volume, open_interest, churn_rate."
                    }
                },
                "/tradeability-score": {
                    "aliases": ["/tradability_score"],
                    "description": "Active markets ranked by custom tradability score based on volume, open interest, and spread.",
                    "query_params": {
                        "limit": "Optional. Max number of markets to return (default 20, max 200).",
                        "scan": "Optional. Number of recent active markets to scan before ranking (default 5000, max 20000).",
                        "min_spread_ticks": "Optional. Minimum bid/ask spread in ticks to include a market (default 1)."
                    }
                },
                "/top-events-open-interest": {
                    "aliases": ["/top_events_open_interest"],
                    "description": "Top events (series) ranked by total open interest at the latest snapshot.",
                    "query_params": {
                        "limit": "Optional. Max number of events to return (default 50, max 200)."
                    }
                },
                "/top-events-volume": {
                    "aliases": ["/top_events_volume"],
                    "description": "Top events (series) ranked by total volume at the latest snapshot.",
                    "query_params": {
                        "limit": "Optional. Max number of events to return (default 50, max 200)."
                    }
                },
                "/global-6h-deltas": {
                    "aliases": ["/global_6h_deltas"],
                    "description": "Global 6h deltas for volume, open interest, priced markets, spreads, and wide-spread breadth.",
                    "query_params": {
                        "limit": "Optional. Max number of 6h periods to return (default 9, max 200)."
                    }
                },
                "/markets/spread-blowouts": {
                    "aliases": ["/markets/spread_blowouts"],
                    "description": "Markets with biggest liquidity deterioration (spread blowouts) over a lookback window.",
                    "query_params": {
                        "hours": "Optional. Lookback window in hours (default 24, max 168).",
                        "limit": "Optional. Max number of markets to return (default 100, max 500)."
                    }
                },
                "/markets/expiring-soon": {
                    "aliases": ["/markets/expiring_soon"],
                    "description": "Markets expiring within a configurable lookahead window, ranked by open interest.",
                    "query_params": {
                        "hours": "Optional. Lookahead window in hours (default 48, max 336).",
                        "limit": "Optional. Max number of markets to return (default 50, max 200)."
                    }
                },
                "/markets/mid-moves": {
                    "aliases": ["/markets/mid_moves"],
                    "description": "Markets with largest mid-price moves over a lookback window, including spread changes.",
                    "query_params": {
                        "hours": "Optional. Lookback window in hours (default 24, max 168).",
                        "limit": "Optional. Max number of markets to return (default 100, max 500)."
                    }
                },
                "/market-movers": {
                    "aliases": ["/market_movers"],
                    "description": "Markets with largest price moves between latest and ~24h-ago snapshots.",
                    "query_params": {
                        "limit": "Optional. Max number of markets to return (default 100, max 500).",
                        "min_diff": "Optional. Minimum absolute price change in ticks to include (default 25)."
                    }
                },
                "/vol/index/global": {
                    "aliases": ["/vol_index_global"],
                    "description": "Global realised log-odds volatility index over recent snapshots.",
                    "query_params": {
                        "points": "Optional. Number of historical index points to return (default 50, max 200).",
                        "min_open_interest": "Optional. Minimum open interest filter per market snapshot (default 100)."
                    }
                }
            }
        },event=event)


    # No DB, no Secrets Manager
    if path == "/health":
        return _resp(200, {"status": "up"}, event=event)

    if path in ("/export-runs", "/export_runs"):
        qs = event.get("queryStringParameters") or {}
        data = _query_export_runs(qs.get("limit", 20))
        return _resp(200, data, event=event)
    
    if path in ("/active-markets", "/active_markets"):
        qs = event.get("queryStringParameters") or {}
        data = query_open_markets(qs.get("limit", 20))
        return _resp(200, data, event=event)
    
    if path in ("/opportunity-gap", "/opportunity_gap"):
        qs = event.get("queryStringParameters") or {}
        try:
            data = _query_opportunity_gap_python(qs.get("limit", 20), qs.get("scan", 1000))
            return _resp(200, data, event=event)
        except Exception as e:
            print(f"opportunity-gap error: {e}")
            return _resp(500, {"error": "opportunity-gap failed", "detail": str(e)}, event=event)

    if path in ("/market-heat", "/market_heat"):
        qs = event.get("queryStringParameters") or {}
        try:
            data = _query_market_heat_python(qs.get("limit", 20), qs.get("scan", 1000))
            return _resp(200, data, event=event)
        except Exception as e:
            print(f"market-heat error: {e}")
            return _resp(500, {"error": "market-heat failed", "detail": str(e)}, event=event)

    if path in ("/markets/screener",):
        qs = event.get("queryStringParameters") or {}
        try:
            data = _query_market_screener(
                limit=qs.get("limit", 250),
                scan=qs.get("scan", 2000),
                min_volume=qs.get("min_volume"),
                min_open_interest=qs.get("min_open_interest"),
                max_spread_ticks=qs.get("max_spread_ticks"),
                sort_by=qs.get("sort_by", "tradability_score"),
            )
            return _resp(200, data, event=event)
        except Exception as e:
            print(f"markets/screener error: {e}")
            return _resp(500, {"error": "markets/screener failed", "detail": str(e)}, event=event)
        
    if path in ("/tradeability-score", "/tradability_score"):
        qs = event.get("queryStringParameters") or {}
        try:
            data = _query_tradability_score(qs.get("limit", 20), qs.get("scan", 1000))
            return _resp(200, data, event=event)
        except Exception as e:
            print(f"tradeability-score error: {e}")
            return _resp(500, {"error": "tradeability-score failed", "detail": str(e)}, event=event)

    if path in ("/top-events-open-interest", "/top_events_open_interest"):
        qs = event.get("queryStringParameters") or {}
        try:
            data = _query_top_events_by_open_interest(qs.get("limit", 50))
            return _resp(200, data, event=event)
        except Exception as e:
            print(f"top-events-open-interest error: {e}")
            return _resp(500, {"error": "top-events-open-interest failed", "detail": str(e)}, event=event)

    if path in ("/top-events-volume", "/top_events_volume"):
        qs = event.get("queryStringParameters") or {}
        try:
            data = _query_top_events_by_volume(qs.get("limit", 50))
            return _resp(200, data, event=event)
        except Exception as e:
            print(f"top-events-volume error: {e}")
            return _resp(500, {"error": "top-events-volume failed", "detail": str(e)}, event=event)

    if path in ("/global-6h-deltas", "/global_6h_deltas"):
        qs = event.get("queryStringParameters") or {}
        try:
            data = _query_global_6h_deltas(qs.get("limit", 9))
            return _resp(200, data, event=event)
        except Exception as e:
            print(f"global-6h-deltas error: {e}")
            return _resp(500, {"error": "global-6h-deltas failed", "detail": str(e)}, event=event)

    if path in ("/markets/spread-blowouts", "/markets/spread_blowouts"):
        qs = event.get("queryStringParameters") or {}
        try:
            data = _query_spread_blowouts(qs.get("hours", 24), qs.get("limit", 100))
            return _resp(200, data, event=event)
        except Exception as e:
            print(f"markets/spread-blowouts error: {e}")
            return _resp(500, {"error": "markets/spread-blowouts failed", "detail": str(e)}, event=event)

    if path in ("/markets/expiring-soon", "/markets/expiring_soon"):
        qs = event.get("queryStringParameters") or {}
        try:
            data = _query_expiring_markets(qs.get("hours", 48), qs.get("limit", 50))
            return _resp(200, data, event=event)
        except Exception as e:
            print(f"markets/expiring-soon error: {e}")
            return _resp(500, {"error": "markets/expiring-soon failed", "detail": str(e)}, event=event)

    if path in ("/markets/mid-moves", "/markets/mid_moves"):
        qs = event.get("queryStringParameters") or {}
        try:
            data = _query_mid_moves(qs.get("hours", 24), qs.get("limit", 100))
            return _resp(200, data, event=event)
        except Exception as e:
            print(f"markets/mid-moves error: {e}")
            return _resp(500, {"error": "markets/mid-moves failed", "detail": str(e)}, event=event)

    if path in ("/market-movers", "/market_movers"):
        qs = event.get("queryStringParameters") or {}
        try:
            data = _query_market_movers(qs.get("limit", 100), qs.get("min_diff", 25))
            return _resp(200, data, event=event)
        except Exception as e:
            print(f"market-movers error: {e}")
            return _resp(500, {"error": "market-movers failed", "detail": str(e)}, event=event)
    if path in ("/vol/index/global", "/vol_index_global"):
        qs = event.get("queryStringParameters") or {}
        try:
            data = _query_global_vol_index(
                qs.get("points", 50),
                qs.get("min_open_interest", 100),
            )
            return _resp(200, data, event=event)
        except Exception as e:
            print(f"global-vol-index error: {e}")
            return _resp(500, {"error": "global-vol-index failed", "detail": str(e)}, event=event)
    return _resp(404, {"error": "not found", "path": path})
