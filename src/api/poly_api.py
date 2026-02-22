import math
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from src.data.db_connect import connect


ALLOWED_ORIGINS = [
    "https://predictionshift.com",
    "http://localhost:5173",
    "http://localhost:3000",
    "https://api.predictionshift.com",
]

app = FastAPI(title="Polymarket Analytics API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# DB helper
# ---------------------------------------------------------------------------

def _run_query(sql: str):
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            if cur.description:
                return cur.fetchall()
            return []


# ---------------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------------

def _fetch_active_raw(scan: int = 5000):
    """Fetch a slice of active markets for in-Python processing."""
    scan = max(1, min(int(scan), 20000))
    sql = f"""
        SELECT
            condition_id, question, category,
            volume, volume_24hr, liquidity,
            outcome_yes_price, outcome_no_price,
            event_id, updated_at
        FROM polymarket.open_markets
        WHERE active = true
        ORDER BY updated_at DESC
        LIMIT {scan}
    """
    rows = _run_query(sql)
    cols = [
        "condition_id", "question", "category",
        "volume", "volume_24hr", "liquidity",
        "outcome_yes_price", "outcome_no_price",
        "event_id", "updated_at",
    ]
    return [dict(zip(cols, r)) for r in rows]


def _query_export_runs(limit: int = 20):
    limit = max(1, min(int(limit), 100))
    sql = f"""
        SELECT run_id, started_at, finished_at, source, host, n_markets, note
        FROM polymarket.export_runs
        ORDER BY run_id DESC
        LIMIT {limit}
    """
    rows = _run_query(sql)
    cols = ["run_id", "started_at", "finished_at", "source", "host", "n_markets", "note"]
    out = []
    for r in rows:
        d = dict(zip(cols, r))
        for ts_col in ("started_at", "finished_at"):
            if d.get(ts_col):
                d[ts_col] = d[ts_col].isoformat()
        out.append(d)
    return out


def _query_active_markets(limit: int = 25):
    """Most active markets by volume."""
    limit = max(1, min(int(limit), 200))
    sql = f"""
        SELECT
            condition_id, question, category,
            volume, volume_24hr, liquidity,
            outcome_yes_price, outcome_no_price,
            event_id, updated_at
        FROM polymarket.open_markets
        WHERE active = true
          AND volume IS NOT NULL
          AND volume > 0
        ORDER BY volume DESC
        LIMIT {limit}
    """
    rows = _run_query(sql)
    cols = [
        "condition_id", "question", "category",
        "volume", "volume_24hr", "liquidity",
        "outcome_yes_price", "outcome_no_price",
        "event_id", "updated_at",
    ]
    out = []
    for r in rows:
        d = dict(zip(cols, r))
        if d.get("updated_at"):
            d["updated_at"] = d["updated_at"].isoformat()
        for k in ("volume", "volume_24hr", "liquidity", "outcome_yes_price", "outcome_no_price"):
            if d.get(k) is not None:
                d[k] = float(d[k])
        out.append(d)
    return out


def _query_tradability_score(limit: int = 100, scan: int = 5000, min_liquidity: float = 0.0):
    """
    Polymarket tradability score: log(1+volume) + 0.7*log(1+liquidity).

    Analog to Kalshi's spread-based score; here liquidity depth replaces the
    inverse-spread signal because Polymarket is a CLOB — we don't have a
    simple bid/ask spread, but we do have on-chain liquidity depth.
    """
    limit = max(1, min(int(limit), 200))
    raw = _fetch_active_raw(scan)

    out = []
    for m in raw:
        vol = float(m.get("volume") or 0)
        liq = float(m.get("liquidity") or 0)
        if liq < min_liquidity:
            continue

        score = math.log1p(vol) + 0.7 * math.log1p(liq)

        out.append({
            "condition_id": m["condition_id"],
            "question": m.get("question"),
            "category": m.get("category"),
            "volume": vol,
            "liquidity": liq,
            "outcome_yes_price": float(m["outcome_yes_price"]) if m.get("outcome_yes_price") is not None else None,
            "tradability_score": round(score, 4),
            "updated_at": m["updated_at"].isoformat() if m.get("updated_at") else None,
        })

    out.sort(key=lambda x: x["tradability_score"], reverse=True)
    return out[:limit]


def _query_market_heat(limit: int = 200, scan: int = 2000):
    """
    Churn rate = volume_24hr / volume.

    High churn = recent activity is a large fraction of total volume → hot market.
    """
    limit = max(1, min(int(limit), 500))
    raw = _fetch_active_raw(scan)

    out = []
    for m in raw:
        vol = float(m.get("volume") or 0)
        vol24 = m.get("volume_24hr")
        if vol24 is None or vol <= 0:
            continue
        vol24 = float(vol24)
        if vol24 <= 0:
            continue

        churn = round(vol24 / vol, 4)
        out.append({
            "condition_id": m["condition_id"],
            "question": m.get("question"),
            "category": m.get("category"),
            "volume": vol,
            "volume_24hr": vol24,
            "churn_rate": churn,
        })

    out.sort(key=lambda x: x["churn_rate"], reverse=True)
    return out[:limit]


def _query_opportunity_gap(limit: int = 50, scan: int = 2000):
    """
    Markets with the most price uncertainty relative to their activity.

    Uncertainty = how close outcome_yes_price is to 0.5 (max uncertainty at 0.5).
    Score = uncertainty * log(1+volume) — high volume AND genuinely uncertain.

    Useful for surfacing markets worth trading vs. ones that are already priced in.
    """
    limit = max(1, min(int(limit), 200))
    raw = _fetch_active_raw(scan)

    out = []
    for m in raw:
        yes = m.get("outcome_yes_price")
        if yes is None:
            continue
        yes = float(yes)
        if yes <= 0.0 or yes >= 1.0:
            continue

        vol = float(m.get("volume") or 0)
        # Uncertainty peaks at 0.5 → score max; certainty at 0/1 → score 0
        uncertainty = 1.0 - abs(yes - 0.5) * 2.0  # maps [0,0.5,1] → [0,1,0]
        score = round(uncertainty * math.log1p(vol), 4)

        out.append({
            "condition_id": m["condition_id"],
            "question": m.get("question"),
            "category": m.get("category"),
            "volume": vol,
            "outcome_yes_price": yes,
            "uncertainty": round(uncertainty, 4),
            "opportunity_score": score,
        })

    out.sort(key=lambda x: x["opportunity_score"], reverse=True)
    return out[:limit]


def _query_market_screener(
    limit: int = 50,
    scan: int = 2000,
    min_volume: Optional[float] = None,
    min_liquidity: Optional[float] = None,
    category: Optional[str] = None,
    sort_by: str = "tradability_score",
):
    limit = max(1, min(int(limit), 200))
    scan = max(1, min(int(scan), 10000))
    raw = _fetch_active_raw(scan)

    def _f(v):
        try:
            return float(v) if v is not None else None
        except Exception:
            return None

    out = []
    for m in raw:
        vol = _f(m.get("volume")) or 0.0
        liq = _f(m.get("liquidity")) or 0.0
        vol24 = _f(m.get("volume_24hr"))
        yes = _f(m.get("outcome_yes_price"))

        if min_volume is not None and vol < min_volume:
            continue
        if min_liquidity is not None and liq < min_liquidity:
            continue
        if category is not None and (m.get("category") or "").lower() != category.lower():
            continue

        tradability_score = math.log1p(vol) + 0.7 * math.log1p(liq)
        churn = round(vol24 / vol, 4) if vol24 and vol > 0 else None
        uncertainty = (1.0 - abs(yes - 0.5) * 2.0) if yes is not None and 0 < yes < 1 else None

        out.append({
            "condition_id": m["condition_id"],
            "question": m.get("question"),
            "category": m.get("category"),
            "volume": vol,
            "volume_24hr": vol24,
            "liquidity": liq,
            "outcome_yes_price": yes,
            "tradability_score": round(tradability_score, 4),
            "churn_rate": churn,
            "uncertainty": round(uncertainty, 4) if uncertainty is not None else None,
            "updated_at": m["updated_at"].isoformat() if m.get("updated_at") else None,
        })

    sort_by = (sort_by or "tradability_score").lower()
    sort_map = {
        "tradability_score": (lambda x: x["tradability_score"], True),
        "volume":            (lambda x: x["volume"], True),
        "volume_24hr":       (lambda x: x.get("volume_24hr") or 0.0, True),
        "liquidity":         (lambda x: x["liquidity"], True),
        "churn":             (lambda x: x.get("churn_rate") or 0.0, True),
        "churn_rate":        (lambda x: x.get("churn_rate") or 0.0, True),
        "uncertainty":       (lambda x: x.get("uncertainty") or 0.0, True),
    }
    key_fn, reverse = sort_map.get(sort_by, (lambda x: x["tradability_score"], True))
    out.sort(key=key_fn, reverse=reverse)
    return out[:limit]


def _query_top_events_by_volume(limit: int = 50):
    limit = max(1, min(int(limit), 200))
    sql = f"""
        WITH latest AS (
            SELECT MAX(snap_ts) AS snap_ts FROM polymarket.market_snapshot_events
        )
        SELECT
            s.event_id,
            e.event_title,
            e.category,
            s.n_markets,
            s.total_volume,
            s.total_liquidity,
            s.avg_yes_price
        FROM polymarket.market_snapshot_events s
        JOIN latest l USING (snap_ts)
        LEFT JOIN polymarket.events e USING (event_id)
        ORDER BY s.total_volume DESC
        LIMIT {limit}
    """
    rows = _run_query(sql)
    cols = ["event_id", "event_title", "category", "n_markets",
            "total_volume", "total_liquidity", "avg_yes_price"]
    out = []
    for r in rows:
        d = dict(zip(cols, r))
        for k in ("n_markets", "total_volume", "total_liquidity", "avg_yes_price"):
            if d.get(k) is not None:
                d[k] = float(d[k])
        out.append(d)
    return out


def _query_top_events_by_liquidity(limit: int = 50):
    limit = max(1, min(int(limit), 200))
    sql = f"""
        WITH latest AS (
            SELECT MAX(snap_ts) AS snap_ts FROM polymarket.market_snapshot_events
        )
        SELECT
            s.event_id,
            e.event_title,
            e.category,
            s.n_markets,
            s.total_volume,
            s.total_liquidity,
            s.avg_yes_price
        FROM polymarket.market_snapshot_events s
        JOIN latest l USING (snap_ts)
        LEFT JOIN polymarket.events e USING (event_id)
        ORDER BY s.total_liquidity DESC
        LIMIT {limit}
    """
    rows = _run_query(sql)
    cols = ["event_id", "event_title", "category", "n_markets",
            "total_volume", "total_liquidity", "avg_yes_price"]
    out = []
    for r in rows:
        d = dict(zip(cols, r))
        for k in ("n_markets", "total_volume", "total_liquidity", "avg_yes_price"):
            if d.get(k) is not None:
                d[k] = float(d[k])
        out.append(d)
    return out


def _query_global_snapshot():
    """Latest global snapshot stats."""
    sql = """
        SELECT
            snap_ts,
            n_active,
            n_priced,
            total_volume,
            total_volume_24hr,
            total_liquidity,
            avg_yes_price
        FROM polymarket.market_snapshot_global
        ORDER BY snap_ts DESC
        LIMIT 1
    """
    rows = _run_query(sql)
    if not rows:
        return None
    cols = ["snap_ts", "n_active", "n_priced", "total_volume",
            "total_volume_24hr", "total_liquidity", "avg_yes_price"]
    d = dict(zip(cols, rows[0]))
    if d.get("snap_ts"):
        d["snap_ts"] = d["snap_ts"].isoformat()
    for k in ("total_volume", "total_volume_24hr", "total_liquidity", "avg_yes_price"):
        if d.get(k) is not None:
            d[k] = float(d[k])
    return d


def _query_global_deltas(limit: int = 9):
    """Recent run-over-run deltas in global volume, liquidity and pricing."""
    limit = max(1, min(int(limit), 200))
    sql = f"""
        WITH g AS (
            SELECT
                snap_ts,
                total_volume,
                total_volume_24hr,
                total_liquidity,
                n_priced,
                avg_yes_price,
                LAG(total_volume)      OVER (ORDER BY snap_ts) AS prev_volume,
                LAG(total_volume_24hr) OVER (ORDER BY snap_ts) AS prev_vol24,
                LAG(total_liquidity)   OVER (ORDER BY snap_ts) AS prev_liq,
                LAG(n_priced)          OVER (ORDER BY snap_ts) AS prev_priced
            FROM polymarket.market_snapshot_global
        )
        SELECT
            snap_ts,
            total_volume - prev_volume      AS d_volume,
            total_volume_24hr - prev_vol24  AS d_volume_24hr,
            total_liquidity - prev_liq      AS d_liquidity,
            n_priced - prev_priced          AS d_priced
        FROM g
        WHERE prev_volume IS NOT NULL
        ORDER BY snap_ts DESC
        LIMIT {limit}
    """
    rows = _run_query(sql)
    cols = ["snap_ts", "d_volume", "d_volume_24hr", "d_liquidity", "d_priced"]
    out = []
    for r in rows:
        d = dict(zip(cols, r))
        if d.get("snap_ts"):
            d["snap_ts"] = d["snap_ts"].isoformat()
        for k in ("d_volume", "d_volume_24hr", "d_liquidity", "d_priced"):
            if d.get(k) is not None:
                d[k] = float(d[k])
        out.append(d)
    return out


def _query_expiring_markets(hours: int = 48, limit: int = 50):
    """Active markets expiring within N hours, by volume descending."""
    hours = max(1, min(int(hours), 336))
    limit = max(1, min(int(limit), 200))
    sql = f"""
        SELECT
            condition_id, question, category,
            end_date, volume, volume_24hr, liquidity,
            outcome_yes_price, event_id
        FROM polymarket.open_markets
        WHERE active = true
          AND end_date IS NOT NULL
          AND end_date <= NOW() + INTERVAL '{hours} hours'
        ORDER BY volume DESC NULLS LAST
        LIMIT {limit}
    """
    rows = _run_query(sql)
    cols = ["condition_id", "question", "category", "end_date",
            "volume", "volume_24hr", "liquidity", "outcome_yes_price", "event_id"]
    out = []
    for r in rows:
        d = dict(zip(cols, r))
        if d.get("end_date"):
            d["end_date"] = d["end_date"].isoformat()
        for k in ("volume", "volume_24hr", "liquidity", "outcome_yes_price"):
            if d.get(k) is not None:
                d[k] = float(d[k])
        out.append(d)
    return out


def _query_mid_moves(hours: int = 24, limit: int = 100):
    """
    Markets with the largest price moves over the lookback window.
    Requires ENABLE_MARKET_SNAPSHOT=1 to have data in market_snapshot_markets.
    """
    hours = max(1, min(int(hours), 168))
    limit = max(1, min(int(limit), 500))
    sql = f"""
        WITH snaps AS (
            SELECT DISTINCT snap_ts FROM polymarket.market_snapshot_markets
        ),
        latest AS (
            SELECT MAX(snap_ts) AS snap_ts FROM snaps
        ),
        base AS (
            SELECT COALESCE(
                (
                    SELECT snap_ts FROM snaps
                    WHERE snap_ts <= (SELECT snap_ts FROM latest) - INTERVAL '{hours} hours'
                    ORDER BY snap_ts DESC LIMIT 1
                ),
                (
                    SELECT snap_ts FROM snaps
                    WHERE snap_ts < (SELECT snap_ts FROM latest)
                    ORDER BY snap_ts DESC LIMIT 1
                )
            ) AS snap_ts
        )
        SELECT
            l.condition_id,
            om.question,
            om.category,
            l.event_id,
            l.outcome_yes_price   AS price_now,
            p.outcome_yes_price   AS price_prev,
            ABS(l.outcome_yes_price - p.outcome_yes_price) AS price_diff,
            l.volume,
            l.liquidity
        FROM polymarket.market_snapshot_markets l
        JOIN base b ON TRUE
        JOIN polymarket.market_snapshot_markets p
          ON p.snap_ts = b.snap_ts
         AND p.condition_id = l.condition_id
        LEFT JOIN polymarket.open_markets om
          ON om.condition_id = l.condition_id
        WHERE l.snap_ts = (SELECT snap_ts FROM latest)
          AND l.outcome_yes_price IS NOT NULL
          AND p.outcome_yes_price IS NOT NULL
        ORDER BY price_diff DESC
        LIMIT {limit}
    """
    rows = _run_query(sql)
    cols = [
        "condition_id", "question", "category", "event_id",
        "price_now", "price_prev", "price_diff", "volume", "liquidity",
    ]
    out = []
    for r in rows:
        d = dict(zip(cols, r))
        for k in ("price_now", "price_prev", "price_diff", "volume", "liquidity"):
            if d.get(k) is not None:
                d[k] = float(d[k])
        out.append(d)
    return out


def _query_top_changes_24h(metric: str = "volume", limit: int = 50, min_prev_value: float = 0.0):
    """Top markets by change from oldest->newest snapshot inside the latest 24h window."""
    limit = max(1, min(int(limit), 500))
    try:
        min_prev_value = float(min_prev_value)
    except Exception:
        min_prev_value = 0.0

    metric_map = {
        "volume": "volume",
        "volume_24hr": "volume_24hr",
        "liquidity": "liquidity",
        "outcome_yes_price": "outcome_yes_price",
        "yes_price": "outcome_yes_price",
    }
    metric_key = (metric or "volume").lower()
    metric_col = metric_map.get(metric_key)
    if metric_col is None:
        raise ValueError("Unsupported metric. Use one of: volume, volume_24hr, liquidity, outcome_yes_price")

    sql = f"""
        WITH latest AS (
            SELECT MAX(snap_ts) AS snap_ts
            FROM polymarket.market_snapshot_markets
        ),
        window_snaps AS (
            SELECT DISTINCT m.snap_ts
            FROM polymarket.market_snapshot_markets m
            JOIN latest l ON TRUE
            WHERE m.snap_ts >= l.snap_ts - INTERVAL '24 hours'
              AND m.snap_ts <= l.snap_ts
        ),
        base AS (
            SELECT MIN(snap_ts) AS snap_ts
            FROM window_snaps
        )
        SELECT
            l.condition_id,
            om.question,
            om.category,
            l.event_id,
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
                        NULL::numeric AS prev_open_interest,
                        NULL::numeric AS current_open_interest,
                        NULL::numeric AS d_open_interest,
            (SELECT snap_ts FROM latest) AS latest_snap_ts,
                        (SELECT snap_ts FROM base) AS base_snap_ts,
                        EXTRACT(EPOCH FROM ((SELECT snap_ts FROM latest) - (SELECT snap_ts FROM base))) / 3600.0 AS window_hours
        FROM polymarket.market_snapshot_markets l
        JOIN base b ON TRUE
        JOIN polymarket.market_snapshot_markets p
          ON p.snap_ts = b.snap_ts
         AND p.condition_id = l.condition_id
        LEFT JOIN polymarket.open_markets om
          ON om.condition_id = l.condition_id
        WHERE l.snap_ts = (SELECT snap_ts FROM latest)
                    AND p.snap_ts = (SELECT snap_ts FROM base)
          AND p.{metric_col} IS NOT NULL
          AND l.{metric_col} IS NOT NULL
          AND ABS(p.{metric_col}) >= {min_prev_value}
        ORDER BY ABS(l.{metric_col} - p.{metric_col}) DESC
                             , ABS(COALESCE(l.volume - p.volume, 0)) DESC
                             , l.volume DESC NULLS LAST
        LIMIT {limit}
    """

    rows = _run_query(sql)
    cols = [
        "condition_id",
        "question",
        "category",
        "event_id",
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


def _query_vol_index_global(points: int = 50, min_liquidity: float = 100.0):
    """
    Polymarket realized volatility index.

    Uses log-odds returns on outcome_yes_price across consecutive snapshots,
    weighted by on-chain liquidity depth. Requires ENABLE_MARKET_SNAPSHOT=1.
    """
    try:
        points = max(1, min(int(points), 200))
        min_liquidity = max(0.0, float(min_liquidity))
    except Exception:
        points, min_liquidity = 50, 100.0

    sql = f"""
        WITH snaps AS (
            SELECT DISTINCT snap_ts
            FROM polymarket.market_snapshot_markets
            ORDER BY snap_ts DESC
            LIMIT {points + 1}
        ),
        ordered AS (
            SELECT snap_ts, ROW_NUMBER() OVER (ORDER BY snap_ts) AS rn FROM snaps
        ),
        pairs AS (
            SELECT
                cur.snap_ts,
                prev.snap_ts AS prev_snap_ts,
                EXTRACT(EPOCH FROM (cur.snap_ts - prev.snap_ts)) / 3600.0 AS dt_hours
            FROM ordered cur
            JOIN ordered prev ON prev.rn = cur.rn - 1
        ),
        joined AS (
            SELECT
                p.snap_ts,
                p.dt_hours,
                m_now.outcome_yes_price AS price_now,
                m_prev.outcome_yes_price AS price_prev,
                m_now.liquidity
            FROM pairs p
            JOIN polymarket.market_snapshot_markets m_now ON m_now.snap_ts = p.snap_ts
            JOIN polymarket.market_snapshot_markets m_prev
              ON m_prev.snap_ts = p.prev_snap_ts
             AND m_prev.condition_id = m_now.condition_id
            WHERE m_now.liquidity >= {min_liquidity}
              AND m_now.outcome_yes_price > 0.01 AND m_now.outcome_yes_price < 0.99
              AND m_prev.outcome_yes_price > 0.01 AND m_prev.outcome_yes_price < 0.99
        ),
        returns AS (
            SELECT
                snap_ts, dt_hours,
                LN(price_now / (1 - price_now)) - LN(price_prev / (1 - price_prev)) AS r,
                liquidity
            FROM joined WHERE dt_hours > 0
        ),
        stats AS (
            SELECT snap_ts, dt_hours,
                   SUM(liquidity * r * r) / NULLIF(SUM(liquidity), 0) AS rv
            FROM returns
            GROUP BY snap_ts, dt_hours
        )
        SELECT snap_ts, dt_hours, rv FROM stats ORDER BY snap_ts DESC LIMIT {points}
    """

    rows = _run_query(sql)
    HOURS_PER_YEAR = 24.0 * 365.0
    series = []
    for snap_ts, dt_hours, rv in rows:
        if rv is None or dt_hours is None or float(dt_hours) <= 0:
            continue
        try:
            rv, dt_hours = float(rv), float(dt_hours)
        except Exception:
            continue
        if rv < 0:
            continue
        annualised_var = rv * (HOURS_PER_YEAR / dt_hours)
        vol_index = 100.0 * math.sqrt(annualised_var) if annualised_var > 0 else 0.0
        series.append({
            "snap_ts": snap_ts.isoformat() if hasattr(snap_ts, "isoformat") else str(snap_ts),
            "dt_hours": round(dt_hours, 6),
            "realised_variance": round(rv, 10),
            "vol_index": round(vol_index, 4),
        })

    return {"points": len(series), "min_liquidity": min_liquidity, "series": series}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

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
def active_markets(limit: int = Query(25, ge=1, le=200)):
    try:
        return _query_active_markets(limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"active-markets failed: {e}")


@app.get("/tradeability-score")
@app.get("/tradability_score", include_in_schema=False)
def tradability_score(
    limit: int = Query(100, ge=1, le=200),
    scan: int = Query(5000, ge=1, le=20000),
    min_liquidity: float = Query(0.0, ge=0),
):
    try:
        return _query_tradability_score(limit, scan, min_liquidity)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"tradeability-score failed: {e}")


@app.get("/market-heat")
@app.get("/market_heat", include_in_schema=False)
def market_heat(
    limit: int = Query(200, ge=1, le=500),
    scan: int = Query(2000, ge=1, le=10000),
):
    try:
        return _query_market_heat(limit, scan)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"market-heat failed: {e}")


@app.get("/opportunity-gap")
@app.get("/opportunity_gap", include_in_schema=False)
def opportunity_gap(
    limit: int = Query(50, ge=1, le=200),
    scan: int = Query(2000, ge=1, le=10000),
):
    try:
        return _query_opportunity_gap(limit, scan)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"opportunity-gap failed: {e}")


@app.get("/markets/screener")
def market_screener(
    limit: int = Query(50, ge=1, le=200),
    scan: int = Query(2000, ge=1, le=10000),
    min_volume: Optional[float] = Query(None),
    min_liquidity: Optional[float] = Query(None),
    category: Optional[str] = Query(None),
    sort_by: str = Query("tradability_score"),
):
    try:
        return _query_market_screener(
            limit=limit,
            scan=scan,
            min_volume=min_volume,
            min_liquidity=min_liquidity,
            category=category,
            sort_by=sort_by,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"markets/screener failed: {e}")


@app.get("/top-events-volume")
@app.get("/top_events_volume", include_in_schema=False)
def top_events_volume(limit: int = Query(50, ge=1, le=200)):
    try:
        return _query_top_events_by_volume(limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"top-events-volume failed: {e}")


@app.get("/top-events-liquidity")
@app.get("/top_events_liquidity", include_in_schema=False)
def top_events_liquidity(limit: int = Query(50, ge=1, le=200)):
    try:
        return _query_top_events_by_liquidity(limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"top-events-liquidity failed: {e}")


@app.get("/global-snapshot")
@app.get("/global_snapshot", include_in_schema=False)
def global_snapshot():
    try:
        result = _query_global_snapshot()
        if result is None:
            raise HTTPException(status_code=404, detail="No snapshot data available yet")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"global-snapshot failed: {e}")


@app.get("/global-deltas")
@app.get("/global_deltas", include_in_schema=False)
def global_deltas(limit: int = Query(9, ge=1, le=200)):
    try:
        return _query_global_deltas(limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"global-deltas failed: {e}")


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
    min_liquidity: float = Query(100.0, ge=0),
):
    try:
        return _query_vol_index_global(points, min_liquidity)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"vol-index-global failed: {e}")
