"""
Refresh kalshi.market_move_stats from snapshot data.

Called at the end of the hourly export run so the stats table
stays in sync with the latest market_snapshot_markets data.
"""

import logging

logger = logging.getLogger(__name__)

# ── DDL (idempotent) ────────────────────────────────────────

_ENSURE_TABLE = """
CREATE TABLE IF NOT EXISTS kalshi.market_move_stats (
    market_ticker    TEXT        PRIMARY KEY,
    avg_move_24h     NUMERIC,
    avg_move_7d      NUMERIC,
    std_move_24h     NUMERIC,
    volatility_score NUMERIC,
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_move_stats_vol_score
    ON kalshi.market_move_stats (volatility_score DESC NULLS LAST);
"""

# ── Populate / refresh (UPSERT) ─────────────────────────────
# Only scans the last 7 days of snapshots to stay fast on EC2.

_REFRESH = """
INSERT INTO kalshi.market_move_stats
    (market_ticker, avg_move_24h, avg_move_7d, std_move_24h, volatility_score, updated_at)
WITH snap_times AS (
    SELECT DISTINCT snap_ts
    FROM   kalshi.market_snapshot_markets
    WHERE  snap_ts > NOW() - INTERVAL '7 days'
),
consecutive AS (
    SELECT
        snap_ts,
        LAG(snap_ts) OVER (ORDER BY snap_ts) AS prev_snap_ts
    FROM snap_times
),
moves AS (
    SELECT
        c.snap_ts,
        n.market_ticker,
        ABS(n.mid - p.mid) AS abs_move
    FROM consecutive c
    JOIN kalshi.market_snapshot_markets n
        ON  n.snap_ts = c.snap_ts
        AND n.snap_ts > NOW() - INTERVAL '7 days'
    JOIN kalshi.market_snapshot_markets p
        ON  p.snap_ts = c.prev_snap_ts
        AND p.market_ticker = n.market_ticker
        AND p.snap_ts > NOW() - INTERVAL '7 days'
    WHERE c.prev_snap_ts IS NOT NULL
      AND n.mid IS NOT NULL
      AND p.mid IS NOT NULL
),
latest_ts AS (
    SELECT MAX(snap_ts) AS ts FROM snap_times
),
stats AS (
    SELECT
        m.market_ticker,
        AVG(CASE WHEN m.snap_ts > lt.ts - INTERVAL '24 hours'
                 THEN m.abs_move END)           AS avg_move_24h,
        AVG(m.abs_move)                         AS avg_move_7d,
        STDDEV_SAMP(CASE WHEN m.snap_ts > lt.ts - INTERVAL '24 hours'
                         THEN m.abs_move END)   AS std_move_24h
    FROM moves m
    CROSS JOIN latest_ts lt
    GROUP BY m.market_ticker
)
SELECT
    market_ticker,
    ROUND(avg_move_24h, 4),
    ROUND(avg_move_7d, 4),
    ROUND(std_move_24h, 4),
    ROUND(std_move_24h / NULLIF(avg_move_24h, 0), 4),
    NOW()
FROM stats
ON CONFLICT (market_ticker) DO UPDATE SET
    avg_move_24h     = EXCLUDED.avg_move_24h,
    avg_move_7d      = EXCLUDED.avg_move_7d,
    std_move_24h     = EXCLUDED.std_move_24h,
    volatility_score = EXCLUDED.volatility_score,
    updated_at       = NOW();
"""


def refresh_move_stats(cur):
    """Ensure the table exists, then recompute stats.

    Accepts an open cursor (inside a connection with autocommit=True
    or an explicit transaction).  Returns the number of rows upserted.
    """
    cur.execute(_ENSURE_TABLE)
    cur.execute(_REFRESH)
    n = cur.rowcount
    return n
