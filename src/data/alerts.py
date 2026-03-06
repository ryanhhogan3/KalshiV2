"""
Alert system for PredictionShift.

Manages the alerts table, evaluates trigger conditions against
the latest biggest-movers data, and dispatches email notifications
via AWS SES.

Tables live in the `alerts` schema.
"""

import logging
import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from src.data.db_connect import connect

logger = logging.getLogger(__name__)

# ── DDL (idempotent) ────────────────────────────────────────

_ENSURE_SCHEMA = """
CREATE SCHEMA IF NOT EXISTS alerts;

CREATE TABLE IF NOT EXISTS alerts.alerts (
    alert_id    BIGSERIAL       PRIMARY KEY,
    user_email  TEXT            NOT NULL,
    platform    TEXT            NOT NULL DEFAULT 'kalshi',   -- 'kalshi' | 'poly'
    market_key  TEXT            NOT NULL,                    -- market_ticker or condition_id
    alert_type  TEXT            NOT NULL,                    -- PRICE_MOVE | VOLATILITY_SPIKE | VOLUME_SURGE
    threshold   NUMERIC         NOT NULL,
    is_active   BOOLEAN         NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    last_triggered_at TIMESTAMPTZ NULL
);

CREATE INDEX IF NOT EXISTS idx_alerts_active
    ON alerts.alerts (is_active, platform)
    WHERE is_active = TRUE;

CREATE INDEX IF NOT EXISTS idx_alerts_email
    ON alerts.alerts (user_email);

-- Log of every triggered alert (audit trail + dedup)
CREATE TABLE IF NOT EXISTS alerts.alert_history (
    history_id      BIGSERIAL       PRIMARY KEY,
    alert_id        BIGINT          NOT NULL REFERENCES alerts.alerts(alert_id),
    triggered_at    TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    market_key      TEXT            NOT NULL,
    alert_type      TEXT            NOT NULL,
    actual_value    NUMERIC         NULL,
    threshold       NUMERIC         NOT NULL,
    sent            BOOLEAN         NOT NULL DEFAULT FALSE,
    error_message   TEXT            NULL
);

CREATE INDEX IF NOT EXISTS idx_alert_history_alert
    ON alerts.alert_history (alert_id, triggered_at DESC);
"""


def ensure_alerts_schema(cur):
    """Create alerts schema + tables if they don't exist."""
    cur.execute(_ENSURE_SCHEMA)
    logger.info("alerts schema ensured")


# ── Alert evaluation ────────────────────────────────────────

# Cooldown: don't re-trigger the same alert within this window
_COOLDOWN_INTERVAL = "1 hour"

_FETCH_ACTIVE_ALERTS = """
SELECT alert_id, user_email, platform, market_key, alert_type, threshold
FROM   alerts.alerts
WHERE  is_active = TRUE
  AND  (last_triggered_at IS NULL
        OR last_triggered_at < NOW() - INTERVAL '{cooldown}')
"""

_KALSHI_MOVERS_SQL = """
WITH snaps AS (
    SELECT DISTINCT snap_ts
    FROM   kalshi.market_snapshot_markets
),
latest AS (
    SELECT MAX(snap_ts) AS snap_ts FROM snaps
),
base AS (
    SELECT COALESCE(
        (SELECT snap_ts FROM snaps
         WHERE snap_ts <= (SELECT snap_ts FROM latest) - INTERVAL '1 hour'
         ORDER BY snap_ts DESC LIMIT 1),
        (SELECT snap_ts FROM snaps
         WHERE snap_ts < (SELECT snap_ts FROM latest)
         ORDER BY snap_ts DESC LIMIT 1)
    ) AS snap_ts
)
SELECT
    l.market_ticker                             AS market_key,
    om.title                                    AS title,
    l.mid                                       AS price_now,
    p.mid                                       AS price_prev,
    (l.mid - p.mid)                             AS move,
    ABS(l.mid - p.mid)                          AS abs_move,
    l.volume,
    COALESCE(ms.volatility_score, 0)            AS volatility_score
FROM kalshi.market_snapshot_markets l
JOIN base b ON TRUE
JOIN kalshi.market_snapshot_markets p
  ON p.snap_ts = b.snap_ts AND p.market_ticker = l.market_ticker
LEFT JOIN kalshi.market_move_stats ms
  ON ms.market_ticker = l.market_ticker
LEFT JOIN kalshi.open_markets om
  ON om.market_ticker = l.market_ticker
WHERE l.snap_ts = (SELECT snap_ts FROM latest)
  AND l.mid IS NOT NULL AND p.mid IS NOT NULL
"""

_POLY_MOVERS_SQL = """
WITH snaps AS (
    SELECT DISTINCT snap_ts
    FROM   polymarket.market_snapshot_markets
),
latest AS (
    SELECT MAX(snap_ts) AS snap_ts FROM snaps
),
base AS (
    SELECT COALESCE(
        (SELECT snap_ts FROM snaps
         WHERE snap_ts <= (SELECT snap_ts FROM latest) - INTERVAL '1 hour'
         ORDER BY snap_ts DESC LIMIT 1),
        (SELECT snap_ts FROM snaps
         WHERE snap_ts < (SELECT snap_ts FROM latest)
         ORDER BY snap_ts DESC LIMIT 1)
    ) AS snap_ts
)
SELECT
    l.condition_id                              AS market_key,
    om.question                                 AS title,
    l.outcome_yes_price                         AS price_now,
    p.outcome_yes_price                         AS price_prev,
    (l.outcome_yes_price - p.outcome_yes_price) AS move,
    ABS(l.outcome_yes_price - p.outcome_yes_price) AS abs_move,
    l.volume,
    COALESCE(ms.volatility_score, 0)            AS volatility_score
FROM polymarket.market_snapshot_markets l
JOIN base b ON TRUE
JOIN polymarket.market_snapshot_markets p
  ON p.snap_ts = b.snap_ts AND p.condition_id = l.condition_id
LEFT JOIN polymarket.market_move_stats ms
  ON ms.condition_id = l.condition_id
LEFT JOIN polymarket.open_markets om
  ON om.condition_id = l.condition_id
WHERE l.snap_ts = (SELECT snap_ts FROM latest)
  AND l.outcome_yes_price IS NOT NULL
  AND p.outcome_yes_price IS NOT NULL
"""


def _fetch_movers(cur, platform: str) -> Dict[str, Dict]:
    """Fetch latest movers for a platform, keyed by market_key."""
    sql = _KALSHI_MOVERS_SQL if platform == "kalshi" else _POLY_MOVERS_SQL
    cur.execute(sql)
    cols = ["market_key", "title", "price_now", "price_prev",
            "move", "abs_move", "volume", "volatility_score"]
    rows = cur.fetchall()
    result = {}
    for r in rows:
        d = dict(zip(cols, r))
        result[d["market_key"]] = d
    return result


def evaluate_alerts(cur) -> List[Dict[str, Any]]:
    """Check all active alerts against current movers.

    Returns a list of triggered alerts with full context.
    """
    ensure_alerts_schema(cur)

    # Fetch active alerts (respecting cooldown)
    cur.execute(_FETCH_ACTIVE_ALERTS.format(cooldown=_COOLDOWN_INTERVAL))
    alerts = cur.fetchall()
    alert_cols = ["alert_id", "user_email", "platform", "market_key",
                  "alert_type", "threshold"]

    if not alerts:
        logger.info("No active alerts to evaluate")
        return []

    # Group alerts by platform to minimize queries
    kalshi_alerts = []
    poly_alerts = []
    for row in alerts:
        d = dict(zip(alert_cols, row))
        if d["platform"] == "kalshi":
            kalshi_alerts.append(d)
        else:
            poly_alerts.append(d)

    # Fetch movers only for platforms that have alerts
    movers = {}
    if kalshi_alerts:
        movers["kalshi"] = _fetch_movers(cur, "kalshi")
    if poly_alerts:
        movers["poly"] = _fetch_movers(cur, "poly")

    triggered = []

    for alert in [dict(zip(alert_cols, r)) for r in alerts]:
        platform_movers = movers.get(alert["platform"], {})
        market = platform_movers.get(alert["market_key"])
        if not market:
            continue  # Market not in latest snapshot

        actual_value = None
        should_trigger = False

        if alert["alert_type"] == "PRICE_MOVE":
            actual_value = abs(float(market["abs_move"] or 0))
            should_trigger = actual_value >= float(alert["threshold"])

        elif alert["alert_type"] == "VOLATILITY_SPIKE":
            actual_value = float(market["volatility_score"] or 0)
            should_trigger = actual_value >= float(alert["threshold"])

        elif alert["alert_type"] == "VOLUME_SURGE":
            actual_value = float(market["volume"] or 0)
            should_trigger = actual_value >= float(alert["threshold"])

        if should_trigger:
            triggered.append({
                **alert,
                "actual_value": actual_value,
                "title": market.get("title"),
                "price_now": market.get("price_now"),
                "price_prev": market.get("price_prev"),
                "move": market.get("move"),
            })

            # Record in history + update cooldown timestamp
            cur.execute(
                """
                INSERT INTO alerts.alert_history
                    (alert_id, market_key, alert_type, actual_value, threshold)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING history_id
                """,
                (alert["alert_id"], alert["market_key"],
                 alert["alert_type"], actual_value, alert["threshold"]),
            )
            history_id = cur.fetchone()[0]

            cur.execute(
                "UPDATE alerts.alerts SET last_triggered_at = NOW() WHERE alert_id = %s",
                (alert["alert_id"],),
            )

            # Attach history_id for the email sender
            triggered[-1]["history_id"] = history_id

    logger.info("Evaluated %d alerts, %d triggered", len(alerts), len(triggered))
    return triggered


def mark_alert_sent(cur, history_id: int, success: bool, error_msg: Optional[str] = None):
    """Update the history row after attempting to send the notification."""
    cur.execute(
        """
        UPDATE alerts.alert_history
        SET sent = %s, error_message = %s
        WHERE history_id = %s
        """,
        (success, error_msg, history_id),
    )
