"""
Polymarket → Postgres exporter.

Mirrors src/data/db_export_open_markets.py for Kalshi.

Run standalone:
    python -m src.data.db_export_poly_markets

Or as a cron / Docker command:
    EXPORT_SOURCE=ec2 python -m src.data.db_export_poly_markets

Required env vars (same as Kalshi exporter):
    DB_HOST, DB_PORT, DB_PASS
Optional:
    DB_NAME       (default: shift)
    DB_USER       (default: shift_user)
    DB_SSLMODE    (default: disable)
    EXPORT_SOURCE (default: unknown)
    ENABLE_MARKET_SNAPSHOT  set to "1" to write per-market snapshot rows
"""

import json
import os
import socket
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.data.db_connect import connect
from src.data.poly.poly_fetch import iter_flat_market_batches

# ---------------------------------------------------------------------------
# DDL — creates schema + tables if they don't already exist
# ---------------------------------------------------------------------------

DDL = """
create schema if not exists polymarket;

-- Current-state cache: one row per market (condition_id is the stable PK)
create table if not exists polymarket.open_markets (
  condition_id        text primary key,
  market_id           text null,
  slug                text null,
  question            text null,
  category            text null,
  active              boolean null,
  closed              boolean null,
  enable_order_book   boolean null,
  start_date          timestamptz null,
  end_date            timestamptz null,
  liquidity           numeric(18,4) null,
  volume              numeric(18,4) null,
  volume_24hr         numeric(18,4) null,
  outcome_yes_price   numeric(8,6) null,   -- 0.000000–1.000000 for binary markets
  outcome_no_price    numeric(8,6) null,
  outcome_prices      jsonb null,          -- full outcome→price map (multi-outcome safe)
  event_id            text null,
  raw                 jsonb not null,
  updated_at          timestamptz not null default now()
);

create index if not exists om_event_id on polymarket.open_markets(event_id);
create index if not exists om_category on polymarket.open_markets(category);

-- Event-level current state (analog to series in Kalshi)
create table if not exists polymarket.events (
  event_id    text primary key,
  event_slug  text null,
  event_title text null,
  category    text null,
  start_date  timestamptz null,
  end_date    timestamptz null,
  raw         jsonb not null,
  updated_at  timestamptz not null default now()
);

-- Operational audit log: one row per exporter run
create table if not exists polymarket.export_runs (
  run_id      bigserial primary key,
  started_at  timestamptz not null default now(),
  finished_at timestamptz null,
  source      text not null,
  host        text not null,
  n_markets   integer not null,
  note        text null
);

-- Global time-series snapshot: one row per run
create table if not exists polymarket.market_snapshot_global (
  run_id              bigint primary key,
  snap_ts             timestamptz not null,
  n_active            bigint not null,
  n_priced            bigint not null,
  total_volume        numeric(24,4) not null,
  total_volume_24hr   numeric(24,4) not null,
  total_liquidity     numeric(24,4) not null,
  avg_yes_price       numeric(8,6) null
);

create index if not exists msg_snap_ts on polymarket.market_snapshot_global(snap_ts);

-- Event-level time-series snapshot (analog to market_snapshot_series in Kalshi)
create table if not exists polymarket.market_snapshot_events (
  snap_ts         timestamptz not null,
  run_id          bigint not null,
  event_id        text not null,
  n_markets       int not null,
  total_volume    numeric(24,4) not null,
  total_liquidity numeric(24,4) not null,
  avg_yes_price   numeric(8,6) null,
  primary key (snap_ts, event_id)
);

create index if not exists mse_event_time on polymarket.market_snapshot_events(event_id, snap_ts);
create index if not exists mse_run_id on polymarket.market_snapshot_events(run_id);

-- Per-market time-series snapshot (optional, gated by ENABLE_MARKET_SNAPSHOT=1)
create table if not exists polymarket.market_snapshot_markets (
  snap_ts           timestamptz not null,
  run_id            bigint not null,
  condition_id      text not null,
  event_id          text null,
  end_date          timestamptz null,
  volume            numeric(18,4) null,
  volume_24hr       numeric(18,4) null,
  liquidity         numeric(18,4) null,
  outcome_yes_price numeric(8,6) null,
  outcome_no_price  numeric(8,6) null,
  primary key (snap_ts, condition_id)
);

create index if not exists msm_cid_time on polymarket.market_snapshot_markets(condition_id, snap_ts);
create index if not exists msm_run_id on polymarket.market_snapshot_markets(run_id);
"""

# ---------------------------------------------------------------------------
# DML
# ---------------------------------------------------------------------------

UPSERT_MARKET = """
insert into polymarket.open_markets (
  condition_id, market_id, slug, question, category,
  active, closed, enable_order_book,
  start_date, end_date,
  liquidity, volume, volume_24hr,
  outcome_yes_price, outcome_no_price, outcome_prices,
  event_id, raw, updated_at
)
values (
  %(condition_id)s, %(market_id)s, %(slug)s, %(question)s, %(category)s,
  %(active)s, %(closed)s, %(enable_order_book)s,
  %(start_date)s, %(end_date)s,
  %(liquidity)s, %(volume)s, %(volume_24hr)s,
  %(outcome_yes_price)s, %(outcome_no_price)s, %(outcome_prices)s,
  %(event_id)s, %(raw)s, now()
)
on conflict (condition_id) do update set
  market_id           = excluded.market_id,
  slug                = excluded.slug,
  question            = excluded.question,
  category            = excluded.category,
  active              = excluded.active,
  closed              = excluded.closed,
  enable_order_book   = excluded.enable_order_book,
  start_date          = excluded.start_date,
  end_date            = excluded.end_date,
  liquidity           = excluded.liquidity,
  volume              = excluded.volume,
  volume_24hr         = excluded.volume_24hr,
  outcome_yes_price   = excluded.outcome_yes_price,
  outcome_no_price    = excluded.outcome_no_price,
  outcome_prices      = excluded.outcome_prices,
  event_id            = excluded.event_id,
  raw                 = excluded.raw,
  updated_at          = now();
"""

UPSERT_EVENT = """
insert into polymarket.events (
  event_id, event_slug, event_title, category,
  start_date, end_date, raw, updated_at
)
values (
  %(event_id)s, %(event_slug)s, %(event_title)s, %(category)s,
  %(start_date)s, %(end_date)s, %(raw)s, now()
)
on conflict (event_id) do update set
  event_slug  = excluded.event_slug,
  event_title = excluded.event_title,
  category    = excluded.category,
  start_date  = excluded.start_date,
  end_date    = excluded.end_date,
  raw         = excluded.raw,
  updated_at  = now();
"""

INSERT_RUN = """
insert into polymarket.export_runs (source, host, n_markets, note)
values (%s, %s, %s, %s)
returning run_id;
"""


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _safe_numeric(v: Any) -> Optional[float]:
    """Cast string/float/int to float, return None on failure."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def normalize_market(m: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a flat market dict (from poly_fetch.flatten_markets_from_events)
    into a row dict ready for the UPSERT_MARKET parameterised query.
    """
    return {
        "condition_id":      m.get("condition_id"),
        "market_id":         m.get("market_id"),
        "slug":              m.get("slug"),
        "question":          m.get("question"),
        "category":          m.get("category"),
        "active":            m.get("active"),
        "closed":            m.get("closed"),
        "enable_order_book": m.get("enable_order_book"),
        "start_date":        m.get("start_date"),
        "end_date":          m.get("end_date"),
        "liquidity":         _safe_numeric(m.get("liquidity")),
        "volume":            _safe_numeric(m.get("volume")),
        "volume_24hr":       _safe_numeric(m.get("volume_24hr")),
        "outcome_yes_price": _safe_numeric(m.get("outcome_yes_price")),
        "outcome_no_price":  _safe_numeric(m.get("outcome_no_price")),
        "outcome_prices":    json.dumps(m.get("outcome_map")) if m.get("outcome_map") else None,
        "event_id":          m.get("event_id"),
        "raw":               json.dumps(m.get("raw") or m),
    }


def normalize_event(m: Dict[str, Any]) -> Dict[str, Any]:
    """Build an event upsert row from a flat market dict."""
    return {
        "event_id":    m.get("event_id"),
        "event_slug":  m.get("event_slug"),
        "event_title": m.get("event_title"),
        "category":    m.get("event_category"),
        "start_date":  m.get("event_start"),
        "end_date":    m.get("event_end"),
        # Store the event-level fields we have as the raw payload
        "raw": json.dumps(
            {
                "event_id":    m.get("event_id"),
                "event_slug":  m.get("event_slug"),
                "event_title": m.get("event_title"),
                "category":    m.get("event_category"),
                "startDate":   m.get("event_start"),
                "endDate":     m.get("event_end"),
            }
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    source = os.environ.get("EXPORT_SOURCE", "unknown")
    enable_market_snap = os.environ.get("ENABLE_MARKET_SNAPSHOT", "0") == "1"
    print("EXPORTER_VERSION=POLY_SNAPSHOTS_V1", __file__)
    host = socket.gethostname()
    snap_ts = datetime.now(timezone.utc)

    print("Starting Polymarket DB export (streaming by page)...")
    total_rows = 0

    with connect() as conn:
        with conn.cursor() as cur:
            # Ensure schema + tables exist
            cur.execute(DDL)

            cur.execute(
                INSERT_RUN,
                (source, host, 0, f"started {snap_ts.isoformat()}"),
            )
            run_id = cur.fetchone()[0]
            print(f"Started run_id={run_id}")

            seen_events: set = set()

            for batch in iter_flat_market_batches():
                if not batch:
                    continue

                market_rows = []
                event_rows = []

                for m in batch:
                    if not m.get("condition_id"):
                        # Skip markets without a stable on-chain identifier
                        print(f"[warn] Skipping market without condition_id: {m.get('market_id')}")
                        continue

                    market_rows.append(normalize_market(m))

                    ev_id = m.get("event_id")
                    if ev_id and ev_id not in seen_events:
                        seen_events.add(ev_id)
                        event_rows.append(normalize_event(m))

                if event_rows:
                    cur.executemany(UPSERT_EVENT, event_rows)

                if market_rows:
                    cur.executemany(UPSERT_MARKET, market_rows)
                    total_rows += len(market_rows)
                    print(f"Upserted batch of {len(market_rows)} markets (total={total_rows})")

            # Update run record with final count
            cur.execute(
                "update polymarket.export_runs set n_markets = %s, finished_at = now() where run_id = %s;",
                (total_rows, run_id),
            )

            # ---------------------------------------------------------------
            # Global snapshot
            # ---------------------------------------------------------------
            try:
                cur.execute(
                    """
                    insert into polymarket.market_snapshot_global (
                      run_id, snap_ts,
                      n_active, n_priced,
                      total_volume, total_volume_24hr, total_liquidity,
                      avg_yes_price
                    )
                    select
                      %s,
                      %s,
                      count(*) filter (where active = true)                       as n_active,
                      count(*) filter (where outcome_yes_price is not null)        as n_priced,
                      coalesce(sum(volume), 0)                                    as total_volume,
                      coalesce(sum(volume_24hr), 0)                               as total_volume_24hr,
                      coalesce(sum(liquidity), 0)                                 as total_liquidity,
                      avg(outcome_yes_price) filter (where outcome_yes_price is not null) as avg_yes_price
                    from polymarket.open_markets
                    where active = true
                    on conflict (run_id) do nothing;
                    """,
                    (run_id, snap_ts),
                )
                print("Inserted global snapshot rows:", cur.rowcount)

                # ---------------------------------------------------------------
                # Event-level snapshot
                # ---------------------------------------------------------------
                cur.execute(
                    """
                    insert into polymarket.market_snapshot_events (
                      snap_ts, run_id, event_id,
                      n_markets, total_volume, total_liquidity, avg_yes_price
                    )
                    select
                      %s,
                      %s,
                      event_id,
                      count(*)                                                    as n_markets,
                      coalesce(sum(volume), 0)                                    as total_volume,
                      coalesce(sum(liquidity), 0)                                 as total_liquidity,
                      avg(outcome_yes_price) filter (where outcome_yes_price is not null) as avg_yes_price
                    from polymarket.open_markets
                    where active = true
                      and event_id is not null
                    group by event_id
                    on conflict (snap_ts, event_id) do nothing;
                    """,
                    (snap_ts, run_id),
                )
                print("Inserted event snapshot rows:", cur.rowcount)

                # ---------------------------------------------------------------
                # Per-market snapshot (optional — set ENABLE_MARKET_SNAPSHOT=1)
                # ---------------------------------------------------------------
                if enable_market_snap:
                    cur.execute(
                        """
                        with pick as (
                          (select * from polymarket.open_markets where active = true
                           order by volume desc nulls last limit 20000)
                          union
                          (select * from polymarket.open_markets where active = true
                           order by liquidity desc nulls last limit 20000)
                          union
                          (select * from polymarket.open_markets where active = true
                           order by volume_24hr desc nulls last limit 20000)
                        )
                        insert into polymarket.market_snapshot_markets (
                          snap_ts, run_id, condition_id, event_id,
                          end_date, volume, volume_24hr, liquidity,
                          outcome_yes_price, outcome_no_price
                        )
                        select
                          %s, %s,
                          condition_id, event_id,
                          end_date, volume, volume_24hr, liquidity,
                          outcome_yes_price, outcome_no_price
                        from pick
                        on conflict (snap_ts, condition_id) do nothing;
                        """,
                        (snap_ts, run_id),
                    )
                    print("Inserted per-market snapshot rows:", cur.rowcount)

            except Exception as e:
                print("Error during snapshot export:", e)
                raise

            print(f"OK: upserted {total_rows} markets, {len(seen_events)} events, run_id={run_id}")


if __name__ == "__main__":
    main()
