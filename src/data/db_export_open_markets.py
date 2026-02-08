import json
import os
import socket
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

from src.data.overview.all_markets import OverviewAllMarkets
from src.data.db_connect import connect

DDL = """
create schema if not exists kalshi;

create table if not exists kalshi.open_markets (
  market_ticker text primary key,
  series_ticker text null,
  title text null,
  status text null,
  open_time timestamptz null,
  close_time timestamptz null,
  expiration_time timestamptz null,
  yes_bid integer null,
  yes_ask integer null,
  no_bid integer null,
  no_ask integer null,
  volume integer null,
  open_interest integer null,
  raw jsonb not null,
  updated_at timestamptz not null default now()
);

create table if not exists kalshi.export_runs (
  run_id bigserial primary key,
  started_at timestamptz not null default now(),
  finished_at timestamptz null,
  source text not null,
  host text not null,
  n_markets integer not null,
  note text null
);
 
create table if not exists kalshi.open_markets_hist (
  snap_id bigserial primary key,
  snap_ts timestamptz not null,
  run_id bigint not null references kalshi.export_runs(run_id),
  market_ticker text not null,
  series_ticker text null,
  expiration_time timestamptz null,
  yes_bid integer null,
  yes_ask integer null,
  volume integer null,
  open_interest integer null,
  updated_at timestamptz not null
);

create index if not exists omh_run on kalshi.open_markets_hist(run_id);
create index if not exists omh_ticker_ts on kalshi.open_markets_hist(market_ticker, snap_ts);

create table if not exists kalshi.market_snapshot_global (
  run_id bigint primary key,
  snap_ts timestamptz not null,
  n_active bigint not null,
  n_priced bigint not null,
  total_volume bigint not null,
  total_open_interest bigint not null,
  avg_spread_ticks numeric null,
  n_wide_spread bigint not null
);

create index if not exists msg_snap_ts on kalshi.market_snapshot_global(snap_ts);

create table if not exists kalshi.market_snapshot_series (
  snap_ts timestamptz not null,
  run_id bigint not null,
  series_ticker text not null,
  n_markets int not null,
  total_volume bigint not null,
  total_open_interest bigint not null,
  avg_spread_ticks numeric null,
  primary key (snap_ts, series_ticker)
);

create index if not exists mss_series_time on kalshi.market_snapshot_series(series_ticker, snap_ts);
create index if not exists mss_run_id on kalshi.market_snapshot_series(run_id);

create table if not exists kalshi.market_snapshot_markets (
  snap_ts timestamptz not null,
  run_id bigint not null,
  market_ticker text not null,
  series_ticker text null,
  expiration_time timestamptz null,
  yes_bid int null,
  yes_ask int null,
  volume int null,
  open_interest int null,
  spread_ticks int null,
  mid numeric null,
  primary key (snap_ts, market_ticker)
);

create index if not exists msm_ticker_time on kalshi.market_snapshot_markets(market_ticker, snap_ts);
create index if not exists msm_series_time on kalshi.market_snapshot_markets(series_ticker, snap_ts);
create index if not exists msm_run_id on kalshi.market_snapshot_markets(run_id);
"""

UPSERT = """
insert into kalshi.open_markets (
  market_ticker, series_ticker, title, status,
  open_time, close_time, expiration_time,
  yes_bid, yes_ask, no_bid, no_ask,
  volume, open_interest,
  raw, updated_at
)
values (
  %(market_ticker)s, %(series_ticker)s, %(title)s, %(status)s,
  %(open_time)s, %(close_time)s, %(expiration_time)s,
  %(yes_bid)s, %(yes_ask)s, %(no_bid)s, %(no_ask)s,
  %(volume)s, %(open_interest)s,
  %(raw)s, now()
)
on conflict (market_ticker) do update set
  series_ticker = excluded.series_ticker,
  title = excluded.title,
  status = excluded.status,
  open_time = excluded.open_time,
  close_time = excluded.close_time,
  expiration_time = excluded.expiration_time,
  yes_bid = excluded.yes_bid,
  yes_ask = excluded.yes_ask,
  no_bid = excluded.no_bid,
  no_ask = excluded.no_ask,
  volume = excluded.volume,
  open_interest = excluded.open_interest,
  raw = excluded.raw,
  updated_at = now();
"""

INSERT_RUN = """
insert into kalshi.export_runs (source, host, n_markets, note)
values (%s, %s, %s, %s)
returning run_id;
"""

FINISH_RUN = """
update kalshi.export_runs set finished_at = now()
where run_id = %s;
"""

def iter_open_markets_batches(limit: int = 1000) -> Iterable[List[Dict[str, Any]]]:
    """Yield pages of open markets from Kalshi.

    This uses OverviewAllMarkets.iter_open_markets to fetch one page at a
    time so that callers can process and drop each batch, keeping memory
    usage bounded.
    """

    overview = OverviewAllMarkets()
    for batch in overview.iter_open_markets(limit=limit):
        yield batch


def _parse_ts(v):
    # Accept ISO strings or None. Keep it simple.
    return v if v else None


def normalize_market(m: Dict[str, Any]) -> Dict[str, Any]:
    # Adapt these mappings to your Kalshi response shape
    # Keep raw full JSON for safety.
    return {
        "market_ticker": m.get("ticker") or m.get("market_ticker"),
        "series_ticker": m.get("series_ticker"),
        "title": m.get("title"),
        "status": m.get("status"),
        "open_time": _parse_ts(m.get("open_time")),
        "close_time": _parse_ts(m.get("close_time")),
        "expiration_time": _parse_ts(m.get("expiration_time")),
        "yes_bid": m.get("yes_bid"),
        "yes_ask": m.get("yes_ask"),
        "no_bid": m.get("no_bid"),
        "no_ask": m.get("no_ask"),
        "volume": m.get("volume"),
        "open_interest": m.get("open_interest"),
        "raw": json.dumps(m),
    }


def main():
    source = os.environ.get("EXPORT_SOURCE", "unknown")  # local|ec2
    print("EXPORTER_VERSION=SNAPSHOTS_V1", __file__)
    host = socket.gethostname()
    snap_ts = datetime.now(timezone.utc)
    enable_market_snap = os.environ.get("ENABLE_MARKET_SNAPSHOT", "0") == "1"

    print("Starting DB export of open markets (streaming by page)...")

    total_rows = 0

    print("Connecting to database and upserting in batches...")
    with connect() as conn:
      with conn.cursor() as cur:
        cur.execute(DDL)

        # We don't yet know how many markets we'll see; start with 0
        # and update n_markets once the streaming export finishes.
        cur.execute(
          INSERT_RUN,
          (source, host, 0, f"started {datetime.now(timezone.utc).isoformat()}"),
        )
        run_id = cur.fetchone()[0]

        # Stream Kalshi pages and upsert one batch at a time to keep
        # memory usage low.
        for batch in iter_open_markets_batches():
          if not batch:
            continue

          rows = [normalize_market(m) for m in batch]

          # Hard fail if any ticker is missing in this page.
          missing = [r for r in rows if not r["market_ticker"]]
          if missing:
            raise ValueError(
              f"{len(missing)} rows missing market_ticker in current batch"
            )

          cur.executemany(UPSERT, rows)
          total_rows += len(rows)
          print(f"Upserted batch of {len(rows)} markets (total={total_rows})")

        # Mark the run as finished and update the final market count.
        cur.execute(
          "update kalshi.export_runs set n_markets = %s, finished_at = now() where run_id = %s;",
          (total_rows, run_id),
        )

        try:
          # Global snapshot for the run (tiny, high value).
          cur.execute(
            """
            insert into kalshi.market_snapshot_global (
              snap_ts,
              run_id,
              n_active,
              n_priced,
              total_volume,
              total_open_interest,
              avg_spread_ticks,
              n_wide_spread
            )
            select
              %s as snap_ts,
              %s as run_id,
              count(*) as n_active,
              count(*) filter (where yes_bid is not null and yes_ask is not null) as n_priced,
              coalesce(sum(volume),0) as total_volume,
              coalesce(sum(open_interest),0) as total_open_interest,
              avg((yes_ask - yes_bid)) filter (where yes_bid is not null and yes_ask is not null) as avg_spread_ticks,
              count(*) filter (where yes_bid is not null and yes_ask is not null and (yes_ask - yes_bid) >= 10) as n_wide_spread
            from kalshi.open_markets
            where status='active'
            on conflict (run_id) do nothing;
            """,
            (snap_ts, run_id),
          )
          print("inserted global snapshot rows:", cur.rowcount)

          # Series-level snapshot for the run (using event_ticker as the grouping key).
          cur.execute(
            """
            insert into kalshi.market_snapshot_series (
              snap_ts,
              run_id,
              series_ticker,
              n_markets,
              total_volume,
              total_open_interest,
              avg_spread_ticks
            )
            select
              %s as snap_ts,
              %s as run_id,
              nullif(raw->>'event_ticker','') as series_ticker,
              count(*) as n_markets,
              coalesce(sum(volume),0) as total_volume,
              coalesce(sum(open_interest),0) as total_open_interest,
              avg((yes_ask - yes_bid))::numeric as avg_spread_ticks
            from kalshi.open_markets
            where status='active'
              and nullif(raw->>'event_ticker','') is not null
              and yes_bid is not null and yes_ask is not null
            group by nullif(raw->>'event_ticker','')
            having coalesce(sum(volume),0) > 0
                or coalesce(sum(open_interest),0) > 0
            on conflict (snap_ts, series_ticker) do nothing;
            """,
            (snap_ts, run_id),
          )
          print("inserted series snapshot rows:", cur.rowcount)

          # Selective per-market snapshots for a bounded set of "interesting" markets.
          if enable_market_snap:
            cur.execute(
              """
              insert into kalshi.market_snapshot_markets (
                snap_ts,
                run_id,
                market_ticker,
                series_ticker,
                expiration_time,
                yes_bid,
                yes_ask,
                volume,
                open_interest,
                spread_ticks,
                mid
              )
              with c as (
                select
                  market_ticker,
                  series_ticker,
                  expiration_time,
                  yes_bid,
                  yes_ask,
                  volume,
                  open_interest,
                  (yes_ask - yes_bid) as spread_ticks,
                  ((yes_bid + yes_ask)/2.0)::numeric as mid
                from kalshi.open_markets
                where status='active'
                  and yes_bid is not null and yes_ask is not null
              ),
              pick as (
                (select * from c order by volume desc nulls last limit 20000)
                union
                (select * from c order by open_interest desc nulls last limit 20000)
                union
                (select * from c where spread_ticks >= 10 order by spread_ticks desc limit 20000)
                union
                (select * from c
                  where expiration_time is not null
                    and expiration_time <= %s + interval '48 hours'
                  order by expiration_time asc
                  limit 20000)
              )
              insert into kalshi.market_snapshot_markets (
                snap_ts, run_id, market_ticker, series_ticker, expiration_time,
                yes_bid, yes_ask, volume, open_interest, spread_ticks, mid
              )
              select
                %s as snap_ts, %s as run_id,
                market_ticker, series_ticker, expiration_time,
                yes_bid, yes_ask, volume, open_interest, spread_ticks, mid
              from pick
              on conflict (snap_ts, market_ticker) do nothing;
              """,
              (snap_ts, snap_ts, run_id),
            )
            print("inserted market snapshot rows:", cur.rowcount)
        except Exception as e:
          print("Error during snapshot export:", e)
          raise

        print(f"OK: upserted {total_rows} markets total, run_id={run_id}")


if __name__ == "__main__":
    main()
