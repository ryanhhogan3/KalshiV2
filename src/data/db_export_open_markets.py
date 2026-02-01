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
    host = socket.gethostname()

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
        print(f"OK: upserted {total_rows} markets total, run_id={run_id}")


if __name__ == "__main__":
    main()
