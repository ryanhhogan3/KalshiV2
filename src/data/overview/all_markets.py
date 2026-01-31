import requests
from datetime import datetime, timezone
import json
import datetime as dt
import os

import psycopg2
from psycopg2.extras import Json



class OverviewAllMarkets:
    def __init__(self):
        self.BASE = "https://api.elections.kalshi.com/trade-api/v2"


    def fetch_all_open_markets(self, limit=1000):
        markets = []
        cursor = None

        while True:
            params = {"status": "open", "limit": limit}
            if cursor:
                params["cursor"] = cursor

            r = requests.get(f"{self.BASE}/markets", params=params, timeout=30)
            r.raise_for_status()
            data = r.json()

            markets.extend(data.get("markets", []))
            cursor = data.get("cursor")

            if not cursor:
                break

        return markets


    def export_open_markets_json(self):
        raw_markets = self.fetch_all_open_markets()

        out = {
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "market_count": len(raw_markets),
                "event_count": 0,
                "series_count": 0,
            },
            "series": {},
        }

        for m in raw_markets:
            series_id = m.get("series_id", "UNKNOWN_SERIES")
            event_id = m.get("event_id", "UNKNOWN_EVENT")

            series = out["series"].setdefault(series_id, {
                "series_id": series_id,
                "events": {}
            })

            events = series["events"]
            event = events.setdefault(event_id, {
                "event_id": event_id,
                "event_title": m.get("event_title"),
                "close_ts": m.get("close_ts"),
                "markets": []
            })

            event["markets"].append({
                "ticker": m.get("ticker"),
                "title": m.get("title"),
                "status": m.get("status"),
                "yes_ask": m.get("yes_ask"),
                "no_ask": m.get("no_ask"),
                "volume": m.get("volume"),
                "open_interest": m.get("open_interest"),
                "close_ts": m.get("close_ts"),
            })

        out["metadata"]["series_count"] = len(out["series"])
        out["metadata"]["event_count"] = sum(
            len(s["events"]) for s in out["series"].values()
        )

        return out

    def create_overview_file(self):
        now = dt.datetime.now().date()
        data = self.export_open_markets_json()

        with open(f"kalshi_open_markets_{now}.json", "w") as f:
            json.dump(data, f, indent=2)

    def save_overview_to_db(self):
        """Persist the current overview snapshot into a Postgres table.

        Expects the following environment variables to be set:
          PGHOST, PGDATABASE, PGUSER, PGPASSWORD, (optional) PGPORT

        And a table like:

          CREATE TABLE kalshi_snapshots (
              id SERIAL PRIMARY KEY,
              generated_at TIMESTAMPTZ NOT NULL,
              market_count INT,
              event_count INT,
              series_count INT,
              payload JSONB NOT NULL
          );
        """

        data = self.export_open_markets_json()

        conn = psycopg2.connect(
            host=os.environ["PGHOST"],
            dbname=os.environ["PGDATABASE"],
            user=os.environ["PGUSER"],
            password=os.environ["PGPASSWORD"],
            port=os.environ.get("PGPORT", 5432),
        )

        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO kalshi_snapshots (
                        generated_at,
                        market_count,
                        event_count,
                        series_count,
                        payload
                    ) VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        data["metadata"]["generated_at"],
                        data["metadata"]["market_count"],
                        data["metadata"]["event_count"],
                        data["metadata"]["series_count"],
                        Json(data),
                    ),
                )
        finally:
            conn.close()