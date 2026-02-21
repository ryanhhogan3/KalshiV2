# Postgres Maintenance & Schema Guide (KalshiV2)

This file collects practical `psql` commands and queries for:

- Checking database / table sizes and health
- Inspecting schemas and indexes
- Looking at the most granular data in `kalshi.open_markets`
- Understanding what each table stores and whether it is current-state or historical

---

## 1. Connecting and basic context

```bash
# From your machine (RDS example from database.helper.md)
export ENDPOINT='test-database-2-instance-1.cyt4g6wgmcz8.us-east-1.rds.amazonaws.com'
psql "host=$ENDPOINT port=5432 dbname=postgres user=postgres sslmode=require"

# If you have a .pg_service.conf entry
psql service=kalshi_rds
```

Once connected, set the database and search path explicitly if needed:

```sql
-- Make sure we're in the right DB
SELECT current_database();

-- Prefer kalshi schema first, but still see public
SET search_path TO kalshi, public;
SHOW search_path;
```

---

## 2. Database- and schema-level size

### 2.1 Database sizes

```sql
-- Size of the current database
SELECT
  current_database() AS db,
  pg_size_pretty(pg_database_size(current_database())) AS size;

-- Sizes of all databases on the instance
SELECT
  datname AS db,
  pg_size_pretty(pg_database_size(datname)) AS size
FROM pg_database
ORDER BY pg_database_size(datname) DESC;
```

### 2.2 Schemas and relations

```sql
-- List schemas (psql meta)
\dn

-- List tables in kalshi schema
\dt kalshi.*

-- Pure SQL: all user tables in kalshi
SELECT
  table_schema,
  table_name
FROM information_schema.tables
WHERE table_schema = 'kalshi'
  AND table_type = 'BASE TABLE'
ORDER BY table_name;
```

---

## 3. Table size and row counts

### 3.1 Size of key tables in kalshi

```sql
-- One row per table with total size (data + indexes)
SELECT
  schemaname,
  relname AS table_name,
  pg_size_pretty(pg_total_relation_size(relid)) AS total_size,
  pg_size_pretty(pg_relation_size(relid))       AS table_size,
  pg_size_pretty(pg_total_relation_size(relid) - pg_relation_size(relid)) AS index_size,
  n_live_tup
FROM pg_catalog.pg_statio_user_tables
WHERE schemaname = 'kalshi'
ORDER BY pg_total_relation_size(relid) DESC;

-- Focus on a few specific tables
SELECT
  'kalshi.open_markets' AS table_name,
  pg_size_pretty(pg_total_relation_size('kalshi.open_markets')) AS total_size
UNION ALL
SELECT
  'kalshi.market_snapshot_series',
  pg_size_pretty(pg_total_relation_size('kalshi.market_snapshot_series'))
UNION ALL
SELECT
  'kalshi.market_snapshot_markets',
  pg_size_pretty(pg_total_relation_size('kalshi.market_snapshot_markets'))
UNION ALL
SELECT
  'kalshi.export_runs',
  pg_size_pretty(pg_total_relation_size('kalshi.export_runs'))
UNION ALL
SELECT
  'kalshi.open_markets_hist',
  pg_size_pretty(pg_total_relation_size('kalshi.open_markets_hist'));
```

### 3.2 Row-count and activity stats

```sql
-- Live row estimates and vacuum/analyze history
SELECT
  schemaname,
  relname,
  n_live_tup,
  last_vacuum,
  last_autovacuum,
  last_analyze,
  last_autoanalyze
FROM pg_stat_user_tables
WHERE schemaname = 'kalshi'
ORDER BY n_live_tup DESC;

-- Index vs seq scan usage (look for tables overusing seq scans)
SELECT
  schemaname,
  relname,
  seq_scan,
  idx_scan,
  n_tup_ins,
  n_tup_upd,
  n_tup_del
FROM pg_stat_user_tables
WHERE schemaname = 'kalshi'
ORDER BY seq_scan DESC;
```

---

## 4. Basic maintenance: VACUUM / ANALYZE (manual)

RDS runs autovacuum, but occasionally you may want to run a targeted operation when troubleshooting bloat or bad row estimates.

```sql
-- Safest: analyze without vacuuming
ANALYZE kalshi.open_markets;
ANALYZE kalshi.market_snapshot_series;
ANALYZE kalshi.market_snapshot_markets;

-- More aggressive: vacuum + analyze a single table
VACUUM (ANALYZE) kalshi.open_markets;

-- Full database-level VACUUM (be cautious on production)
VACUUM (ANALYZE);
```

On RDS, avoid `VACUUM FULL` unless you are absolutely sure you need it; it takes exclusive locks and rewrites tables.

---

## 5. Inspecting schemas and columns

### 5.1 Column definitions

```sql
-- Columns for kalshi.open_markets
SELECT
  column_name,
  data_type,
  is_nullable,
  column_default
FROM information_schema.columns
WHERE table_schema = 'kalshi'
  AND table_name   = 'open_markets'
ORDER BY ordinal_position;

-- Same pattern for other tables
SELECT
  column_name,
  data_type,
  is_nullable,
  column_default
FROM information_schema.columns
WHERE table_schema = 'kalshi'
  AND table_name   = 'market_snapshot_global'
ORDER BY ordinal_position;
```

### 5.2 Index definitions

```sql
-- All indexes in kalshi schema
SELECT
  tab.relname AS table_name,
  idx.relname AS index_name,
  pg_get_indexdef(i.indexrelid) AS indexdef
FROM pg_index i
JOIN pg_class idx ON idx.oid = i.indexrelid
JOIN pg_class tab ON tab.oid = i.indrelid
JOIN pg_namespace ns ON ns.oid = tab.relnamespace
WHERE ns.nspname = 'kalshi'
ORDER BY tab.relname, idx.relname;
```

---

## 6. `kalshi.open_markets`: most granular view

From the DDL in `src/data/db_export_open_markets.py`:

- **Primary key**: `market_ticker` (one row per market)
- **Key columns**: `series_ticker`, `title`, `status`, `open_time`, `close_time`, `expiration_time`, `yes_bid`, `yes_ask`, `no_bid`, `no_ask`, `volume`, `open_interest`, `raw` (jsonb), `updated_at`
- Rows are **upserted** on every exporter run (`ON CONFLICT (market_ticker) DO UPDATE`).
- This table is a **current-state cache**, not a time-series; each row is overwritten in place.

### 6.1 Quick current snapshot

```sql
-- Latest view of markets, most recently updated first
SELECT
  market_ticker,
  series_ticker,
  title,
  status,
  yes_bid,
  yes_ask,
  no_bid,
  no_ask,
  volume,
  open_interest,
  updated_at
FROM kalshi.open_markets
ORDER BY updated_at DESC
LIMIT 50;

-- A specific market
SELECT *
FROM kalshi.open_markets
WHERE market_ticker = 'SOME_TICKER_HERE';
```

### 6.2 Inspecting the raw JSONB payload

```sql
-- Pretty-printed raw payload for a single market
SELECT
  market_ticker,
  jsonb_pretty(raw) AS raw_pretty
FROM kalshi.open_markets
WHERE market_ticker = 'SOME_TICKER_HERE';

-- Peek at a few random markets
SELECT
  market_ticker,
  jsonb_pretty(raw) AS raw_pretty
FROM kalshi.open_markets
ORDER BY random()
LIMIT 3;
```

### 6.3 Pulling structured fields from `raw`

```sql
-- Pull key attributes out of raw JSON without dumping the whole document
SELECT
  market_ticker,
  raw->>'event_ticker'         AS event_ticker,
  raw->>'market_type'          AS market_type,
  raw->>'response_price_units' AS price_units,
  raw->>'rules_primary'        AS rules_primary,
  raw->>'rules_secondary'      AS rules_secondary
FROM kalshi.open_markets
WHERE status = 'active'
ORDER BY updated_at DESC
LIMIT 50;

-- Sanity-check distinct event_tickers
SELECT DISTINCT raw->>'event_ticker' AS event_ticker
FROM kalshi.open_markets
WHERE raw->>'event_ticker' IS NOT NULL
ORDER BY 1
LIMIT 100;
```

### 6.4 Finding problematic or stale rows

```sql
-- Markets missing a ticker (should be empty; exporter hard-fails otherwise)
SELECT *
FROM kalshi.open_markets
WHERE market_ticker IS NULL;

-- Markets stale beyond a threshold
SELECT
  market_ticker,
  updated_at,
  now() - updated_at AS age
FROM kalshi.open_markets
WHERE updated_at < now() - interval '1 day'
ORDER BY updated_at ASC
LIMIT 50;
```

---

## 7. Time-series / snapshot tables and retention

These are defined in `src/data/db_export_open_markets.py` and populated by the exporter after each run.

### 7.1 `kalshi.export_runs` (run log / heartbeat)

Definition (simplified):

- `run_id bigserial primary key`
- `started_at timestamptz not null default now()`
- `finished_at timestamptz null`
- `source text not null` (e.g., `local`, `ec2`)
- `host text not null` (hostname/container)
- `n_markets integer not null` (final count, updated at end of run)
- `note text null`

Behavior:

- **One row per exporter execution** (append-only).
- `n_markets` and `finished_at` are updated at the end of a successful run.
- Intended to be kept as a **historical log**; you might prune very old rows but generally this stays forever.

Useful queries:

```sql
-- Recent runs
SELECT
  run_id,
  started_at,
  finished_at,
  n_markets,
  source,
  host,
  note
FROM kalshi.export_runs
ORDER BY run_id DESC
LIMIT 20;

-- Detect failed / incomplete runs
SELECT
  run_id,
  started_at,
  finished_at,
  n_markets,
  note
FROM kalshi.export_runs
WHERE finished_at IS NULL
ORDER BY run_id DESC;
```

### 7.2 `kalshi.market_snapshot_global` (one row per run)

Definition (simplified):

- `run_id bigint primary key`
- `snap_ts timestamptz not null`
- `n_active bigint not null` (count of active markets)
- `n_priced bigint not null` (active markets with both `yes_bid` and `yes_ask`)
- `total_volume bigint not null`
- `total_open_interest bigint not null`
- `avg_spread_ticks numeric null`
- `n_wide_spread bigint not null` (priced markets with spread ≥ 10 ticks)

Behavior:

- Inserted **once per run**; `ON CONFLICT (run_id) DO NOTHING` keeps it idempotent.
- This is a **compact historical time series** of the whole market regime.

Useful queries:

```sql
-- Latest global snapshots
SELECT
  snap_ts,
  run_id,
  n_active,
  n_priced,
  total_volume,
  total_open_interest,
  avg_spread_ticks,
  n_wide_spread
FROM kalshi.market_snapshot_global
ORDER BY snap_ts DESC
LIMIT 50;

-- 24h change in total open interest
WITH latest AS (
  SELECT *
  FROM kalshi.market_snapshot_global
  ORDER BY snap_ts DESC
  LIMIT 1
),
prev AS (
  SELECT *
  FROM kalshi.market_snapshot_global
  WHERE snap_ts <= (SELECT snap_ts FROM latest) - interval '24 hours'
  ORDER BY snap_ts DESC
  LIMIT 1
)
SELECT
  latest.snap_ts          AS t_now,
  prev.snap_ts            AS t_prev,
  latest.total_open_interest - prev.total_open_interest AS delta_oi
FROM latest, prev;
```

### 7.3 `kalshi.market_snapshot_series` (group-level time series)

Definition (simplified):

- Primary key: `(snap_ts, series_ticker)`
- `snap_ts timestamptz not null`
- `run_id bigint not null`
- `series_ticker text not null` (currently populated with `raw->>'event_ticker'`)
- `n_markets int not null`
- `total_volume bigint not null`
- `total_open_interest bigint not null`
- `avg_spread_ticks numeric null`

Behavior:

- **Append-only time series**: one row per (timestamp, event_ticker) per run.
- Grows quickly (hundreds of thousands of rows per snap); main driver of historical storage after `open_markets`.
- Designed to be kept for analytics, but a good candidate for **retention policies** (e.g., keep 30–90 days).

Useful queries:

```sql
-- Recent snapshots for a particular event_ticker
SELECT
  snap_ts,
  series_ticker,
  n_markets,
  total_volume,
  total_open_interest,
  avg_spread_ticks
FROM kalshi.market_snapshot_series
WHERE series_ticker = 'SOME_EVENT_TICKER'
ORDER BY snap_ts DESC
LIMIT 100;

-- Top events by open interest at a given snapshot
SELECT
  snap_ts,
  series_ticker,
  total_open_interest
FROM kalshi.market_snapshot_series
WHERE snap_ts = (
  SELECT MAX(snap_ts)
  FROM kalshi.market_snapshot_series
)
ORDER BY total_open_interest DESC
LIMIT 20;
```

### 7.4 `kalshi.market_snapshot_markets` (selected market-level time series)

Definition (simplified):

- Primary key: `(snap_ts, market_ticker)`
- `snap_ts timestamptz not null`
- `run_id bigint not null`
- `market_ticker text not null`
- `series_ticker text null`
- `expiration_time timestamptz null`
- `yes_bid int null`, `yes_ask int null`
- `volume int null`, `open_interest int null`
- `spread_ticks int null`, `mid numeric null`

Behavior:

- **Append-only time series for a bounded set of “interesting” markets** per run.
- Exporter selects markets based on volume, open_interest, wide spreads, and near expiration, with a cap (e.g., top 20k per criterion).
- Designed for **charting and modeling** individual markets over time.

Useful queries:

```sql
-- Time series for a single market
SELECT
  snap_ts,
  yes_bid,
  yes_ask,
  mid,
  volume,
  open_interest
FROM kalshi.market_snapshot_markets
WHERE market_ticker = 'SOME_TICKER_HERE'
ORDER BY snap_ts;

-- Markets with widest spreads at the latest snapshot
WITH latest AS (
  SELECT MAX(snap_ts) AS snap_ts
  FROM kalshi.market_snapshot_markets
)
SELECT
  m.market_ticker,
  m.snap_ts,
  m.spread_ticks,
  m.mid,
  m.volume,
  m.open_interest
FROM kalshi.market_snapshot_markets m
JOIN latest l ON m.snap_ts = l.snap_ts
ORDER BY m.spread_ticks DESC
LIMIT 50;
```

### 7.5 `kalshi.open_markets_hist` (optional full-history lake)

Definition (simplified):

- `snap_id bigserial primary key`
- `snap_ts timestamptz not null`
- `run_id bigint not null REFERENCES kalshi.export_runs(run_id)`
- `market_ticker text not null`
- `series_ticker text null`
- `expiration_time timestamptz null`
- `yes_bid int null`, `yes_ask int null`
- `volume int null`, `open_interest int null`
- `updated_at timestamptz not null`

Behavior:

- Meant to store a **full copy of every market on every run** (very large).
- As of the current exporter code, this table is **defined but not populated** (empty or nearly empty); it is a future option.
- If you ever turn it on, treat it as a long-term history lake and plan **partitioning + retention** up front.

Example checks:

```sql
-- Confirm whether it's being used
SELECT COUNT(*) AS n_rows
FROM kalshi.open_markets_hist;

SELECT
  MIN(snap_ts) AS first_snap,
  MAX(snap_ts) AS last_snap
FROM kalshi.open_markets_hist;
```

---

## 8. `public.ingest_events` (ingest diagnostics)

From `database.helper.md`, you also have an ingest log table in the `public` schema:

- `id bigint` — primary key
- `ts_utc timestamp` — event time
- `source text` — origin of data
- `host text` — server identifier
- `note text` — details / free text

Behavior:

- **Append-only event log** used for diagnostics and auditing of ingest pipelines.
- You can safely apply **retention** on very old rows if desired.

Example queries:

```sql
-- Recent ingest events
SELECT
  id,
  ts_utc,
  source,
  host,
  note
FROM public.ingest_events
ORDER BY ts_utc DESC
LIMIT 50;

-- Count events by source in the last 24h
SELECT
  source,
  COUNT(*) AS n_events
FROM public.ingest_events
WHERE ts_utc >= now() - interval '24 hours'
GROUP BY source
ORDER BY n_events DESC;
```

---

## 9. Example retention / housekeeping queries (use with care)

Before running any deletions, **always** run the `SELECT` form (with `LIMIT`) first to verify what you are about to delete, and consider adding partitions instead of raw deletes for very large tables.

```sql
-- Keep only the last 90 days of per-market snapshots
DELETE FROM kalshi.market_snapshot_markets
WHERE snap_ts < now() - interval '90 days';

-- Keep only the last 30 days of series snapshots
DELETE FROM kalshi.market_snapshot_series
WHERE snap_ts < now() - interval '30 days';

-- Optionally thin out very old export_runs (if desired)
DELETE FROM kalshi.export_runs
WHERE started_at < now() - interval '365 days';
```

For serious long-term scale, consider **time-based partitioning** on `snap_ts` for the snapshot tables so that retention becomes a matter of dropping old partitions instead of large deletes.

---

## 10. Quick cheat sheet

- **DB size**: `SELECT pg_size_pretty(pg_database_size(current_database()));`
- **Largest kalshi tables**: use the `pg_statio_user_tables` size query in section 3.1
- **Table row counts & autovacuum history**: see section 3.2 (`pg_stat_user_tables`)
- **Current markets (latest state)**: select from `kalshi.open_markets` ordered by `updated_at DESC`
- **Historical global metrics**: `kalshi.market_snapshot_global`
- **Historical event-level metrics**: `kalshi.market_snapshot_series`
- **Selected per-market time series**: `kalshi.market_snapshot_markets`
- **Run log / health**: `kalshi.export_runs`
- **Optional full history**: `kalshi.open_markets_hist` (currently schema-only)
- **Ingest diagnostics**: `public.ingest_events`
