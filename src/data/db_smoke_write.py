import socket
import time
from db_connect import connect


DDL = """
create table if not exists ingest_events (
  id bigserial primary key,
  ts_utc timestamptz not null default now(),
  source text not null,
  host text not null,
  note text not null
);
"""

INSERT = """
insert into ingest_events (source, host, note)
values (%s, %s, %s)
returning id, ts_utc;
"""

SELECT_LAST = """
select id, ts_utc, source, host, note
from ingest_events
order by id desc
limit 5;
"""


def main():
    hostname = socket.gethostname()
    note = f"hello from {hostname} at {int(time.time())}"

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(DDL)
            cur.execute(INSERT, ("smoke_test", hostname, note))
            row = cur.fetchone()
            print("Inserted:", row)

            cur.execute(SELECT_LAST)
            print("Last 5 rows:")
            for r in cur.fetchall():
                print(" ", r)


if __name__ == "__main__":
    main()
