import os
from dataclasses import dataclass

import psycopg


@dataclass(frozen=True)
class DbConfig:
    host: str
    port: int
    dbname: str
    user: str
    password: str
    sslmode: str


def load_config() -> DbConfig:
    return DbConfig(
        host=os.environ["DB_HOST"],
        port=int(os.environ["DB_PORT"]),
        dbname=os.environ.get("DB_NAME", "shift"),
        user=os.environ.get("DB_USER", "shift_user"),
        password=os.environ["DB_PASS"],
        sslmode=os.environ.get("DB_SSLMODE", "disable"),
    )


def connect() -> psycopg.Connection:
    cfg = load_config()
    conn = psycopg.connect(
        host=cfg.host,
        port=cfg.port,
        dbname=cfg.dbname,
        user=cfg.user,
        password=cfg.password,
        sslmode=cfg.sslmode,
        connect_timeout=8,
    )
    # Good defaults for scripts
    conn.autocommit = True
    return conn
