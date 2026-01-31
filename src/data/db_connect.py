import json
import os
import subprocess
from dataclasses import dataclass

import psycopg


@dataclass(frozen=True)
class DbConfig:
    host: str
    port: int
    dbname: str
    user: str
    sslmode: str
    secret_arn: str


def _get_password_from_secrets_manager(secret_arn: str) -> str:
    """
    Uses AWS CLI credentials on the machine:
      - Local: your configured AWS profile
      - EC2: instance role permissions
    """
    secret_str = subprocess.check_output(
        [
            "aws",
            "secretsmanager",
            "get-secret-value",
            "--secret-id",
            secret_arn,
            "--query",
            "SecretString",
            "--output",
            "text",
        ],
        text=True,
    )
    return json.loads(secret_str)["password"]


def load_config() -> DbConfig:
    return DbConfig(
        host=os.environ["DB_HOST"],
        port=int(os.environ["DB_PORT"]),
        dbname=os.environ.get("DB_NAME", "postgres"),
        user=os.environ.get("DB_USER", "postgres"),
        sslmode=os.environ.get("DB_SSLMODE", "require"),
        secret_arn=os.environ["DB_SECRET_ARN"],
    )


def connect() -> psycopg.Connection:
    cfg = load_config()
    pwd = _get_password_from_secrets_manager(cfg.secret_arn)
    conn = psycopg.connect(
        host=cfg.host,
        port=cfg.port,
        dbname=cfg.dbname,
        user=cfg.user,
        password=pwd,
        sslmode=cfg.sslmode,
        connect_timeout=8,
    )
    # Good defaults for scripts
    conn.autocommit = True
    return conn
