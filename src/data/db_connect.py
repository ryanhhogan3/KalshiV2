import json
import os
from dataclasses import dataclass

import boto3
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
    """Fetch the DB password from AWS Secrets Manager using boto3.

    This relies on the container having access to AWS credentials
    (e.g., EC2 instance role) and a region configuration. The secret
    is expected to be a JSON object with a "password" key, same as
    before when we used the AWS CLI.
    """

    # Let AWS SDK resolve credentials from instance metadata / env.
    # Region can come from AWS_REGION or the standard config chain.
    region = os.environ.get("AWS_REGION")
    if region:
        client = boto3.client("secretsmanager", region_name=region)
    else:
        client = boto3.client("secretsmanager")

    resp = client.get_secret_value(SecretId=secret_arn)
    secret_str = resp.get("SecretString")
    if not secret_str and "SecretBinary" in resp:
        # Fallback if secret is stored as binary
        secret_str = resp["SecretBinary"].decode("utf-8")

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
