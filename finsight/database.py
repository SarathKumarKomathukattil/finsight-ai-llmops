
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def get_connection():
    """Connect to local PostgreSQL."""

    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "127.0.0.1"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        sslmode="disable",
        connect_timeout=10,
    )
