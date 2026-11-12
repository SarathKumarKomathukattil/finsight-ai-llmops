
from pathlib import Path
from finsight.database import get_connection


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    schema_path = PROJECT_ROOT / "sql" / "schema.sql"
    schema = schema_path.read_text(encoding="utf-8")

    if not schema.strip():
        raise ValueError("schema.sql is empty")

    with get_connection() as conn:
        conn.execute(schema)

        result = conn.execute(
            "SELECT to_regclass('public.chunks')"
        ).fetchone()

        if result[0] is None:
            raise RuntimeError("Chunks table was not created")

    print("Database initialized successfully")


if __name__ == "__main__":
    main()

