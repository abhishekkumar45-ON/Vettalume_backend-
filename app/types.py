from sqlalchemy import JSON
from sqlalchemy.dialects.postgresql import JSONB

# JSONB on Postgres (indexable, fast), plain JSON on SQLite so the smoke tests run anywhere.
JSONType = JSON().with_variant(JSONB(), "postgresql")
