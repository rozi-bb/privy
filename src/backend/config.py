"""Backend settings, read from the environment."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://privy:privy@localhost:5433/privy"
)

# Signs the login tokens. Anyone holding it can mint a token for any user, so
# it lives only in .env and has no default.
JWT_SECRET = os.environ.get("JWT_SECRET", "")
JWT_EXPIRE_MINUTES = int(os.environ.get("JWT_EXPIRE_MINUTES", "1440"))
