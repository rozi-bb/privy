"""Backend settings, read from the environment."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://privy:privy@localhost:5433/privy"
)
