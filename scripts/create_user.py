"""Create a Privy account through the backend (the chat UI has no sign-up page).

The backend must be running. The password is asked for without echoing it:

    uv run python scripts/create_user.py ana@example.com
"""

from __future__ import annotations

import getpass
import os
import sys

import httpx

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    email = sys.argv[1]
    password = getpass.getpass("Password (at least 8 characters): ")
    if password != getpass.getpass("Repeat password: "):
        print("Passwords do not match.")
        return 1

    response = httpx.post(
        f"{BACKEND_URL}/auth/register",
        json={"email": email, "password": password},
        timeout=10.0,
    )
    if response.status_code == 201:
        print(f"Account created for {response.json()['email']}.")
        return 0
    print(
        f"Could not create the account ({response.status_code}): {response.json()['detail']}"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
