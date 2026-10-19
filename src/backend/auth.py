"""Sign-in: password hashing, login tokens, and the current-user dependency.

Passwords are stored only as argon2 hashes. A successful login returns a JWT
whose subject is the user id; every protected route reads it from the
`Authorization: Bearer <token>` header through `current_user`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated, Any
from uuid import UUID

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pwdlib import PasswordHash

from backend import config, db

JWT_ALGORITHM = "HS256"

_password_hash = PasswordHash.recommended()

# Verified against when the email is unknown, so a wrong email takes as long
# as a wrong password and response times do not reveal who has an account.
_DUMMY_HASH = _password_hash.hash("not-a-real-password")

# auto_error=False: a missing header must give 401, like a bad token does.
_bearer = HTTPBearer(auto_error=False)

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Not signed in",
    headers={"WWW-Authenticate": "Bearer"},
)


def hash_password(password: str) -> str:
    """Hash a password for storage."""
    return _password_hash.hash(password)


def verify_password(password: str, password_hash: str | None) -> bool:
    """Check a password against a stored hash (or the dummy one if None)."""
    return _password_hash.verify(password, password_hash or _DUMMY_HASH)


def create_token(user_id: UUID) -> str:
    """Issue a login token for the user."""
    now = datetime.now(UTC)
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + timedelta(minutes=config.JWT_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, config.JWT_SECRET, algorithm=JWT_ALGORITHM)


def _user_id_from_token(token: str) -> UUID:
    try:
        payload = jwt.decode(
            token,
            config.JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
            options={"require": ["sub", "exp"]},
        )
        return UUID(payload["sub"])
    except (jwt.InvalidTokenError, ValueError) as error:
        raise _UNAUTHORIZED from error


async def current_user(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> dict[str, Any]:
    """Return the signed-in user, or answer 401."""
    if credentials is None:
        raise _UNAUTHORIZED
    user_id = _user_id_from_token(credentials.credentials)
    # A valid token for a deleted account must not keep working.
    user = await db.get_user(request.app.state.pool, user_id)
    if user is None:
        raise _UNAUTHORIZED
    return user


CurrentUser = Annotated[dict[str, Any], Depends(current_user)]
