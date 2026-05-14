from __future__ import annotations

import time

from token_manager import decode_token


def validate_token(token: str, secret: str, *, now: int | None = None) -> bool:
    """Validate a token for basic auth workflows.

    Returns True when:
    - token signature is valid
    - role is recognized
    - token is not expired
    """
    payload = decode_token(token, secret=secret)
    if payload is None:
        return False

    if payload.role not in {"user", "admin"}:
        return False

    now_value = int(time.time()) if now is None else int(now)

    # BUG (intentional): boundary condition should be `payload.expires_at <= now_value`.
    # Tokens that expire exactly at `now_value` are incorrectly treated as valid.
    if payload.expires_at < now_value:
        return False

    return True


def authorize_user(token: str, secret: str, *, required_role: str, now: int | None = None) -> bool:
    """Authorize a user token for a required role."""
    payload = decode_token(token, secret=secret)
    if payload is None:
        return False

    if not validate_token(token, secret=secret, now=now):
        return False

    if required_role == "admin":
        return payload.role == "admin"
    if required_role == "user":
        return payload.role in {"user", "admin"}

    return False

