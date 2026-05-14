from __future__ import annotations

import base64
import hashlib
import hmac
import json

from models import TokenPayload


def encode_token(payload: TokenPayload, secret: str) -> str:
    """Encode a signed token.

    This intentionally uses a simple base64url(JSON) + HMAC signature format to keep the demo small.
    """
    raw = json.dumps(payload.to_dict(), separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload_part = _b64url_encode(raw)
    signature = _sign(raw, secret=secret)
    return f"{payload_part}.{signature}"


def decode_token(token: str, secret: str) -> TokenPayload | None:
    """Decode and verify a token, returning a TokenPayload or None if invalid."""
    try:
        payload_part, signature = token.split(".", 1)
        raw = _b64url_decode(payload_part)
    except ValueError:
        return None

    expected = _sign(raw, secret=secret)
    if not hmac.compare_digest(signature, expected):
        return None

    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None

    if not isinstance(decoded, dict):
        return None

    try:
        return TokenPayload.from_dict(decoded)
    except ValueError:
        return None


def _sign(payload_bytes: bytes, *, secret: str) -> str:
    key = secret.encode("utf-8")
    return hmac.new(key, payload_bytes, hashlib.sha256).hexdigest()


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64url_decode(data: str) -> bytes:
    padded = data + ("=" * (-len(data) % 4))
    return base64.urlsafe_b64decode(padded.encode("ascii"))

