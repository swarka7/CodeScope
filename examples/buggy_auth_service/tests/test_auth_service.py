from __future__ import annotations

from auth_service import validate_token
from models import TokenPayload
from token_manager import encode_token

SECRET = "dev-secret"


def test_valid_token_is_accepted() -> None:
    now = 1_000
    payload = TokenPayload(user_id="u1", role="user", expires_at=now + 60)
    token = encode_token(payload, secret=SECRET)

    assert validate_token(token, secret=SECRET, now=now) is True


def test_invalid_signature_is_rejected() -> None:
    now = 1_000
    payload = TokenPayload(user_id="u1", role="user", expires_at=now + 60)
    token = encode_token(payload, secret="wrong-secret")

    assert validate_token(token, secret=SECRET, now=now) is False


def test_expired_token_is_rejected() -> None:
    # This test intentionally fails due to a bug in `validate_token`.
    now = 1_000
    payload = TokenPayload(user_id="u1", role="user", expires_at=now)
    expired_token = encode_token(payload, secret=SECRET)

    result = validate_token(expired_token, secret=SECRET, now=now)
    assert result is False
