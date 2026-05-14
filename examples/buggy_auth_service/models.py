from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenPayload:
    user_id: str
    role: str
    expires_at: int

    def to_dict(self) -> dict[str, object]:
        return {"user_id": self.user_id, "role": self.role, "expires_at": self.expires_at}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> TokenPayload:
        user_id = data.get("user_id")
        role = data.get("role")
        expires_at = data.get("expires_at")

        if not isinstance(user_id, str) or not user_id:
            raise ValueError("user_id must be a non-empty string")
        if not isinstance(role, str) or not role:
            raise ValueError("role must be a non-empty string")
        if not isinstance(expires_at, int):
            raise ValueError("expires_at must be an int epoch timestamp (seconds)")

        return cls(user_id=user_id, role=role, expires_at=expires_at)
