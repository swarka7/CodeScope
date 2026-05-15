from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class Movie:
    title: str
    genres: tuple[str, ...]
    release_year: int
    description: str
    movie_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(frozen=True, slots=True)
class Review:
    movie_id: str
    user_id: str
    rating: float
    text: str
