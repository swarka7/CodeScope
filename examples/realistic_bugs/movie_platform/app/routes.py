from __future__ import annotations

from app.models import Movie
from app.search import (
    MovieSearchCriteria,
    MovieSearchService,
)


def list_movies(
    service: MovieSearchService,
    *,
    query: str | None = None,
    genre: str | None = None,
    min_rating: float | None = None,
    release_year: int | None = None,
) -> list[Movie]:
    criteria = MovieSearchCriteria(
        query=query,
        genre=genre,
        min_rating=min_rating,
        release_year=release_year,
    )
    return service.search(criteria)
