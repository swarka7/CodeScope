from __future__ import annotations

from dataclasses import dataclass

from app.models import Movie
from app.ratings import RatingSummary
from app.repository import MovieRepository


@dataclass(frozen=True, slots=True)
class MovieSearchCriteria:
    query: str | None = None
    genre: str | None = None
    min_rating: float | None = None
    release_year: int | None = None


class MovieSearchService:
    def __init__(self, repository: MovieRepository, ratings: RatingSummary) -> None:
        self._repository = repository
        self._ratings = ratings

    def search(self, criteria: MovieSearchCriteria) -> list[Movie]:
        results = list(self._repository.list_movies())

        if criteria.query:
            query = criteria.query.casefold()
            results = [
                movie
                for movie in results
                if query in movie.title.casefold() or query in movie.description.casefold()
            ]

        if criteria.release_year is not None:
            results = [movie for movie in results if movie.release_year == criteria.release_year]

        if criteria.min_rating is not None:
            results = [
                movie
                for movie in results
                if self._ratings.average_rating(movie.movie_id) >= criteria.min_rating
            ]

        return sorted(results, key=lambda movie: movie.title)
