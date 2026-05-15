from __future__ import annotations

from collections.abc import Iterable

from app.models import Movie, Review


class MovieRepository:
    def __init__(self) -> None:
        self._movies: dict[str, Movie] = {}
        self._reviews: list[Review] = []

    def add_movie(self, movie: Movie) -> Movie:
        self._movies[movie.movie_id] = movie
        return movie

    def add_review(self, review: Review) -> None:
        self._reviews.append(review)

    def list_movies(self) -> Iterable[Movie]:
        return tuple(self._movies.values())

    def list_reviews(self, movie_id: str) -> Iterable[Review]:
        return tuple(review for review in self._reviews if review.movie_id == movie_id)
