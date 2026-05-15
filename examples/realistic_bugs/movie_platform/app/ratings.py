from __future__ import annotations

from app.repository import MovieRepository


class RatingSummary:
    def __init__(self, repository: MovieRepository) -> None:
        self._repository = repository

    def average_rating(self, movie_id: str) -> float:
        reviews = list(self._repository.list_reviews(movie_id))
        if not reviews:
            return 0.0
        return sum(review.rating for review in reviews) / len(reviews)
