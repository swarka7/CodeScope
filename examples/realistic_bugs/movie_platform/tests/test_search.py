from __future__ import annotations

from app.models import Movie, Review
from app.ratings import RatingSummary
from app.repository import MovieRepository
from app.routes import list_movies
from app.search import MovieSearchService


def test_text_search_matches_title_and_description() -> None:
    service, _ = _build_catalog()

    results = list_movies(service, query="orbital")

    assert [movie.title for movie in results] == ["Orbital Dawn"]


def test_minimum_rating_filter_excludes_lower_rated_movies() -> None:
    service, _ = _build_catalog()

    results = list_movies(service, min_rating=4.8)

    assert [movie.title for movie in results] == ["Kitchen Stories", "Orbital Dawn"]


def test_release_year_filter_limits_results() -> None:
    service, _ = _build_catalog()

    results = list_movies(service, release_year=2019)

    assert [movie.title for movie in results] == ["Atlas Road"]


def test_combined_filters_require_genre_rating_and_year_to_match() -> None:
    service, _ = _build_catalog()

    results = list_movies(service, genre="sci-fi", min_rating=4.5, release_year=2021)

    assert [movie.title for movie in results] == ["Orbital Dawn"]


def _build_catalog() -> tuple[MovieSearchService, MovieRepository]:
    repository = MovieRepository()
    movies = [
        Movie(
            title="Orbital Dawn",
            genres=("sci-fi", "adventure"),
            release_year=2021,
            description="A rescue crew crosses deep space after a station blackout.",
        ),
        Movie(
            title="Kitchen Stories",
            genres=("documentary",),
            release_year=2021,
            description="A quiet portrait of chefs rebuilding a neighborhood restaurant.",
        ),
        Movie(
            title="Neon Harbor",
            genres=("thriller",),
            release_year=2021,
            description="A detective follows a smuggling ring through the city docks.",
        ),
        Movie(
            title="Atlas Road",
            genres=("drama",),
            release_year=2019,
            description="A family-run transport company tries to survive a strike.",
        ),
    ]
    for movie in movies:
        repository.add_movie(movie)

    _rate(repository, movies[0], 5.0, 4.6)
    _rate(repository, movies[1], 4.9, 4.8)
    _rate(repository, movies[2], 4.7, 4.4)
    _rate(repository, movies[3], 4.0, 4.1)

    return MovieSearchService(repository, RatingSummary(repository)), repository


def _rate(repository: MovieRepository, movie: Movie, *ratings: float) -> None:
    for index, rating in enumerate(ratings, start=1):
        repository.add_review(
            Review(
                movie_id=movie.movie_id,
                user_id=f"user-{index}",
                rating=rating,
                text="Worth watching.",
            )
        )
