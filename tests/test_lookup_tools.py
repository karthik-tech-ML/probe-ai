"""
Regression tests for the field-anchored lookup tools.

The old ILIKE '%Director: %name%' let % cross newlines, so an actor's name
in the Cast line "matched" the Director field. These tests pin that down.
The pure tests need no database; the integration tests skip if it's down.
"""

import re

import pytest

from src.agent.tools import field_pattern

CHUNK = (
    "Title: Inception\n"
    "Director: Christopher Nolan\n"
    "Cast: Leonardo DiCaprio, Joseph Gordon-Levitt, Tom Hardy\n"
    "Genres: Action, Science Fiction, Adventure\n"
    "Release Year: 2010\n"
    "Plot: A thief who steals corporate secrets through dream-sharing technology.\n"
)


def _match(field: str, value: str) -> bool:
    # Python equivalent of Postgres '(?n)' + '~*': line anchors, case-insensitive
    return re.search(field_pattern(field, value), CHUNK, re.MULTILINE | re.IGNORECASE) is not None


def test_director_matches_director_line():
    assert _match("Director", "Christopher Nolan")
    assert _match("Director", "nolan")  # case-insensitive, substring


def test_director_does_not_match_name_from_other_lines():
    assert not _match("Director", "Leonardo DiCaprio")  # cast line
    assert not _match("Director", "Tom Hardy")           # cast line
    assert not _match("Director", "thief")               # plot line


def test_cast_matches_cast_line_only():
    assert _match("Cast", "Tom Hardy")
    assert not _match("Cast", "Christopher Nolan")  # director line
    assert not _match("Cast", "thief")              # plot line


def test_genre_matches_genres_line_only():
    assert _match("Genres", "Science Fiction")
    assert not _match("Genres", "Inception")  # title line


def test_special_characters_are_escaped():
    assert not _match("Director", ".*")  # not a wildcard
    assert not _match("Director", "(Nolan")  # unbalanced paren is literal, not a regex error


@pytest.fixture(scope="module")
def db():
    try:
        from src.database.connection import get_session
        session = get_session()
        session.execute(__import__("sqlalchemy").text("select 1"))
        session.close()
    except Exception as e:  # DB not running
        pytest.skip(f"database unavailable: {e}")


def test_director_search_no_longer_returns_other_directors_films(db):
    from src.agent.tools import search_by_director

    nolan = search_by_director.invoke({"director_name": "Christopher Nolan"})
    assert len(nolan) >= 5
    assert {m["director"] for m in nolan} == {"Christopher Nolan"}

    # DiCaprio is an actor in the dataset, not a director
    assert search_by_director.invoke({"director_name": "Leonardo DiCaprio"}) == []


def test_cast_search_still_finds_actors(db):
    from src.agent.tools import search_by_cast

    films = search_by_cast.invoke({"actor_name": "Leonardo DiCaprio"})
    assert len(films) >= 10
    assert all("Leonardo DiCaprio" in m["cast"] for m in films)
    assert search_by_cast.invoke({"actor_name": "Christopher Nolan"}) == []


def test_production_pattern_stays_on_production_line():
    chunk = "Title: X\nCast: Warner Bros.\nProduction: Legendary Pictures, Syncopy\n"
    on_prod = lambda v: re.search(field_pattern("Production", v), chunk, re.MULTILINE | re.IGNORECASE)
    assert on_prod("Legendary Pictures")
    assert not on_prod("Warner Bros.")  # appears only on the Cast line


def test_filter_movies_by_both_production_companies(db):
    from src.agent.tools import filter_movies

    both = filter_movies.invoke({
        "production_companies": ["Warner Bros.", "Legendary Pictures"], "limit": 50,
    })
    movies = [m for m in both if "movie_id" in m]
    titles = {m["title"] for m in movies}
    assert {"Inception", "The Dark Knight", "Interstellar", "Man of Steel"} <= titles
    assert len(movies) >= 20
    assert all("Warner Bros." in m["production"] and "Legendary Pictures" in m["production"] for m in movies)

    # one company alone matches at least as many
    one = [m for m in filter_movies.invoke({"production_companies": ["Legendary Pictures"], "limit": 50})
           if "movie_id" in m]
    assert len(one) >= len(movies)


def test_filter_movies_without_production_filter_is_unchanged(db):
    from src.agent.tools import filter_movies

    out = [m for m in filter_movies.invoke({"genre": "Comedy", "min_rating": 7.5, "year_from": 2010})
           if "movie_id" in m]
    assert out and all("production" not in m for m in out)
