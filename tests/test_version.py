import pytest

from sherma.version import (
    WILDCARD,
    find_best_match,
    find_latest,
    matches_specifier,
    parse_version,
)


def test_parse_version():
    v = parse_version("1.2.3")
    assert str(v) == "1.2.3"


def test_parse_version_invalid():
    with pytest.raises(ValueError):
        parse_version("not-a-version")


def test_matches_specifier():
    assert matches_specifier("1.2.3", ">=1.0")
    assert not matches_specifier("0.9.0", ">=1.0")


def test_matches_specifier_wildcard_pattern():
    assert matches_specifier("1.2.3", "==1.*")
    assert not matches_specifier("2.0.0", "==1.*")


def test_find_latest():
    assert find_latest(["1.0.0", "2.0.0", "1.5.0"]) == "2.0.0"


def test_find_latest_ignores_wildcard():
    assert find_latest(["1.0.0", WILDCARD]) == "1.0.0"


def test_find_latest_empty():
    assert find_latest([]) is None


def test_find_latest_only_wildcard():
    assert find_latest([WILDCARD]) is None


def test_find_best_match_exact():
    versions = ["1.0.0", "1.1.0", "2.0.0"]
    assert find_best_match(versions, "==1.1.0") == "1.1.0"


def test_find_best_match_range():
    versions = ["1.0.0", "1.1.0", "1.2.0", "2.0.0"]
    assert find_best_match(versions, "==1.*") == "1.2.0"


def test_find_best_match_wildcard_query():
    versions = ["1.0.0", "2.0.0"]
    assert find_best_match(versions, WILDCARD) == "2.0.0"


def test_find_best_match_wildcard_entry_fallback():
    versions = [WILDCARD]
    assert find_best_match(versions, "==1.*") == WILDCARD


def test_find_best_match_no_match():
    versions = ["1.0.0"]
    assert find_best_match(versions, "==2.*") is None


def test_find_best_match_wildcard_query_with_wildcard_entry():
    versions = [WILDCARD]
    assert find_best_match(versions, WILDCARD) == WILDCARD
