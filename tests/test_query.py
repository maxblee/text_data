"""Tests `text_data.query`."""
import pytest

from text_data.query import Query


@pytest.mark.parametrize(
    "query,output",
    [
        ("", [[]]),
        ("The cat AND the dog", [[["The", "cat"]], [["the", "dog"]]]),
        ("The cat AND 'the dog' ran", [[["The", "cat"]], [["the", "dog"], ["ran"]]]),
        ('The cat AND "the dog" ran', [[["The", "cat"]], [["the", "dog"], ["ran"]]]),
        (
            "The cat ran to the dog's owner's party",
            [[["The", "cat", "ran", "to", "the", "dog's", "owner's", "party"]]],
        ),
    ],
)
def test_query_parsing_words(query, output):
    """Tests whether parsing queries produces the right sets of words."""
    assert [
        [q.words for q in subquery]
        for subquery in Query(query, query_tokenizer=str.split).queries
    ] == output


@pytest.mark.parametrize(
    "query,output",
    [
        ("The cat AND the dog", [["OR"], ["AND"]]),
        ("The cat AND 'the dog' ran", [["OR"], ["AND", "AND"]]),
        ("Cat AND dog NOT terrier", [["OR"], ["AND"], ["NOT"]]),
        ("Cat AND dog 'boxer' NOT terrier", [["OR"], ["AND", "AND"], ["NOT"]]),
    ],
)
def test_query_modifier_type(query, output):
    """Tests whether the query modifiers work."""
    assert [
        [q.modifier for q in subquery]
        for subquery in Query(query, query_tokenizer=str.split).queries
    ] == output


@pytest.mark.parametrize(
    "query,output",
    [
        ("The cat AND the dog", [[False], [False]]),
        ("The cat AND 'the dog' ran", [[False], [True, False]]),
        ('The cat AND "the dog"', [[False], [True]]),
    ],
)
def test_query_is_exact(query, output):
    """Tests the things that determine whether to do a phrase search or an inexact search."""
    assert [
        [q.exact for q in subquery]
        for subquery in Query(query, query_tokenizer=str.split).queries
    ] == output


@pytest.mark.parametrize("query", ["AND", " OR", "NOT ", " AND "])
def test_starting_query_with_keyword_raises_error(query):
    """Starting queries with keywords should cause a ValueError."""
    with pytest.raises(ValueError):
        Query(query, query_tokenizer=str.split)


@pytest.mark.parametrize(
    "query,output",
    [
        ("the cat AND the dog", [[["the", "cat"]], [["the", "dog"]]]),
        ("The cat AND the dog", [[["the", "cat"]], [["the", "dog"]]]),
        ("'The cat' AND the dog", [[["the", "cat"]], [["the", "dog"]]]),
        ('"The cat" AND the dog', [[["the", "cat"]], [["the", "dog"]]]),
    ],
)
def test_default_tokenizer(query, output):
    """The default query tokenizer should lowercase words without mishandling quotations."""
    assert [[q.words for q in subquery] for subquery in Query(query).queries] == output
