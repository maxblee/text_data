"""Tests the tokenizers in `text_data.tokenize`."""
import pytest

from text_data import tokenize


@pytest.mark.parametrize(
    "input,regex,expected",
    [
        ("Ex doc", r"[A-Za-z]+", (["Ex", "doc"], [(0, 2), (3, 6)])),
        ("It's a cat", r"\S+", (["It's", "a", "cat"], [(0, 4), (5, 6), (7, 10)])),
    ],
)
def test_regex_tokenizer(input, regex, expected):
    """Tests the regex tokenizer *without inverse matches*."""
    assert tokenize.tokenize_regex_positions(regex, input) == expected


@pytest.mark.parametrize(
    "input,regex,expected",
    [
        ("Ex doc ", r"\s+", (["Ex", "doc"], [(0, 2), (3, 6)])),
        ("Ex doc", r"\s+", (["Ex", "doc"], [(0, 2), (3, 6)])),
        ("Ex", r"\s+", (["Ex"], [(0, 2)])),
    ],
)
def test_regex_inverse_match(input, regex, expected):
    """Tests the regex tokenizer when inverse is selected."""
    assert (
        tokenize.tokenize_regex_positions(regex, input, inverse_match=True) == expected
    )


@pytest.mark.parametrize(
    "input,regex,inverse",
    [("Everyman", r"\s+", False), ("", r".", False), ("Everyman", r"\w+", True)],
)
def test_empty_regex_match(input, regex, inverse):
    """Makes sure passing a non-matching regex to the regex matcher returns an empty list."""
    assert tokenize.tokenize_regex_positions(regex, input, inverse_match=inverse) == (
        [],
        [],
    )


def test_postprocessing():
    """Tests postprocessing functions in `tokenize.postprocess_positions`."""
    tokenized_text = tokenize.tokenize_regex_positions(r"\w+", "Ex doc")
    assert tokenize.postprocess_positions([str.lower], tokenized_text) == (
        ["ex", "doc"],
        [(0, 2), (3, 6)],
    )
    assert tokenize.postprocess_positions(
        [lambda x: x.lower() if x != "doc" else None], tokenized_text
    ) == (["ex"], [(0, 2)])
