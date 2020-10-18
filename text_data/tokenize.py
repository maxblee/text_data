"""This is a module for tokenizing data.

The primary motivation behind this module is that effectively
presenting search results revolves around knowing the positions
of the words *prior* to tokenization. In order to handle these raw
positions, the index `text_data.Corpus` uses optionally stores the
original character-level positions of words.

This module offers basic ways to produce those positions for use
in `text_data.Corpus`. However, you'll likely need to customize
them for most applications. In order to do that, you just need to create
a function that takes a string of text and converts it into two lists:
one with the list of words and another with a list of tuples, each of
them having two integers (the first for the starting character of the
original text, the other with the ending character).

Additionally, I've supplied a few off-the-shelf functions that should often
be useful:

- whitespace_tokenizer
"""
import functools
import re
from typing import Callable, List, Optional, Tuple

TokenizeResult = Tuple[List[str], List[Tuple[int, int]]]


def tokenize_regex_positions(
    pattern: str, document_text: str, inverse_match: bool = False
) -> TokenizeResult:
    """Finds all of the tokens matching a regular expression.

    Returns the positions of those tokens along with the tokens themselves.

    Args:
        pattern: A raw regular expression string
        document_text: The raw document text
        inverse_match: If true, tokenizes the text between matches.

    Returns:
        A tuple consisting of the list of words and a list of tuples,
        where each tuple represents the start and end character positions
        of the phrase.
    """
    tokens = []
    spans = []
    current_idx = 0
    regex = re.compile(pattern)
    for match in regex.finditer(document_text):
        match_start, match_end = match.span()
        if inverse_match:
            # if we don't include this, an inverse match will return a list with
            # an empty string (see `tests/test_tokenizers.py::test_empty_regex_match`)
            if current_idx == match_start == 0:
                current_idx = match_end
                continue
            start_idx, end_idx = current_idx, match_start
        else:
            start_idx, end_idx = match_start, match_end
        current_idx = match_end
        tokens.append(document_text[start_idx:end_idx])
        spans.append((start_idx, end_idx))
    # in inverse matches, make sure everything that doesn't match is included
    if inverse_match and current_idx != len(document_text):
        tokens.append(document_text[current_idx:])
        spans.append((current_idx, len(document_text)))
    return tokens, spans


def postprocess_positions(
    postprocess_funcs: List[Callable[[str], Optional[str]]],
    tokenized_docs: TokenizeResult,
) -> TokenizeResult:
    """Runs postprocessing functions to produce final tokenized documents.

    This function allows you to take results from `tokenize_regex_positions`
    (or something that has a similar function signature) and run postprocessing
    on them.

    These postprocessing functions should take a string (i.e. one of the individual tokens),
    but they can return either a string or None. If they return None, the token
    will not appear in the final tokenized result.

    Args:
        postprocess_funcs: A list of postprocessing functions (e.g. `str.lower`)
        tokenized_docs: The tokenized results (e.g. the output of `tokenize_regex_positions`)
    """
    post_tokens = []
    post_spans = []
    # there's probably a more elegant way to do this, but default for a,b in x doesn't
    # work here because spans is a tuple
    tokens, spans = tokenized_docs
    for token, span in zip(tokens, spans):
        func_result = token
        for func in postprocess_funcs:
            if func_result is not None:
                func_result = func(func_result)  # type: ignore
        if func_result is not None:
            post_tokens.append(func_result)
            post_spans.append(span)
    return post_tokens, post_spans


boundary_tokenizer = functools.partial(tokenize_regex_positions, r"\w+")
whitespace_tokenizer = functools.partial(tokenize_regex_positions, r"\s+")


def default_tokenizer(document_text: str) -> TokenizeResult:
    """This is the default tokenizer used in `text_data.Corpus`.

    Functionally, it is very simple. It splits on word boundaries
    and then converts text into lowercase.

    Args:
        document_text: The document you're tokenizing.
    """
    return postprocess_positions([str.lower], boundary_tokenizer(document_text))


def query_tokenizer(document_text: str) -> List[str]:
    """This is identical to `default_tokenizer` but it doesn't include positions.

    The purpose of this is to allow the tokenizer to be passed into `text_data.Query`.

    Args:
        document_text: The query you're tokenizing.
    """
    return document_text.split()
