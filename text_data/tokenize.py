"""This is a module for tokenizing data.

The primary motivation behind this module is that effectively
presenting search results revolves around knowing the positions
of the words *prior* to tokenization. In order to handle these raw
positions, the index :class:`text_data.index.Corpus` uses stores the
original character-level positions of words.

This module offers a default tokenizer that you can use
for :class:`text_data.index.Corpus`. However, you'll likely need to customize
them for most applications. That said, doing so should not be difficult.

One of the functions in this module, :func:`~text_data.tokenize.corpus_tokenizer`,
is designed specifically to create tokenizers that can be used
directly by :class:`text_data.index.Corpus`. All you have to do
is create a regular expression that splits words from nonwords
and then create a series of postprocessing functions to clean the
text (including, optionally, removing tokens). If possible,
I would recommend taking this approach, since it allows you
to mostly ignore the picky preferences of the underlying API.
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
    tokenize_func: Callable[[str], TokenizeResult],
    document: str,
) -> TokenizeResult:
    """Runs postprocessing functions to produce final tokenized documents.

    This function allows you to take :func:`~text_data.tokenize.tokenize_regex_positions`
    (or something that has a similar function signature) and run postprocessing
    on it. It requires that you also give it a document, which it will tokenize
    using the tokenizing function you give it.

    These postprocessing functions should take a string (i.e. one of the individual tokens),
    but they can return either a string or None. If they return None, the token
    will not appear in the final tokenized result.

    Args:
        postprocess_funcs: A list of postprocessing functions (e.g. :code:`str.lower`)
        tokenize_func: A function that takes raw text and converts it into
            a list of strings and a list of character-level positions
            (e.g. the output of :func:`text_data.tokenize.tokenize_regex_positions`)
        document: The (single) text you want to tokenize.
        tokenized_docs: The tokenized results
            (e.g. the output of :func:`text_data.tokenize.tokenize_regex_positions`)
    """
    post_tokens = []
    post_spans = []
    # there's probably a more elegant way to do this, but default for a,b in x doesn't
    # work here because spans is a tuple
    tokens, spans = tokenize_func(document)
    for token, span in zip(tokens, spans):
        func_result = token
        for func in postprocess_funcs:
            if func_result is not None:
                func_result = func(func_result)  # type: ignore
        if func_result is not None:
            post_tokens.append(func_result)
            post_spans.append(span)
    return post_tokens, post_spans


def corpus_tokenizer(
    regex_patten: str,
    postprocess_funcs: List[Callable[[str], Optional[str]]],
    inverse_match: bool = False,
) -> Callable[[str], TokenizeResult]:
    r"""This is designed to make it easy to build a custom tokenizer for :class:`text_data.index.Corpus`.

    It acts as a combination of :func:`~text_data.tokenize.tokenize_regex_positions` and
    :func:`~text_data.tokenize.postprocess_positions`, making it simple to create
    tokenizers for :class:`text_data.index.Corpus`.

    In other words, if you pass the tokenizer a regular expression pattern, set :code:`inverse_match`
    as you would for :func:`~text_data.tokenize.tokenize_regex_positions`, and add
    a list of postprocessing functions as you would for :func:`~text_data.tokenize.postprocess_positions`,
    this tokenizer will return a function that you can use directly as an argument in :class:`text_data.index.Corpus`.

    Examples:
        Let's say that we want to build a tokenizing function that splits on vowels or whitespace.
        We also want to lowercase all of the remaining words:

        >>> split_vowels = corpus_tokenizer(r"[aeiou\s]+", [str.lower], inverse_match=True)
        >>> split_vowels("Them and you")
        (['th', 'm', 'nd', 'y'], [(0, 2), (3, 4), (6, 8), (9, 10)])

        You can additionally use this function to remove stopwords, although
        `I generally would recommend against it <http://languagelog.ldc.upenn.edu/myl/Monroe.pdf>`_.
        The postprocessing functions optionally return a string or a :code:`NoneType`,
        and :code:`None` values simply don't get tokenized:

        >>> skip_stopwords = corpus_tokenizer(r"\w+", [lambda x: x if x != "the" else None])
        >>> skip_stopwords("I ran to the store")
        (['I', 'ran', 'to', 'store'], [(0, 1), (2, 5), (6, 8), (13, 18)])
    """
    tokenize_func = functools.partial(
        tokenize_regex_positions, regex_patten, inverse_match=inverse_match
    )
    return functools.partial(postprocess_positions, postprocess_funcs, tokenize_func)


#: This is the default tokenizer for :code:`text_data.index.Corpus`.
#:
#: It simply splits on words (:code:`"\w+"`) and lowercases words.
default_tokenizer = corpus_tokenizer(r"\w+", [str.lower])
