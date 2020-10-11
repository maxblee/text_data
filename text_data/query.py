"""Build and run search queries for `Corpus`."""
import collections
import re
from typing import Callable, List

from text_data import tokenize

QUERY_TERMS = {"AND", "OR", "NOT"}
QueryItem = collections.namedtuple("QueryItem", "words exact modifier")


class Query:
    """Represents a query.

    This is used internaly by `Corpus` to handle searching.

    To parse a query, you need to use the following syntax:
        - Exact phrase searches should be enclosed in quotes
        - Inexact phrase searches should not be in quotes.
        - You can use any of the query terms to combine queries:
            - AND to find things matching *all* of the queries
            - OR to find things matching *any* of the queries
            - NOT to find things matching one query but *not* the other
        - To include any of the above query terms as search parameters,
        you must encapsulate them in quotes. You must do the same for phrases
        ending in spaces *even if you have a customized tokenizer*.
        - Query terms *must* be in all caps. This is designed to make unintentionally
        entering a query term less likely without sacrificing readability.

    Args:
        query_string: The human-readable query
        query_tokenizer: A function to tokenize phrases in the query
            (Defaults to string.split).
            **Note:** This specifically tokenizes individual phrases in the query.
            As a result, the function does not need to handle quotations.
    """

    def __init__(
        self,
        query_string: str,
        query_tokenizer: Callable[[str], List[str]] = tokenize.query_tokenizer,
    ):
        # starting with a key word should raise an error
        if (
            re.search(fr"^\s*({'|'.join(QUERY_TERMS)})(?:\s+|$)", query_string)
            is not None
        ):
            raise ValueError("You cannot use a keyword at the beginning of the query")
        # this holds outputs of queries, as set objects
        self.queries = []
        self.raw_query = query_string
        current_idx = 0
        # set the first
        last_modifier = "OR"
        term_regex = re.compile(fr"\s+({'|'.join(QUERY_TERMS)})\s+")
        for term in term_regex.finditer(query_string):
            query_items = query_string[current_idx : term.start()].strip()
            self.queries.append(
                self._parse_subquery(query_items, last_modifier, query_tokenizer)
            )
            last_modifier = term.group(1)
            current_idx = term.end()
        end_query = query_string[current_idx:].strip()
        self.queries.append(
            self._parse_subquery(end_query, last_modifier, query_tokenizer)
        )

    def _parse_subquery(
        self,
        query: str,
        last_modifier: str,
        query_tokenizer: Callable[[str], List[str]] = tokenize.query_tokenizer,
    ) -> List[QueryItem]:
        """This parses queries between QUERY_TERM objects. Internal to init.

        Args:
            query: The subquery
            last_modifier: Specifies the last query term that was used (or OR if none)
            query_tokenizer: Passed directly from __init__
        """
        matches = []
        current_idx = 0
        quote_regex = re.compile(
            r"(?:\s|^)(\'(?P<single>[^\']+)\'|\"(?P<double>[^\"]+)\")(?:\s|$)"
        )
        for exact_match in quote_regex.finditer(query):
            single_quote = exact_match.group("single")
            double_quote = exact_match.group("double")
            quoted_matl = single_quote if single_quote is not None else double_quote
            pre_match = query[current_idx : exact_match.start()].strip()
            if pre_match != "":
                matches.append(
                    QueryItem(query_tokenizer(pre_match), False, last_modifier)
                )
            matches.append(QueryItem(query_tokenizer(quoted_matl), True, last_modifier))
            current_idx = exact_match.end()
        post_match = query[current_idx:].strip()
        if post_match != "":
            matches.append(QueryItem(query_tokenizer(post_match), False, last_modifier))
        return matches

    def __repr__(self):
        return f"<Query ({self.queries})>"

    def __str__(self):
        return self.raw_query
