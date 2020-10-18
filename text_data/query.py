"""This builds and runs search queries for :class:`text_data.index.Corpus`.

For the most part, you won't be using this directly. Instead, you'll likely
be using :class:`text_data.index.Corpus`. However, viewing the :code:`__repr__`
for the query you're running can be helpful for debugging or validating
queries.
"""
import collections
import re
from typing import Callable, List

QUERY_TERMS = {"AND", "OR", "NOT"}
#: This represents an set of words you want to search for.
#:
#: Each query item has attached to it a set of words,
#: an identifier stating whether the query terms are part of
#: an exact phrase (i.e. whether the order matters)
#: and what kind of query (a boolean AND query, a boolean OR query, or a boolean NOT query),
#: is being performed on the query.
#:
#: Args:
#:      words (List[str]): A list of words representing all of the words that will be searched for.
#:      exact (bool): Whether the search terms are part of an exact phrase match
#:      modifier (str): The boolean query (AND, OR, or NOT)
QueryItem = collections.namedtuple("QueryItem", "words exact modifier")


class Query:
    r"""Represents a query. This is used internaly by :class:`text_data.index.Corpus` to handle searching.

    The basic formula for writing queries should be familiar; all of the
    queries are simple boolean phrases. But here are more complete specifications:

    In order to search for places where two words appeared, you simply need
    to type the two words::

        Query("i am")

    Searches using this query will look for documents where the words "i"
    and "am" both appeared. To have them look for places where either
    word appeared, use an "OR" query::

        Query("i OR am")

    Alternatively, you can look for documents where one word occurred but the other
    didn't using a NOT query::

        Query("i NOT am")

    To search for places where the phrase "i am" appeared, use quotes::

        Query("'i am'")

    You can use AND queries to limit the results of previous sets of queries.
    For instance::

        Query("i OR am AND you")

    will find places where either where "you" and *either* "I" or "am" appeared.

    In order to search for the literal words 'AND', 'OR', or 'NOT',
    you must encapsulate them in quotes::

        Query("'AND'")

    Finally, you may customize the way your queries are parsed by passing
    a tokenizer. By default, :code:`Query` identifies strings of text
    that it needs to split and uses :code:`str.split` to split the strings.
    But you can change how to split the text, which can be helpful/necessary
    if the words you're searching for have spaces in them. For instance,
    this will split the words you're querying by spaces, unless the words
    are 'united states'::

        >>> import re
        >>> us_phrase = re.compile(r"(united states|\S+)")
        >>> Query("he is from the united states", query_tokenizer=us_phrase.findall)
        <Query ([[QueryItem(words=['he', 'is', 'from', 'the', 'united states'], exact=False, modifier='OR')]])>

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
        query_tokenizer: Callable[[str], List[str]] = str.split,
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
        query_tokenizer: Callable[[str], List[str]] = str.split,
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
