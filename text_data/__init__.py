"""Top-level package for Text Data."""
import collections
import functools
import itertools
import sys
from typing import (
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from IPython import display
import numpy as np

from text_data import core, tokenize
from text_data.query import Query, QueryItem

__author__ = """Max Lee"""
__email__ = "maxbmhlee@gmail.com"
__version__ = "0.1.0"

# represents the return result of a query position
PositionResult = collections.namedtuple(
    "PositionResult", "doc_id first_idx last_idx raw_start raw_end"
)
SearchResult = Union[int, PositionResult]
CorpusClass = TypeVar("CorpusClass", bound="Corpus")


class WordIndex:
    """This is a class designed to contain quick lookups of words and phrases.

    It mainly provides convenience lookups on top of `core.PositionalIndex`
    and is designed solely for use inside of `Corpus`.

    Args:
        tokenized_documents: A list of documents, where each document is a list of words.
        indexed_locations: The actual positions
    """

    def __init__(
        self,
        tokenized_documents: List[List[str]],
        indexed_locations: Optional[List[Tuple[int, int]]],
    ):
        self.index = core.PositionalIndex(tokenized_documents, indexed_locations)

    def __len__(self):
        return len(self.index)

    @property
    def vocab(self) -> Set[str]:
        """Returns all of the unique words in the index."""
        return self.index.vocabulary()

    @property
    def vocab_size(self) -> int:
        """Returns the total number of unique words in the dictionary."""
        return self.index.vocab_size

    @property
    def num_words(self) -> int:
        """Returns the total number of words in the dictionary (not just unique)."""
        return self.index.num_words

    def most_common(self, num_words: Optional[int] = None) -> List[Tuple[str, int]]:
        """Returns the most common items.

        This is essentially `collections.Counter.most_common`.

        Args:
            num_words: The number of words you return. If you enter None
            or you enter a number larger than the total number of words,
            returns all of the words, in sorted order from most common to least common.
        """
        return self.index.most_common(num_words)

    def word_count(self, word: str) -> int:
        """Returns the total number of times the word appeared.

        Defaults to 0 if the word never appeared.

        Args:
            word: The string word (or phrase).
        """
        return self.index.word_count(word)

    def word_frequency(self, word: str) -> float:
        """Returns the frequency in which the word appeared.

        Args:
            word: The string word or phrase.
        """
        return self.index.word_frequency(word)

    def document_count(self, word: str) -> int:
        """Returns the total number of documents a word appears in."""
        return self.index.document_count(word)

    def document_frequency(self, word: str) -> float:
        """Returns the frequency in which a word appears in a document."""
        return self.index.document_frequency(word)

    def term_count(self, word: str, document: int) -> int:
        """Returns the total number of times a word appeared in a document."""
        return self.index.term_count(word, document)


class Corpus(WordIndex):
    """This class holds the core data behind text_data.

    It holds the raw text, the index, and the tokenized text.
    Using it, you can compute statistics about the corpus,
    you can query the corpus, or you can visualize some findings.

    The separator, prefix and suffix are all designed for updating
    n-gram indexes (just using default values.)

    Args:
        documents: A list of raw text items (not tokenized)
        tokenizer: A function to tokenize the documents
        sep: The separator you want to use for computing n-grams. Defaults to " "
        prefix: The prefix for n-grams. Defaults to "".
        suffix: The suffix for n-grams. Defaults to "".
    """

    def __init__(
        self,
        documents: List[str],
        tokenizer: Callable[
            [str], tokenize.TokenizeResult
        ] = tokenize.default_tokenizer,
        sep: Optional[str] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        include_positions: bool = True,
    ):
        self.documents = documents
        self.tokenizer = tokenizer
        if len(documents) > 0:
            tokenized_docs = [tokenizer(doc) for doc in documents]
            # for some reason mypy doesn't get that this converts from a list of tuples to a tuple of lists
            words, positions = map(list, zip(*tokenized_docs))  # type: ignore
        else:
            words, positions = [], []
        self.tokenized_documents = words
        self.ngram_indexes: Dict[int, WordIndex] = {}
        self.ngram_sep = sep
        self.ngram_prefix = prefix
        self.ngram_suffix = suffix
        super().__init__(self.tokenized_documents, positions)  # type: ignore

    @classmethod
    def chunks(
        cls: Type[CorpusClass],
        documents: Iterator[str],
        tokenizer: Callable[
            [str], tokenize.TokenizeResult
        ] = tokenize.default_tokenizer,
        sep: Optional[str] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        chunksize: int = 1_000_000,
    ) -> Generator[CorpusClass, None, None]:
        """Iterates through documents, yielding a `Corpus` with `chunksize` documents.

        This is designed to allow you to technically use `Corpus` on large
        document sets. However, you should note that searching for documents
        will only work within the context of the current chunk.

        The same is true for any frequency metrics. As such, you should probably
        limit metrics to raw counts.

        Args:
            documents: A list of raw text items (not tokenized)
            tokenizer: A function to tokenize the documents
            sep: The separator you want to use for computing n-grams. Defaults to " "
            prefix: The prefix for n-grams. Defaults to "".
            suffix: The suffix for n-grams. Defaults to "".
            chunksize: The number of documents in each chunk.
        """
        if chunksize < 1:
            raise ValueError("The chunksize must be a positive, non-zero integer.")
        current_chunksize = 0
        current_document_set: List[str] = []
        for document in documents:
            if current_chunksize == chunksize:
                yield cls(current_document_set, tokenizer, sep, prefix, suffix)
                current_document_set = []
                current_chunksize = 0
            current_chunksize += 1
            current_document_set.append(document)

    def add_ngram_index(
        self,
        n: int = 1,
        default: bool = True,
        sep: Optional[str] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ):
        """Adds an n-gram index to the corpus.

        This is intended solely for being able to compute statistics
        on e.g. bigrams or trigrams.

        **Note**: If you have already added an n-gram index for the corpus,
        this will re-index.

        Args:
            n: The number of n-grams (defaults to unigrams)
            default: If true, will keep the values stored in init (including defaults)
            sep: The separator in between words (if storing n-grams)
            prefix: The prefix before the first word of each n-gram
            suffix: The suffix after the last word of each n-gram
        """
        if default:
            ngram_words = core.ngrams_from_documents(
                self.tokenized_documents,
                n,
                self.ngram_sep,
                self.ngram_prefix,
                self.ngram_suffix,
            )
        else:
            ngram_words = core.ngrams_from_documents(
                self.tokenized_documents, n, sep, prefix, suffix
            )
        self.ngram_indexes[n] = WordIndex(ngram_words, None)

    def update(self, new_documents: List[str]):
        """Updates the indexes for the corpus, given a new set of documents.

        This also updates the n-gram indexes.

        Args:
            new_documents: A list of new documents. The tokenizer used is the same
            tokenizer used to initialize the corpus.
        """
        words, positions = map(
            list, zip(*[self.tokenizer(doc) for doc in new_documents])  # type: ignore
        )
        self.index.add_documents(words, positions)
        self.documents += new_documents
        self.tokenized_documents += words
        for ngram, index in self.ngram_indexes.items():
            ngram_tok = core.ngrams_from_documents(
                words,
                ngram,
                self.ngram_sep,
                self.ngram_prefix,
                self.ngram_suffix,
            )
            index.index.add_documents(ngram_tok)

    def _yield_subquery_document_results(
        self, subquery: List[QueryItem]
    ) -> Generator[Set[int], None, None]:
        """Yields sets of documents given QueryItem objects."""
        for search_item in subquery:
            if search_item.exact:
                yield self.index.find_documents_with_phrase(search_item.words)
            else:
                yield self.index.find_documents_with_words(search_item.words)

    def _yield_subquery_phrase_results(
        self, subquery: List[QueryItem]
    ) -> Generator[Set[PositionResult], None, None]:
        """Yields sets of positions in format of PositionResult."""
        for search_item in subquery:
            if search_item.exact:
                yield {
                    PositionResult(*res)
                    for res in self.index.find_phrase_positions(search_item.words)
                }
            else:
                yield {
                    PositionResult(*res)
                    for res in self.index.find_wordlist_positions(search_item.words)
                }

    def _search_item(
        self,
        yield_function: Callable[
            [List[QueryItem]], Generator[Set[SearchResult], None, None]
        ],
        query_instance: Query,
    ) -> Set[SearchResult]:
        """Internal function for search_documents and search_occurrences."""
        query_results = set()
        query_docs: Set[int] = set()
        for subquery in query_instance.queries:
            subquery_results = functools.reduce(
                lambda x, y: x.union(y), yield_function(subquery)
            )
            # we'll add *all* the positions to set
            # and only return the ones with the relevant docs
            query_results.update(subquery_results)
            subquery_docs = {
                q.doc_id if isinstance(q, PositionResult) else q
                for q in subquery_results
            }
            modifier_type = {x.modifier for x in subquery}
            if modifier_type == {"AND"}:
                query_docs.intersection_update(subquery_docs)
            elif modifier_type == {"OR"}:
                query_docs.update(subquery_docs)
            elif modifier_type == {"NOT"}:
                query_docs -= subquery_docs
        if all((isinstance(q, PositionResult) for q in subquery_results)):
            return {q for q in query_results if q.doc_id in query_docs}  # type: ignore
        return query_docs  # type: ignore

    def search_documents(
        self,
        query: str,
        query_tokenizer: Callable[[str], List[str]] = tokenize.query_tokenizer,
    ) -> Set[int]:
        """Search documents from a query.

        In order to figure out the intracacies of writing queries,
        you should view `text_data.query.Query`. In general,
        standard boolean (AND, OR, NOT) searches work perfectly reasonably.
        You should generally not need to set `query_tokenizer` to anything
        other than the default (string split).

        This produces a set of unique documents, where each document is
        the index of the document. To view the documents by their ranked importance
        (ranked largely using TF-IDF), use `ranked_documents`.

        Args:
            query: A string boolean query (as defined in `text_data.Query`)
            query_tokenizer: A function to tokenize the words in your query. This
                allows you to optionally search for words in your index that include
                spaces (since it defaults to string.split).
        """
        return self._search_item(  # type: ignore
            self._yield_subquery_document_results,  # type: ignore
            Query(query, query_tokenizer),
        )

    def search_occurrences(
        self,
        query: str,
        query_tokenizer: Callable[[str], List[str]] = tokenize.query_tokenizer,
    ) -> Set[PositionResult]:
        """Search for matching positions within a search.

        This allows you to figure out all of the occurrences
        matching your query. In addition, this is used internally
        to display search results.

        Args:
            query: The string query. See `text_data.query.Query` for details.
            query_tokenizer: The tokenizing function for the query.
            See `text_data.query.Query` or `text_data.Corpus.search_documents`
            for details.

        Returns:
            A set of tuples, where each tuple contains (in order)
            the index of the document (within self.documents),
            the starting index of the match (within self.tokenized_documents[document id]),
            the ending index of the match,
            the starting index of the text (within self.documents), and the ending index of the text.
        """
        return self._search_item(  # type: ignore
            self._yield_subquery_phrase_results,  # type: ignore
            Query(query, query_tokenizer),
        )

    def ranked_search(
        self,
        query_string: str,
        query_tokenizer: Callable[[str], List[str]] = tokenize.query_tokenizer,
    ) -> List[List[PositionResult]]:
        """This produces a list of search responses in ranked order.

        More specifically, the documents are ranked in order of the
        sum of the TF-IDF scores for each word in the query
        (with the exception of words that are negated using a NOT operator).

        To compute the TF-IDF scores, I simply have computed the dot products
        between the raw query counts and the TF-IDF scores of all the unique
        words in the query. This is roughly equivalent to the `ltn.lnn`
        normalization scheme [described in Manning](https://nlp.stanford.edu/IR-book/html/htmledition/document-and-query-weighting-schemes-1.html).
        (The catch is that I have normalized the term-frequencies in the document
        to the length of the document.)

        There is one result for each document. This result is ordered
        first by the number of words in the match and second by the word's
        proximity to the next word in the document.

        Args:
            query: Query string
            query_tokenizer: Function for tokenizing the results.

        Returns:
            A list of tuples, each in the same format as `search_occurrences`.
        """
        query_results = []
        query = Query(query_string, query_tokenizer)
        # sorting the results allows grouping using itertools + allows linear distance calc
        all_matches = sorted(
            self._search_item(
                self._yield_subquery_phrase_results, query  # type: ignore
            )
        )
        # runs into divide by 0 error on idf computation without this being explicit
        if all_matches == []:
            return []
        # create a list of words from the query where the word is not being excluded
        query_tokens = [
            word
            for subquery in query.queries
            for q_item in subquery
            for word in q_item.words
            if q_item.modifier != "NOT"
        ]
        query_counts = collections.Counter(query_tokens)
        query_words = list(query_counts)
        query_freqs = np.array([query_counts[w] for w in query_words])
        idf = np.array([len(self) / self.document_count(w) for w in query_words])
        for doc, matches in itertools.groupby(
            all_matches, key=lambda x: x.doc_id  # type: ignore
        ):
            term_counts = np.array([self.term_count(w, doc) for w in query_words])
            term_freqs = term_counts / len(self.tokenized_documents[doc])
            log_term_freqs = np.log(term_freqs + 1)
            tfidf = np.dot(query_freqs, log_term_freqs * idf)
            position_match, num_dist = self._sort_positions(matches)  # type: ignore
            query_results.append((tfidf, -num_dist, position_match))
        # sort results by TF-IDF scores, followed by the smallest distance
        # followed by the document order
        return [result for *sort_terms, result in sorted(query_results, reverse=True)]

    def _sort_positions(
        self, matches: Iterator[PositionResult]
    ) -> Tuple[List[PositionResult], int]:
        """Chooses one positional match for a given document.

        Internal method for ranked positional search. Sorts by the longest
        phrase, followed by for the phrase that is closest to another phrase.
        """
        results: List[PositionResult] = []
        max_len = None
        min_closest = None
        prev_result = None
        for match in matches:
            item_length = match.last_idx - match.first_idx
            if prev_result is None:
                dist_to_prev = None
            else:
                dist_to_prev = match.first_idx - prev_result.last_idx
            # first set the current match to the choice
            # if it is longer than remainders
            if max_len is None or item_length > max_len:
                max_len = item_length
                results.insert(0, match)
            if isinstance(dist_to_prev, int):
                if min_closest is None or dist_to_prev < min_closest:
                    min_closest = dist_to_prev
                    # only change current result if the length of the phrase
                    # is not less than the max length
                    if item_length == max_len:
                        results.insert(0, match)
            prev_result = match
            # don't insert twice
            if results[0] != match:
                results.append(match)
        # this maps distance to a guaranteed number and favors words with multiple items
        num_dist = min_closest if min_closest is not None else sys.maxsize
        # above technically returns an optional position, but it's
        # a private method that only runs when the position isn't none
        return results, num_dist  # type: ignore

    def display_search_results(
        self,
        search_query: str,
        query_tokenizer: Callable[[str], List[str]] = tokenize.query_tokenizer,
        max_results: Optional[int] = None,
        window_size: Optional[int] = None,
    ) -> display.HTML:
        """Shows the results of a ranked query.

        This function runs a query and then renders the result in human-readable
        HTML. For each result, you will get a document ID and the count of the result.

        In addition, all of the matching occurrences of phrases or words you
        searched for will be highlighted in bold. You can optionally decide
        how many results you want to return and how long you want each result to be
        (up to the length of the whole document).

        Args:
            search_query: The query you're searching for
            query_tokenizer: The tokenizer for the query
            max_results: The maximum number of results. If None, returns all results.
            window_size: The number of characters you want to return around the matching phrase.
            If None, returns the entire document.
        """
        return display.HTML(
            self._show_html_occurrences(
                search_query, query_tokenizer, max_results, window_size
            )
        )

    def _show_html_occurrences(
        self,
        search_query: str,
        query_tokenizer: Callable[[str], List[str]] = tokenize.query_tokenizer,
        max_results: Optional[int] = None,
        window_size: Optional[int] = None,
    ) -> str:
        """Plots all of the search matches as HTML phrases.

        Each result contains information about the order in which it appeared,
        the document ID, and bolds each matching phrase. The results are centered
        around the match that appears to be closest based on _sort_positions.

        Args:
            search_query: The query you're searching for
            query_tokenizer: The tokenizer for the query
            max_results: The maximum number of results. If None, returns all results.
            window_size: The number of characters you want to return around the matching phrase.
            If None, returns the entire document.
        """
        results = ""
        query_results = self.ranked_search(search_query, query_tokenizer)
        max_results = len(query_results) if max_results is None else max_results
        for count, doc in enumerate(self.ranked_search(search_query, query_tokenizer)):
            if count >= max_results:
                break
            if len(doc) > 0:
                highlight_doc = doc[0]
                results += f"<p><b>Showing Result {count} (Document ID {highlight_doc.doc_id})</b></p>"
                results += "<p style='white-space=pre-wrap;'>"
                sel_doc = self.documents[highlight_doc.doc_id]
                # replace the window size with the document size if it's set to None
                doc_window = window_size if window_size is not None else len(sel_doc)
                # we want to center the output around the best match in the document
                start_window = max(0, highlight_doc.raw_start - doc_window)
                end_window = min(len(sel_doc), highlight_doc.raw_end + doc_window)
                current_idx = start_window
                for item in sorted(
                    filter(
                        lambda x: x.raw_end >= start_window
                        and x.raw_start <= end_window,
                        doc,
                    )
                ):
                    if current_idx == start_window:
                        actual_start = min(
                            highlight_doc.raw_start - doc_window, item.raw_start
                        )
                        if actual_start > 0:
                            results += "<b>&hellip;</b>"
                    if current_idx < item.raw_start:
                        results += sel_doc[current_idx : item.raw_start]
                    results += f"<b>{sel_doc[item.raw_start:item.raw_end]}</b>"
                    current_idx = item.raw_end
                raw_end_window = max(current_idx, end_window)
                results += sel_doc[current_idx:raw_end_window]
                if raw_end_window < len(sel_doc):
                    results += "<b>&hellip;</b>"
                results += "</p>"
        return results

    def search_document_count(
        self,
        query_string: str,
        query_tokenizer: Callable[[str], List[str]] = tokenize.query_tokenizer,
    ) -> int:
        """Finds the total number of documents matching a query.

        By entering a search, you can get the total number of documents
        that match the query.

        Args:
            query_string: The query you're searching for
            query_tokenizer: The tokenizer for the query
        """
        return len(self.search_documents(query_string, query_tokenizer))

    def search_document_freq(
        self,
        query_string: str,
        query_tokenizer: Callable[[str], List[str]] = tokenize.query_tokenizer,
    ) -> float:
        """Finds the percentage of documents that match a query.

        Args:
            query_string: The query you're searching for
            query_tokenizer: The tokenizer for the query
        """
        return self.search_document_count(query_string, query_tokenizer) / len(self)

    def search_occurrence_count(
        self,
        query_string: str,
        query_tokenizer: Callable[[str], List[str]] = tokenize.query_tokenizer,
    ) -> int:
        """Finds the total number of matches you have for a query.

        Args:
            query_string: The query you're searching for
            query_tokenizer: The tokenizer for the query
        """
        return len(self.search_occurrences(query_string, query_tokenizer))

    @core.requires_display_extra
    def _render_bar_chart(
        self,
        metric_func: Callable[[str, Callable[[str], List[str]]], Union[float, int]],
        metric_name: str,
        queries: List[str],
        query_tokenizer: Callable[[str], List[str]] = tokenize.query_tokenizer,
    ):
        """Creates a bar chart given a callable that returns a metric.

        Internal for `display_document_count`, `display_document_freqs`, and `display_occurrence_count`.

        Args:
            metric_func: A function that takes in a query and returns a metric
                (e.g. `Corpus.search_document_count`)
            metric_name: The name for the metric (used as an axis label)
            queries: A list of queries
            query_tokenizer: A function for tokenizing the queries
        """
        import altair as alt

        json_data = [
            {"Query": query, metric_name: metric_func(query, query_tokenizer)}
            for query in queries
        ]
        data = alt.Data(values=json_data)
        return (
            alt.Chart(data)
            .mark_bar()
            .encode(x=f"{metric_name}:Q", y=alt.Y("Query:N", sort="-x"))
        )

    @core.requires_display_extra
    def display_document_count(
        self,
        queries: List[str],
        query_tokenizer: Callable[[str], List[str]] = tokenize.query_tokenizer,
    ):
        """Returns a bar chart (in altair) showing the queries with the largest number of documents.

        Args:
            queries: A list of queries (in the same form you use to search for things)
            query_tokenizer: The tokenizer for the query
        """
        return self._render_bar_chart(
            self.search_document_count, "Number of documents", queries, query_tokenizer
        )

    @core.requires_display_extra
    def display_document_frequency(
        self,
        queries: List[str],
        query_tokenizer: Callable[[str], List[str]] = tokenize.query_tokenizer,
    ):
        """Displays a bar chart showing the percentages of documents with a given query.

        Args:
            queries: A list of queries
            query_tokenizer: A tokenizer for each query
        """
        return self._render_bar_chart(
            self.search_document_freq, "Document Frequency", queries, query_tokenizer
        )

    @core.requires_display_extra
    def display_occurrence_count(
        self,
        queries: List[str],
        query_tokenizer: Callable[[str], List[str]] = tokenize.query_tokenizer,
    ):
        """Display the number of times a query matches.

        Args:
            queries: A list of queries
            query_tokenizer: The tokenizer for the query
        """
        return self._render_bar_chart(
            self.search_occurrence_count, "Number of matches", queries, query_tokenizer
        )

    def _document_html(self, doc_idx: int) -> str:
        """Return the HTML to display a document.

        Internal for `display_document` and `display_documents`.
        """
        content = self.documents[doc_idx]
        return f"<p><b>Document at index {doc_idx}</b></p><p>{content}</p>"

    def display_document(self, doc_idx: int) -> display.HTML:
        """Print an entire document, given its index.

        Args:
            doc_idx: The index of the document
        """
        return display.HTML(self._document_html(doc_idx))

    def display_documents(self, documents: List[int]) -> display.HTML:
        """Display a number of documents, at the specified indexes.

        Args:
            documents: A list of document indexes.
        """
        html = "".join([self._document_html(idx) for idx in documents])
        return display.HTML(html)
