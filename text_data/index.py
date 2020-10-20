"""This module handles the indexing of `text_data`.

Its two classes — :code:`WordIndex` and :code:`Corpus` — form the central part
of this library.

:class:`text_data.index.WordIndex` indexes lists of documents — which themselves form
lists of words or phrases — and offers utilities for performing
statistical calculations on your data.

Using the index, you can find out how many times a given word appeared in a
document or do more complicated things, like finding the TF-IDF values
for every single word across all of the documents in a corpus. In addition
to offering a bunch of different ways to compute statistics, :code:`WordIndex`
also offers capabilities for creating new :code:`WordIndex` objects — something
that can be very helpful if you're trying to figure out what
makes a set of documents different from some other documents.

The :class:`text_data.index.Corpus`, meanwhile, is a wrapper over :code:`WordIndex` that offers tools for searching
through sets of documents. In addition, it offers tools for visually seeing the results of search queries.
"""
# https://stackoverflow.com/questions/33533148/how-do-i-type-hint-a-method-with-the-type-of-the-enclosing-class
from __future__ import annotations

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

#: This represents the position of a word or phrase within a document.
#:
#: See :meth:`text_data.index.Corpus.search_occurrences` for more details and an example.
#:
#: Args:
#:      doc_id (int): The index of the document within the index
#:      first_idx (int): The index of the first word within the tokenized document at :code:`corpus.tokenized_documents[doc_id]`.
#:      last_idx (int): The index of the last word within the tokenized document at :code:`corpus.tokenized_documents[doc_id]`.
#:      raw_start (Optional[int]): The starting character-level index within the raw string document at :code:`corpus.documents[doc_id]`.
#:      raw_end (Optional[int]): The index after the ending character-level index within the raw string document at :code:`corpus.documents[doc_id]`.
PositionResult = collections.namedtuple(
    "PositionResult", "doc_id first_idx last_idx raw_start raw_end"
)

SearchResult = Union[int, PositionResult]
CorpusClass = TypeVar("CorpusClass", bound="Corpus")


class WordIndex:
    r"""An inverted, positional index containing the words in a corpus.

    This is designed to allow people to be able to quickly compute statistics
    about the language used across a corpus. The class offers a couple of broad
    strategies for understanding the ways in which words are used across documents.

    **Manipulating Indexes**
    These functions are designed to allow you to create new indexes based
    on ones you already have. They operate kind of like slices and filter
    functions in :code:`pandas`, where your goal is to be able to
    create new data structures that you can analyze independently from
    ones you've already created. Most of them can also be used
    with method chaining. However, some of these functions remove
    positional information from the index, so be careful.

    - :meth:`~text_data.index.WordIndex.copy` creates an identical copy of a :code:`WordIndex`
      object.
    - :meth:`~text_data.index.WordIndex.slice` and :meth:`~text_data.index.WordIndex.split_off`
      both take sets of document indexes and create new indexes with only those
      documents.
    - :meth:`~text_data.index.WordIndex.add_documents` allows you to add new
      documents into an existing :code:`WordIndex` object.
      :meth:`~text_data.index.WordIndex.concatenate` similarly combines
      :code:`WordIndex` objects into a single :code:`WordIndex`.
    - :meth:`~text_data.index.WordIndex.flatten` takes a :code:`WordIndex`
      and returns an identical index that only has one document.
    - :meth:`~text_data.index.WordIndex.skip_words` takes a set of words
      and returns a :code:`WordIndex` that does not have those words.
    - :meth:`~text_data.index.WordIndex.reset_index` changes the index of words.

    **Corpus Information**

    A number of functions are designed to allow you to look up information
    about the corpus. For instance, you can collect a sorted list
    or a set of all the unique words in the corpus. Or you can get a list
    of the most commonly appearing elements:

    * :attr:`~text_data.index.WordIndex.vocab` and :attr:`~text_data.index.WordIndex.vocab_list`
      both return the unique words or phrases appearing in the index.
    * :attr:`~text_data.index.WordIndex.vocab_size` gets the number of unique words in the index.
    * :attr:`~text_data.index.WordIndex.num_words` gets the total number of words in the index.


    **Word Statistics**

    These allow you to gather statistics about single words
    or about word, document pairs. For instance, you can see
    how many words there are in the corpus, how many unique words there are,
    or how often a particular word appears in a document.

    The statistics generally fit into four categories.
    The first category computes statistics about how often a specific word appears
    in the corpus as a whole. The second category computes statistics
    about how often a specific word appears in a specific document.
    The third and fourth categories echo those first two categories
    but perform the statistics efficiently across the corpus as a whole,
    creating 1-dimensional numpy arrays in the case of the word-corpus
    statistics and 2-dimensional numpy arrays in the case of the word-document
    statistics. Functions in these latter two categories all end in
    :code:`_vector` and :code:`_matrix` respectively.

    Here's how those statistics map to one another:

    .. list-table:: Word Statistics
        :widths: 40 40 40 40
        :header-rows: 1

        * - Word-Corpus
          - Word-Document
          - Vector
          - Matrix
        * - :meth:`~text_data.index.WordIndex.word_count`
          - :meth:`~text_data.index.WordIndex.term_count`
          - :meth:`~text_data.index.WordIndex.word_count_vector`
          - :meth:`~text_data.index.WordIndex.count_matrix`
        * - :meth:`~text_data.index.WordIndex.word_frequency`
          - :meth:`~text_data.index.WordIndex.term_frequency`
          - :meth:`~text_data.index.WordIndex.word_freq_vector`
          - :meth:`~text_data.index.WordIndex.frequency_matrix`
        * - :meth:`~text_data.index.WordIndex.document_count`
          -
          - :meth:`~text_data.index.WordIndex.doc_count_vector`
          -
        * - :meth:`~text_data.index.WordIndex.document_frequency`
          -
          - :meth:`~text_data.index.WordIndex.doc_freq_vector`
          -
        * - :meth:`~text_data.index.WordIndex.idf`
          -
          - :meth:`~text_data.index.WordIndex.idf_vector`
          -
        * - :meth:`~text_data.index.WordIndex.odds_word`
          - :meth:`~text_data.index.WordIndex.odds_document`
          - :meth:`~text_data.index.WordIndex.odds_vector`
          - :meth:`~text_data.index.WordIndex.odds_matrix`
        * - :code:`__contains__`
          - :meth:`~text_data.index.WordIndex.doc_contains`
          -
          - :meth:`~text_data.index.WordIndex.one_hot_matrix`
        * -
          -
          -
          - :meth:`~text_data.index.WordIndex.tfidf_matrix`

    In the case of the vector and matrix calculations, the arrays
    represent the unique words of the vocabulary,
    presented in sorted order. As a result, you can safely run
    element-wise calculations over the matrices.

    In addition to the term vector and term-document matrix functions, there is
    :meth:`~text_data.index.WordIndex.get_top_words`, which is designed
    to allow you to find the highest or lowest scores and their associated words along any
    term vector or term-document matrix you please.

    Note:
        For the most part, you will not want to instantiate :code:`WordIndex` directly.
        Instead, you will likely use :class:`~text_data.index.Corpus`, which subclasses
        :code:`WordIndex`.

        That's because :class:`~text_data.index.Corpus` offers utilities for searching through
        documents. In addition, with the help of tools from :py:mod:`text_data.tokenize`,
        instantiating :class:`~text_data.index.Corpus` objects is a bit simpler than
        instantiating :code:`WordIndex` objects directly.

        I particularly recommend that you **do not** instantiate the
        :code:`indexed_locations` directly (i.e. outside of :class:`~text_data.index.Corpus`).
        The only way you can do anything with :code:`indexed_locations` from outside of
        :class:`~text_data.index.Corpus` is by using an internal attribute
        and hacking through poorly documented Rust code.

    Args:
        tokenized_documents: A list of documents where each document is a list of words.
        indexed_locations: A list of documents where each documents contains a list
            of the start end positions of the words in :code:`tokenized_documents`.
    """

    def __init__(
        self,
        tokenized_documents: List[List[str]],
        indexed_locations: Optional[List[Tuple[int, int]]] = None,
    ):
        self.index = core.PositionalIndex(tokenized_documents, indexed_locations)

    def __len__(self):
        return len(self.index)

    def __contains__(self, item: str) -> bool:
        """Determines whether the index has a given word."""
        return item in self.index

    # Index manipulation
    # These functions take create new indexes based on existing ones
    # one of them — split_off — mutates the existing object;
    # the others don't
    @classmethod
    def _from_index(cls, index: core.PositionalIndex) -> WordIndex:
        """Creates an index from the Rust PositionalIndex."""
        # probably a better way to do this, but this just sets an empty data
        # and overrides the value of `self.index`
        cls_item = cls([])
        cls_item.index = index
        return cls_item

    def copy(self) -> WordIndex:
        """This creates a copy of itself."""
        return self._from_index(self.index.copy())

    def reset_index(self, start_idx: Optional[int] = None):
        """An in-place operation that resets the document indexes for this corpus.

        When you reset the index, all of the documents change their
        values, starting at :code:`start_idx` (and incrementing from there). For the most
        part, you will not need to do this, since most of the library
        does not give you the option to change the document indexes. However,
        it may be useful when you're using :meth:`~text_data.index.WordIndex.slice`
        or :meth:`~text_data.index.WordIndex.split_off`.

        Args:
            start_idx: The first (lowest) document index you want to set.
                Values must be positive. Defaults to 0.
        """
        self.index.reset_index(start_idx)

    def concatenate(self, other: WordIndex, ignore_index: bool = True) -> WordIndex:
        """Creates a :code:`WordIndex` object with the documents of both this object and the other.

        See :func:`text_data.multi_corpus.concatenate` for more details.

        Args:
            ignore_index: If set to :code:`True`, which is the default, the
                document indexes will be re-indexed starting from 0.

        Raises:
            ValueError: If :code:`ignore_index` is set to :code:`False` and some
                of the indexes overlap.
        """
        concat = self.index.concat(self.index, other.index, ignore_index)
        return self._from_index(concat)

    def add_documents(
        self,
        tokenized_documents: List[List[str]],
        indexed_locations: Optional[List[Tuple[int, int]]] = None,
    ):
        """This function updates the index with new documents.

        It operates similarly to :meth:`text_data.index.Corpus.update`,
        taking new documents and mutating the existing one.

        Example:
            >>> tokenized_words = ["im just a simple document".split()]
            >>> index = WordIndex(tokenized_words)
            >>> len(index)
            1
            >>> index.num_words
            5
            >>> index.add_documents(["now im an entire corpus".split()])
            >>> len(index)
            2
            >>> index.num_words
            10
        """
        self.index.add_documents(tokenized_documents, indexed_locations)

    def skip_words(self, words: Set[str]) -> WordIndex:
        """Creates a :code:`WordIndex` without any of the skipped words.

        This enables you to create an index that does not contain rare
        words, for example. The index will not have any positions
        associated with them, so be careful when implementing
        it on a :class:`text_data.index.Corpus` object.

        Example:
            >>> skip_words = {"document"}
            >>> corpus = Corpus(["example document", "document"])
            >>> "document" in corpus
            True
            >>> without_document = corpus.skip_words(skip_words)
            >>> "document" in without_document
            False
        """
        index = self.index.skip_words(words)
        return self._from_index(index)

    def flatten(self) -> WordIndex:
        """Flattens a multi-document index into a single-document corpus.

        This creates a new :code:`WordIndex` object stripped of any positional
        information that has a single document in it. However, the list of words
        and their indexes remain.

        Example:
            >>> corpus = Corpus(["i am a document", "so am i"])
            >>> len(corpus)
            2
            >>> flattened = corpus.flatten()
            >>> len(flattened)
            1
            >>> assert corpus.most_common() == flattened.most_common()
        """
        index = self.index.flatten()
        return self._from_index(index)

    def slice(self, indexes: Set[int]) -> WordIndex:
        """Returns an index that just contains documents from the set of words.

        Args:
            indexes: A set of index values for the documents.

        Example:
            >>> index = WordIndex([["example"], ["document"], ["another"], ["example"]])
            >>> sliced_idx = index.slice({0, 2})
            >>> len(sliced_idx)
            2
            >>> sliced_idx.most_common()
            [('another', 1), ('example', 1)]
        """
        return self._from_index(self.index.slice(indexes))

    def split_off(self, indexes: Set[int]) -> WordIndex:
        """Returns an index with just a set of documents, while removing them from the index.

        Args:
            indexes: A set of index values for the documents.

        Note:
            This removes words from the index inplace. So be make sure you
            want to do that before using this function.

        Example:
            >>> index = WordIndex([["example"], ["document"], ["another"], ["example"]])
            >>> split_idx = index.split_off({0, 2})
            >>> len(split_idx)
            2
            >>> len(index)
            2
            >>> split_idx.most_common()
            [('another', 1), ('example', 1)]
            >>> index.most_common()
            [('document', 1), ('example', 1)]
        """
        return self._from_index(self.index.split_off(indexes))

    # Corpus Information
    # This section returns simple information about a corpus — how many
    # words are there, what the vocabulary is, etc.

    @property
    def vocab(self) -> Set[str]:
        """Returns all of the unique words in the index.

        Example:
            >>> corpus = Corpus(["a cat and a dog"])
            >>> corpus.vocab == {"a", "cat", "and", "dog"}
            True
        """
        return self.index.vocabulary()

    @property
    def vocab_list(self) -> List[str]:
        """Returns a sorted list of the words appearing in the index.

        This is primarily intended for use in matrix or vector functions,
        where the order of the words matters.

        Example:
            >>> corpus = Corpus(["a cat and a dog"])
            >>> corpus.vocab_list
            ['a', 'and', 'cat', 'dog']
        """
        return self.index.vocabulary_list()

    @property
    def vocab_size(self) -> int:
        """Returns the total number of unique words in the corpus.

        Example:
            >>> corpus = Corpus(["a cat and a dog"])
            >>> corpus.vocab_size
            4
        """
        return self.index.vocab_size()

    @property
    def num_words(self) -> int:
        """Returns the total number of words in the corpus (not just unique).

        Example:
            >>> corpus = Corpus(["a cat and a dog"])
            >>> corpus.num_words
            5
        """
        return self.index.num_words

    # Point Estimates
    # This section contains point estimates for the index. This includes
    # statistics related to how often words appear in a particular document,
    # how often words appear across the corpus, etc.
    #
    # Word Statistics
    # These provide statistics about specific words, without
    # requiring any information about the documents.

    def document_count(self, word: str) -> int:
        """Returns the total number of documents a word appears in.

        Example:
            >>> corpus = Corpus(["example document", "another example"])
            >>> corpus.document_count("example")
            2
            >>> corpus.document_count("another")
            1

        Args:
            word: The word you're looking up.
        """
        return self.index.document_count(word)

    def document_frequency(self, word: str) -> float:
        """Returns the percentage of documents that contain a word.

        Example:
            >>> corpus = Corpus(["example document", "another example"])
            >>> corpus.document_frequency("example")
            1.0
            >>> corpus.document_frequency("another")
            0.5

        Args:
            word: The word you're looking up.
        """
        return self.index.document_frequency(word)

    def idf(self, word: str) -> float:
        r"""Returns the inverse document frequency.

        If the number of documents in your :code:`WordIndex`
        :code:`index` is :math:`N` and the document frequency from
        :meth:`~text_data.index.WordIndex.document_frequency` is
        :math:`df`, the inverse document frequency is :math:`\frac{N}{df}`.

        Example:
            >>> corpus = Corpus(["example document", "another example"])
            >>> corpus.idf("example")
            1.0
            >>> corpus.idf("another")
            2.0

        Args:
            word: The word you're looking for.
        """
        return self.index.idf(word)

    def docs_with_word(self, word: str) -> Set[int]:
        """Returns a list of all the documents containing a word.

        Example:
            >>> corpus = Corpus(["example document", "another document"])
            >>> assert corpus.docs_with_word("document") == {0, 1}
            >>> assert corpus.docs_with_word("another") == {1}

        Args:
            word: The word you're looking up.
        """
        return self.index.docs_with_word(word)

    def word_counter(self, word: str) -> Dict[int, int]:
        """Maps the documents containing a word to the number of times the word appeared.

        Examples:
            >>> corpus = Corpus(["a bird", "a bird and a plane", "two birds"])
            >>> corpus.word_counter("a") == {0: 1, 1: 2}
            True

        Args:
            word: The word you're looking up

        Returns:
            A dictionary mapping the document index of the word to the number of times
                it appeared in that document.
        """
        return self.index.word_counter(word)

    def most_common(self, num_words: Optional[int] = None) -> List[Tuple[str, int]]:
        """Returns the most common items.

        This is nearly identical to :code:`collections.Counter.most_common`.
        However, unlike `collections.Counter.most_common`, the values that
        are returned appear in alphabetical order.

        Example:
            >>> corpus = Corpus(["i walked to the zoo", "i bought a zoo"])
            >>> corpus.most_common()
            [('i', 2), ('zoo', 2), ('a', 1), ('bought', 1), ('the', 1), ('to', 1), ('walked', 1)]
            >>> corpus.most_common(2)
            [('i', 2), ('zoo', 2)]

        Args:
            num_words: The number of words you return. If you enter None
                or you enter a number larger than the total number of words,
                it returns all of the words, in sorted order from most common to least common.
        """
        return self.index.most_common(num_words)

    def max_word_count(self) -> Optional[Tuple[str, int]]:
        """Returns the most common word and the number of times it appeared in the corpus.

        Returns :code:`None` if there are no words in the corpus.

        Example:
            >>> corpus = Corpus([])
            >>> corpus.max_word_count() is None
            True
            >>> corpus.update(["a bird a plane superman"])
            >>> corpus.max_word_count()
            ('a', 2)
        """
        return self.index.max_word_count()

    def word_count(self, word: str) -> int:
        """Returns the total number of times the word appeared.

        Defaults to 0 if the word never appeared.

        Example:
            >>> corpus = Corpus(["this is a document", "a bird and a plane"])
            >>> corpus.word_count("document")
            1
            >>> corpus.word_count("a")
            3
            >>> corpus.word_count("malarkey")
            0

        Args:
            word: The string word (or phrase).
        """
        return self.index.word_count(word)

    def word_frequency(self, word: str) -> float:
        """Returns the frequency in which the word appeared in the corpus.

        Example:
            >>> corpus = Corpus(["this is fun", "or is it"])
            >>> np.isclose(corpus.word_frequency("fun"), 1. / 6.)
            True
            >>> np.isclose(corpus.word_frequency("is"), 2. / 6.)
            True

        Args:
            word: The string word or phrase.
        """
        return self.index.word_frequency(word)

    def odds_word(self, word: str, sublinear: bool = False) -> float:
        r"""Returns the odds of seeing a word at random.

        In statistics, the *odds* of something happening are the probability
        of it happening, versus the probability of it not happening,
        that is :math:`\frac{p}{1 - p}`. The "log odds" of
        something happening — the result of using :code:`self.log_odds_word` —
        is similarly equivalent to :math:`log_{2}{\frac{p}{1 - p}}`.

        (The probability in this case is simply the word frequency.)

        Example:
            >>> corpus = Corpus(["i like odds ratios"])
            >>> np.isclose(corpus.odds_word("odds"), 1. / 3.)
            True
            >>> np.isclose(corpus.odds_word("odds", sublinear=True), np.log2(1./3.))
            True

        Args:
            word: The word you're looking up.
            sublinear: If true, returns the
        """
        return self.index.odds_word(word, sublinear)

    # Word-Document Statistics
    #
    #   These provide statistics about words within a particular document.
    def doc_contains(self, word: str, document: int) -> bool:
        """States whether the given document contains the word.

        Example:
            >>> corpus = Corpus(["words", "more words"])
            >>> corpus.doc_contains("more", 0)
            False
            >>> corpus.doc_contains("more", 1)
            True

        Args:
            word: The word you're looking up.
            document: The index of the document.

        Raises:
            ValueError: If the document you're looking up doesn't exist.
        """
        return self.index.doc_contains(word, document)

    def term_count(self, word: str, document: int) -> int:
        """Returns the total number of times a word appeared in a document.

        Assuming the document exists, returns 0 if the word does not
        appear in the document.

        Example:
            >>> corpus = Corpus(["i am just thinking random thoughts", "am i"])
            >>> corpus.term_count("random", 0)
            1
            >>> corpus.term_count("random", 1)
            0

        Args:
            word: The word you're looking up.
            document: The index of the document.

        Raises:
            ValueError: If you selected a document
        """
        return self.index.term_count(word, document)

    def term_frequency(self, word: str, document: int) -> float:
        """Returns the proportion of words in document :code:`document` that are :code:`word`.

        Example:
            >>> corpus = Corpus(["just coming up with words", "more words"])
            >>> np.isclose(corpus.term_frequency("words", 1), 0.5)
            True
            >>> np.isclose(corpus.term_frequency("words", 0), 0.2)
            True

        Args:
            word: The word you're looking up
            document: The index of the document

        Raises:
            ValueError: If the document you're looking up doesn't exist
        """
        return self.index.term_frequency(word, document)

    def odds_document(self, word: str, document: int, sublinear: bool = False) -> float:
        """Returns the odds of finding a word in a document.

        This is the equivalent of :meth:`~text_data.index.WordIndex.odds_word`.
        But insteasd of calculating items at the word-corpus level, the calculations
        are performed at the word-document level.

        Example:
            >>> corpus = Corpus(["this is a document", "document two"])
            >>> corpus.odds_document("document", 1)
            1.0
            >>> corpus.odds_document("document", 1, sublinear=True)
            0.0

        Args:
            word: The word you're looking up
            document: The index of the document
            sublinear: If :code:`True`, returns the log-odds of finding the word in the document.

        Raises:
            ValueError: If the document doesn't exist.
        """
        return self.index.odds_document(word, document, sublinear)

    # Vector computations
    #
    # All of the functions in this grouping return term-vectors where each
    # item in the vector refers to a word in the vocabulary (returned by
    # `self.vocab_list`. Because this is internally stored
    # as a BTreeMap, these words appear in sorted order.)

    def doc_count_vector(self) -> np.array:
        """Returns the total number of documents each word appears in.

        Example:
            >>> corpus = Corpus(["example", "another example"])
            >>> corpus.doc_count_vector()
            array([1., 2.])
        """
        return self.index.doc_count_vector()

    def doc_freq_vector(self) -> np.array:
        """Returns the proportion of documents each word appears in.

        Example:
            >>> corpus = Corpus(["example", "another example"])
            >>> corpus.doc_freq_vector()
            array([0.5, 1. ])
        """
        return self.index.doc_freq_vector()

    def idf_vector(self) -> np.array:
        """Returns the inverse document frequency vector.

        Example:
            >>> corpus = Corpus(["example", "another example"])
            >>> corpus.idf_vector()
            array([2., 1.])
        """
        return self.index.idf_vector()

    def word_count_vector(self) -> np.array:
        """Returns the total number of times each word appeared in the corpus.

        Example:
            >>> corpus = Corpus(["example", "this example is another example"])
            >>> corpus.word_count_vector()
            array([1., 3., 1., 1.])
        """
        return self.index.word_count_vector()

    def word_freq_vector(self) -> np.array:
        """Returns the frequency in which each word appears over the corpus.

        Example:
            >>> corpus = Corpus(["example", "this example is another example"])
            >>> corpus.word_freq_vector()
            array([0.16666667, 0.5       , 0.16666667, 0.16666667])
        """
        return self.index.word_freq_vector()

    def odds_vector(self, sublinear: bool = False) -> np.array:
        """Returns a vector of the odds of each word appearing at random.

        Example:
            >>> corpus = Corpus(["example", "this example is another example"])
            >>> corpus.odds_vector()
            array([0.2, 1. , 0.2, 0.2])
            >>> corpus.odds_vector(sublinear=True)
            array([-2.32192809,  0.        , -2.32192809, -2.32192809])

        Args:
            sublinear: If true, returns the log odds.
        """
        return self.index.odds_vector(sublinear)

    # Matrix statistics
    # All of these functions create term-document matrix statistics

    def count_matrix(self) -> np.array:
        """Returns a matrix showing the number of times each word appeared in each document.

        Example:
            >>> corpus = Corpus(["example", "this example is another example"])
            >>> corpus.count_matrix().tolist() == [[0., 1.], [1., 2.], [0., 1.], [0., 1.]]
            True
        """
        return self.index.count_matrix()

    def frequency_matrix(self) -> np.array:
        """Returns a matrix showing the frequency of each word appearing in each document.

        Example:
            >>> corpus = Corpus(["example", "this example is another example"])
            >>> corpus.frequency_matrix().tolist() == [[0.0, 0.2], [1.0, 0.4], [0.0, 0.2], [0.0, 0.2]]
            True
        """
        # tf_matrix defaults to sublinear=True, normalize=False which are good settings
        # for computing TF-IDF. But with the opposite of those, we get the raw counts / the doc lengths
        return self.index.tf_matrix(sublinear=False, normalize=True)

    def one_hot_matrix(self) -> np.array:
        """Returns a matrix showing whether each given word appeared in each document.

        For these matrices, all cells contain a floating point value of either a
        1., if the word is in that document, or a 0. if the word is not in the document.

        These are sometimes referred to as 'one-hot encoding matrices' in machine learning.

        Example:
            >>> corpus = Corpus(["example", "this example is another example"])
            >>> np.array_equal(
            ...     corpus.one_hot_matrix(),
            ...     np.array([[0., 1.], [1., 1.], [0., 1.], [0., 1.]])
            ... )
            True
        """
        return self.index.one_hot_matrix()

    def odds_matrix(
        self, sublinear: bool = False, add_k: Optional[float] = None
    ) -> np.array:
        r"""Returns the odds of finding a word in a document for every possible word-document pair.

        Because not all words are likely to appear in all of the documents,
        this implementation adds :code:`1` to all of the numerators before taking the
        frequencies. So

        :math:`O(w) = \frac{c_{i} + 1}{N + \vert V \vert}`

        where :math:`\vert V \vert` is the total number of unique words in each document,
        :math:`N` is the total number of total words in each document, and :math:`c_i`
        is the count of a word in a document.

        Example:
            >>> corpus = Corpus(["example document", "another example"])
            >>> corpus.odds_matrix()
            array([[0.33333333, 1.        ],
                   [1.        , 0.33333333],
                   [1.        , 1.        ]])
            >>> corpus.odds_matrix(sublinear=True)
            array([[-1.5849625,  0.       ],
                   [ 0.       , -1.5849625],
                   [ 0.       ,  0.       ]])

        Args:
            sublinear: If :code:`True`, computes the log-odds.
            add_k: This adds :code:`k` to each of the non-zero elements in the matrix.
                Since :math:`\log{1} = 0`, this prevents 50 percent probabilities
                from appearing to be the same as elements that don't exist.
        """
        count_matrix = self.count_matrix()
        # find the number of unique words to add 1/|V| to each probability
        # to avoid divide by zero problems
        num_unique = np.count_nonzero(count_matrix, axis=0)
        num_words = count_matrix.sum(axis=0)
        nonzero_freq = (count_matrix + 1.0) / (num_unique + num_words)
        raw_odds = nonzero_freq / (1.0 - nonzero_freq)
        if sublinear:
            return np.log2(raw_odds)
        return raw_odds

    def tfidf_matrix(
        self,
        norm: Optional[str] = "l2",
        use_idf: bool = True,
        smooth_idf: bool = False,
        sublinear_tf: bool = True,
        add_k: int = 1,
    ) -> np.array:
        r"""This creates a term-document TF-IDF matrix from the index.

        In natural language processing, TF-IDF is a mechanism for finding
        out which words are distinct across documents. It's used particularly
        widely in information retrieval, where your goal is to rank documents
        that you know match a query by how relevant you think they'll be.

        The basic intuition goes like this: If a word appears particularly
        frequently in a document, it's probably more relevant to that document
        than if the word occurred more rarely. But, some words are simply
        common: If document X uses the word 'the' more often than the word
        'idiomatic,' that really tells you more about the words 'the' and
        'idiomatic' than it does about the document.

        TF-IDF tries to balance these two competing interests by taking the
        'term frequency,' or how often a word appears in the document,
        and normalizing it by the 'document frequency,' or the proportion
        of documents that contain the word. This has the effect of reducing
        the weights of common words (and even setting the weights of some
        very common words to 0 in some implementations).

        It should be noted that there are a number of different implementations
        of TF-IDF. Within information retrieval, TF-IDF is part of the `'SMART
        Information Retrieval System' <https://en.wikipedia.org/wiki/SMART_Information_Retrieval_System>`_.
        Although the exact equations can vary considerably, they typically follow the same approach:
        First, they find some value to represent the frequency of each word in the
        document. Often (but not always), this is just the raw number of times
        in which a word appeared in the document. Then, they normalize that
        based on the document frequency. And finally, they normalize
        those values based on the length of the document, so that long documents
        are not weighted more favorably (or less favorably) than shorter documents.

        The approach that I have taken to this is shamelessly cribbed from
        `scikit's TfidfTransformer <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html>`_.
        Specifically, I've allowed for some customization of the specific formula
        for TF-IDF while not including methods that require access to the raw
        documents, which would be computationally expensive to perform. This allows
        for the following options:

        * You can set the term frequency to either take the raw count of the word
          in the document (:math:`c_{t,d}`) or by using :code:`sublinear_tf=True`
          and taking :math:`1 + \log_{2}{c_{t,d}}`
        * You can skip taking the inverse document frequency :math:`df^{-1}`
          altogether by setting :code:`use_idf=False` or you can smooth the inverse
          document frequency by setting :code:`smooth_idf=True`.
          This adds one to the numerator and the denominator. (**Note:** Because
          this method is only run on a vocabulary of words that are in the corpus,
          there can't be any divide by zero errors, but this allows you to
          replicate scikit's :code:`TfidfTransformer`.)
        * You can add some number to the logged inverse document frequency
          by setting :code:`add_k` to something other than 1. This is the only
          difference between this implementation and scikits, as scikit automatically
          setts :code:`k` at 1.
        * Finally, you can choose how to normalize the document lengths. By default,
          this takes the L-2 norm, or :math:`\sqrt{\sum{w_{i,k}^{2}}}`, where :math:`w_{i,k}`
          is the weight you get from multiplying the term frequency by the inverse document
          frequency. But you can also set the norm to :code:`'l1'` to get the L1-norm,
          or :math:`\sum{\vert w_{i,k} \vert}`. Or you can set it to :code:`None` to avoid
          doing any document-length normalization at all.

        Examples:
            To get a sense of the different options, let's start by creating
            a pure count matrix with this method. To do that, we'll set
            :code:`norm=None` so we're not normalizing by the length of the document,
            :code:`use_idf=False` so we're not doing anything with the document
            frequency, and :code:`sublinear_tf=False` so we're not taking the
            logged counts:

            >>> corpus = Corpus(["a cat", "a"])
            >>> tfidf_count_matrix = corpus.tfidf_matrix(norm=None, use_idf=False, sublinear_tf=False)
            >>> assert np.array_equal(tfidf_count_matrix, corpus.count_matrix())

            In this particular case, setting :code:`sublinear_tf` to :code:`True`
            will produce the same result since all of the counts are 1 or 0
            and :math:`\log{1} + 1 = 1`:

            >>> assert np.array_equal(corpus.tfidf_matrix(norm=None, use_idf=False), tfidf_count_matrix)

            Now, we can incorporate the inverse document frequency. Because the word
            'a' appears in both documents, its inverse document frequency in is 1;
            the inverse document frequency of 'cat' is 2, since 'cat' appears in half
            of the documents. We're additionally taking the base-2 log of the inverse document
            frequency and adding 1 to the final result. So we get:

            >>> idf_add_1 = corpus.tfidf_matrix(norm=None, sublinear_tf=False, smooth_idf=False)
            >>> assert idf_add_1.tolist() == [[1., 1.], [2.,0.]]

            Or we can add nothing to the logged values:

            >>> idf = corpus.tfidf_matrix(norm=None, sublinear_tf=False, smooth_idf=False, add_k=0)
            >>> assert idf.tolist() == [[0.0, 0.0], [1.0, 0.0]]

            The L-1 norm normalizes the results by the sum of the absolute values of their
            weights. In the case of the count matrix, this is equivalent to creating
            the frequency matrix:

            >>> tfidf_freq_mat = corpus.tfidf_matrix(norm="l1", use_idf=False, sublinear_tf=False)
            >>> assert np.array_equal(tfidf_freq_mat, corpus.frequency_matrix())

        Args:
            norm: Set to 'l2' for the L2 norm (square root of the sums of the square weights),
                'l1' for the l1 norm (the summed absolute value, or None for no normalization).
            use_idf: If you set this to False, the weights will only include the term frequency
                (adjusted however you like)
            smooth_idf: Adds a constant to the numerator and the denominator.
            sublinear_tf: Computes the term frequency in log space.
            add_k: This adds k to every value in the IDF. scikit adds 1
                to all documents, but this allows for more variable computing
                (e.g. adding 0 if you want to remove words appearing in every document)
        """
        if norm not in {"l2", "l1", None}:
            raise ValueError("The norm you select must be an L1 norm, L2 norm, or None")
        # first compute the term frequency. Don't do any normalization yet.
        tf = self.index.tf_matrix(sublinear=sublinear_tf)
        # there are more elegant ways to do this, but this just ensures that
        # we don't compute the IDF if we don't need it
        if not use_idf:
            raw_idf = None
        elif smooth_idf:
            # number of docs + 1 over document counts + 1
            raw_idf = (len(self) + 1) / (self.index.doc_count_vector() + 1)
        else:
            raw_idf = self.index.idf_vector()
        if raw_idf is not None:
            # convert idf to log space and add k to all
            idf = np.log2(raw_idf) + add_k
            raw_tfidf = np.apply_along_axis(lambda x: x * idf, 0, tf)
        else:
            raw_tfidf = tf
        if norm is None:
            return raw_tfidf
        int_norm = 1 if norm == "l1" else 2
        return raw_tfidf / np.linalg.norm(raw_tfidf, ord=int_norm, axis=0)

    def get_top_words(
        self, term_matrix: np.array, top_n: Optional[int] = None, reverse: bool = True
    ) -> Tuple[np.array, np.array]:
        """Get the top values along a term matrix.

        Given a matrix where each row represents a word in your vocabulary,
        this returns a numpy matrix of those top values, along with an array
        of their respective words.

        You can choose the number of results you want to get by setting
        :code:`top_n` to some positive value, or you can leave it be and return
        all of the results in sorted order. Additionally, by setting
        :code:`reverse` to False (instead of its default of :code:`True`), you can
        return the scores from smallest to largest.

        Args:
            term_matrix: a matrix of floats where each row represents a word
            top_n: The number of values you want to return. If None, returns
                all values.
            reverse: If true (the default), returns the N values with the highest scores.
                If false, returns the N values with the lowest scores.

        Returns:
            A tuple of 2-dimensional numpy arrays, where the first item
            is an array of the top-scoring words and the second item
            is an array of the top scores themselves. Both arrays
            are of the same size, that is :code:`min(self.vocab_size, top_n)`
            by the number of columns in the term matrix.

        Raises:
            ValueError: If :code:`top_n` is less than 1, if there are
                not the same number of rows in the matrix as there are unique
                words in the index, or if the numpy array doesn't have 1 or 2 dimensions.

        Example:
            The first thing you need to do in order to use this function is create
            a 1- or 2-dimensional term matrix, where the number of rows
            corresponds to the number of unique words in the corpus. Any of the
            functions within :code:`WordIndex` that ends in :code:`_matrix(**kwargs)`
            (for 2-dimensional arrays) or :code:`_vector(**kwargs)` (for 1-dimensional
            arrays) will do the trick here. I'll show an example with both a
            word count vector and a word count matrix:

            >>> corpus = Corpus(["The cat is near the birds", "The birds are distressed"])
            >>> corpus.get_top_words(corpus.word_count_vector(), top_n=2)
            (array(['the', 'birds'], dtype='<U10'), array([3., 2.]))
            >>> corpus.get_top_words(corpus.count_matrix(), top_n=1)
            (array([['the', 'the']], dtype='<U10'), array([[2., 1.]]))

            Similarly, you can return the scores from lowest to highest by setting :code:`reverse=False`.
            (This is not the default.):

            >>> corpus.get_top_words(-1. * corpus.word_count_vector(), top_n=2, reverse=False)
            (array(['the', 'birds'], dtype='<U10'), array([-3., -2.]))
        """
        # https://github.com/numpy/numpy/issues/15128
        # this works independently of whether the array is 1-d or 2-d
        num_rows = np.shape(term_matrix)[0]
        if top_n is not None and top_n < 0:
            raise ValueError(
                "You must enter a positive number of results you wish to return"
            )
        if num_rows != self.vocab_size:
            raise ValueError(
                (
                    "You must pass a term matrix (a 1- or 2-dimensional array "
                    "where every unique word is a row) to this function"
                )
            )
        if term_matrix.ndim != 1 and term_matrix.ndim != 2:
            raise ValueError("You must enter a 1- or 2-dimensional array")
        # get the sorting order of the top
        sorted_vals = term_matrix.argsort(axis=0)
        k = top_n if top_n is not None else num_rows
        if reverse:
            row_slice, ordering = (slice(-k, None), slice(None, None, -1))
        else:
            row_slice, ordering = (slice(None, k), slice(None))
        # reshaping enables vectorized vocab searches; but can only do with 2-dimensional arrays
        vocab = np.array(self.vocab_list)
        if term_matrix.ndim == 2:
            vocab.resize((num_rows, 1))
        # for every row, column, gets the highest ranked matches
        top_scores = np.take_along_axis(term_matrix, sorted_vals, axis=0)[row_slice][
            ordering
        ]
        top_vocab = np.take_along_axis(vocab, sorted_vals, axis=0)[row_slice][ordering]
        return top_vocab, top_scores


class Corpus(WordIndex):
    r"""This is probably going to be your main entrypoint into :code:`text_data`.

    The corpus holds the raw text, the index, and the tokenized text
    of whatever you're trying to analyze. Its primary role is to extend
    the functionality of :class:`~text_data.index.WordIndex` to
    support searching. This means that you can use the :code:`Corpus`
    to search for arbitrarily long phrases using boolean search
    methods (AND, NOT, BUT).

    In addition, it allows you to add indexes so you can calculate
    statistics on phrases. By using :meth:`~text_data.index.Corpus.add_ngram_index`,
    you can figure out the frequency or TF-IDF values of multi-word
    phrases while still being able to search through your normal index.

    **Initializing Data**

    To instantiate the corpus, you need to include a list of documents
    where each document is a string of text and a tokenizer. There
    is a default tokenizer, which simply lowercases words and splits
    documents on :code:`r"\w+"`. For most tasks, this will be insufficient.
    But :py:mod:`text_data.tokenize` offers convenient ways that should
    make building the vast majority of tokenizers easy.

    The :code:`Corpus` can be instantiated using :code:`__init__` or by
    using :meth:`~text_data.index.Corpus.chunks`, which yields a generator,
    adding a mini-index. This allows you to technically perform
    calculations in-memory on larger databases.

    You can also initialize a :code:`Corpus` object by using the
    :meth:`~text_data.index.Corpus.slice`, :meth:`~text_data.index.Corpus.copy`,
    :meth:`~text_data.index.Corpus.split_off`, or :meth:`~text_data.index.Corpus.concatenate`
    methods. These methods work identically to their equivalent methods in
    :class:`text_data.index.WordIndex` while updating extra data that
    the corpus has, updating n-gram indexes, and automatically re-indexing the
    corpus.

    **Updating Data**

    There are two methods for updating or adding data to the :code:`Corpus`.
    :meth:`~text_data.index.Corpus.update` allows you to add new documents
    to the corpus. :meth:`~text_data.index.Corpus.add_ngram_index`
    allows you to add multi-word indexes.

    **Searching**

    There are a few methods devoted to searching. :meth:`~text_data.index.Corpus.search_documents`
    allows you to find all of the individual documents matching a query.
    :meth:`~text_data.index.Corpus.search_occurrences` shows all of the individual
    occurrences that matched your query. :meth:`~text_data.index.Corpus.ranked_search`
    finds all of the individual occurrences and sorts them according to a variant
    of their TF-IDF score.

    **Statistics**

    Three methods allow you to get statistics about a search.
    :meth:`~text_data.index.Corpus.search_document_count` allows you to find
    the total number of documents matching your query.
    :meth:`~text_data.index.Corpus.search_document_freq` shows the proportion
    of documents matching your query. And :meth:`~text_data.index.Corpus.search_occurrence_count`
    finds the total number of matches you have for your query.

    **Display**

    There are a number of functions designed to help you visually see the results of your
    query. :meth:`~text_data.index.Corpus.display_document` and
    :meth:`~text_data.index.Corpus.display_documents` render your documents in HTML.
    :meth:`~text_data.index.Corpus.display_document_count`,
    :meth:`~text_data.index.Corpus.display_document_frequency`,
    and :meth:`~text_data.index.Corpus.display_occurrence_count`
    all render bar charts showing the number of query results you got.
    And :meth:`~text_data.index.Corpus.display_search_results` shows
    the result of your search.

    Attributes:
        documents: A list of all the raw, non-tokenized documents in the corpus.
        tokenizer: A function that converts a list of strings (one of the documents
            from documents into a list of words and a list of the character-level
            positions where the words are located in the raw text). See :mod:`text_data.tokenize`
            for details.
        tokenized_documents: A list of the tokenized documents (each a list of words)
        ngram_indexes: A list of :class:`~text_data.index.WordIndex` objects
            for multi-word (n-gram) indexes. See :meth:`~text_data.index.Corpus.add_ngram_index`
            for details.
        ngram_sep: A separator in between words. See :meth:`~text_data.index.Corpus.add_ngram_index`
            for details.
        ngram_prefix: A prefix to go before any n-gram phrases. See :meth:`~text_data.index.Corpus.add_ngram_index`
            for details.
        ngram_suffix: A suffix to go after any n-gram phrases. See :meth:`~text_data.index.Corpus.add_ngram_index`
            for details.

    Args:
        documents: A list of the raw, un-tokenized texts.
        tokenizer: A function to tokenize the documents. See :mod:`text_data.tokenize` for details.
        sep: The separator you want to use for computing n-grams. See :meth:`~text_data.index.Corpus.add_ngram_index`
            for details.
        prefix: The prefix you want to use for n-grams. See :meth:`~text_data.index.Corpus.add_ngram_index`
            for details.
        suffix: The suffix you want to use for n-grams. See :meth:`~text_data.index.Corpus.add_ngram_index`
            for details.
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
    ):
        self.documents = documents
        self.tokenizer = tokenizer
        if len(documents) > 0:
            tokenized_docs = [tokenizer(doc) for doc in documents]
            # mypy doesn't get that this converts from a list of tuples to a tuple of lists
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
        """Iterates through documents, yielding a :code:`Corpus` with :code:`chunksize` documents.

        This is designed to allow you to technically use :code:`Corpus` on large
        document sets. However, you should note that searching for documents
        will only work within the context of the current chunk.

        The same is true for any frequency metrics. As such, you should probably
        limit metrics to raw counts or aggregations you've derived from raw counts.

        Example:
            >>> for docs in Corpus.chunks(["chunk one", "chunk two"], chunksize=1):
            ...     print(len(docs))
            1
            1

        Args:
            documents: A list of raw text items (not tokenized)
            tokenizer: A function to tokenize the documents
            sep: The separator you want to use for computing n-grams.
            prefix: The prefix for n-grams.
            suffix: The suffix for n-grams.
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
        yield cls(current_document_set, tokenizer, sep, prefix, suffix)

    def add_ngram_index(
        self,
        n: int = 1,
        default: bool = True,
        sep: Optional[str] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ):
        """Adds an n-gram index to the corpus.

        This creates a :class:`~text_data.index.WordIndex` object
        that you can access by typing :code:`self.ngram_indexes[n]`.

        There are times when you might want to compute
        TF-IDF scores, word frequency scores or similar scores
        over a multi-word index. For instance, you might want to know how
        frequently someone said 'United States' in a speech, without
        caring how often they used the word 'united' or 'states'.

        This function helps you do that. It automatically splits up your
        documents into an overlapping set of :code:`n`-length phrases.

        Internally, this takes each of your tokenized documents,
        merges them into lists of :code:`n`-length phrases, and joins
        each of those lists by a space. However, you can customize
        this behavior. If you set :code:`prefix`, each of the n-grams
        will be prefixed by that string; if you set :code:`suffix`,
        each of the n-grams will end with that string. And if you
        set :code:`sep`, each of the words in the n-gram will be separated
        by the separator.

        Example:
            Say you have a simple four word corpus. If you use the default
            settings, here's what your n-grams will look like:

            >>> corpus = Corpus(["text data is fun"])
            >>> corpus.add_ngram_index(n=2)
            >>> corpus.ngram_indexes[2].vocab_list
            ['data is', 'is fun', 'text data']

            By altering :code:`sep`, :code:`prefix`, or :code:`suffix`,
            you can alter that behavior. But, be careful to set :code:`default`
            to :code:`False` if you want to change the behavior
            from something you set up in :code:`__init__`. If you don't,
            this will use whatever settings you instantiated the class with.

            >>> corpus.add_ngram_index(n=2, sep="</w><w>", prefix="<w>", suffix="</w>", default=False)
            >>> corpus.ngram_indexes[2].vocab_list
            ['<w>data</w><w>is</w>', '<w>is</w><w>fun</w>', '<w>text</w><w>data</w>']

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

    def add_documents(
        self,
        tokenized_documents: List[List[str]],
        indexed_locations: Optional[List[Tuple[int, int]]] = None,
    ):
        """This overrides the :meth:`~text_data.index.WordIndex.add_documents` method.

        Because :meth:`~text_data.index.Corpus` objects can have
        n-gram indices, simply running :code:`add_documents` would
        cause n-gram indices to go out of sync with the overall
        corpus. In order to prevent that, this function raises
        an error if you try to run it.

        Raises:
            NotImplementedError: Warns you to use :code:`~text_data.index.Corpus.update`
            instead.
        """
        raise NotImplementedError(
            "You can not use `add_documents` with a `Corpus`. Use `update` instead."
        )

    def _copy_standard(self, other: Corpus):
        """This copies common information. It's internal to `copy`, `slice`, `split_off`, and `concatenate`."""
        other.tokenizer = self.tokenizer
        other.ngram_sep = self.ngram_sep
        other.ngram_prefix = self.ngram_prefix
        other.ngram_suffix = self.ngram_suffix

    def concatenate(self, other: Corpus) -> Corpus:  # type: ignore
        """This combines two :code:`Corpus` objects into one, much like :meth:`text_data.index.WordIndex.concatenate`.

        However, the new :code:`Corpus` has data from this corpus, including n-gram data.
        Because of this, the two :code:`Corpus` objects must have the same keys for their
        n-gram dictionaries.

        Example:
            >>> corpus_1 = Corpus(["i am an example"])
            >>> corpus_2 = Corpus(["i am too"])
            >>> corpus_1.add_ngram_index(n=2)
            >>> corpus_2.add_ngram_index(n=2)
            >>> combined_corpus = corpus_1.concatenate(corpus_2)
            >>> combined_corpus.most_common()
            [('am', 2), ('i', 2), ('an', 1), ('example', 1), ('too', 1)]
            >>> combined_corpus.ngram_indexes[2].most_common()
            [('i am', 2), ('am an', 1), ('am too', 1), ('an example', 1)]

        Args:
            other: another :code:`Corpus` object with the same ngram indexes.

        Raises:
            ValueError: if the n-gram indexes between the two corpuses are not the same.
        """
        new_corpus = Corpus([])
        new_corpus.index = self.index.concat(self.index, other.index, ignore_index=True)
        self._copy_standard(new_corpus)
        for n, index in self.ngram_indexes.items():
            new_corpus.ngram_indexes[n] = index.concatenate(other.ngram_indexes[n])
        new_corpus.documents = self.documents + other.documents
        new_corpus.tokenized_documents = (
            self.tokenized_documents + other.tokenized_documents
        )
        return new_corpus

    def copy(self) -> Corpus:
        """This creates a shallow copy of a :class:`Corpus` object.

        It extends the contents of :class:`Corpus` to also store data about
        the objects themselves.
        """
        new_index = Corpus([])
        new_index.index = new_index.copy()
        new_index.documents = self.documents.copy()
        new_index.tokenized_documents = self.tokenized_documents.copy()
        new_index.ngram_indexes = {
            n: index.copy() for n, index in self.ngram_indexes.items()
        }
        self._copy_standard(new_index)
        return new_index

    def slice(self, indexes: Set[int]) -> Corpus:
        """This creates a :code:`Corpus` object only including the documents listed.

        This overrides the method in :meth:`text_data.index.WordIndex`, which
        does the same thing (but without making changes to the underlying document set).
        This also creates slices of any of the n-gram indexes you have created.

        Note:
            This also changes the indexes for the new corpus so they all go from
            0 to :code:`len(indexes)`.

        Args:
            indexes: A set of document indexes you want to have in the new index.

        Example:
            >>> corpus = Corpus(["example document", "another example", "yet another"])
            >>> corpus.add_ngram_index(n=2)
            >>> sliced_corpus = corpus.slice({1})
            >>> len(sliced_corpus)
            1
            >>> sliced_corpus.most_common()
            [('another', 1), ('example', 1)]
            >>> sliced_corpus.ngram_indexes[2].most_common()
            [('another example', 1)]
        """
        new_index = Corpus([])
        self._copy_standard(new_index)
        new_index.index = self.index.slice(indexes)
        new_index.reset_index()
        for n, index in self.ngram_indexes.items():
            new_index.ngram_indexes[n] = index.slice(indexes)
            new_index.ngram_indexes[n].reset_index()
        documents, tokenized_docs = [], []
        sorted_indexes = sorted(indexes)
        for idx in sorted_indexes:
            documents.append(self.documents[idx])
            tokenized_docs.append(self.tokenized_documents[idx])
        new_index.documents = documents
        new_index.tokenized_documents = tokenized_docs
        return new_index

    def split_off(self, indexes: Set[int]) -> Corpus:
        """This operates like :meth:`~text_data.index.WordIndex.split_off`.

        But it additionally maintains the state of the :class:`Corpus` data,
        similar to how :meth:`~text_data.index.Corpus.slice` works.

        Example:
            >>> corpus = Corpus(["i am an example", "so am i"])
            >>> sliced_data = corpus.split_off({0})
            >>> corpus.documents
            ['so am i']
            >>> sliced_data.documents
            ['i am an example']
            >>> corpus.most_common()
            [('am', 1), ('i', 1), ('so', 1)]
        """
        new_index = Corpus([])
        self._copy_standard(new_index)
        new_index.index = super().split_off(indexes)
        new_index.reset_index()
        self.reset_index()
        for n, index in self.ngram_indexes.items():
            new_index.ngram_indexes[n] = index.split_off(indexes)
            new_index.ngram_indexes[n].reset_index()
            self.ngram_indexes[n].reset_index()
        own_docs, own_tokenized = [], []
        documents, tokenized_docs = [], []
        # we have to match to the length of the documents because len(self)
        # is tied to the length of the index, which is smaller now
        for idx in range(len(self.documents)):
            if idx in indexes:
                documents.append(self.documents[idx])
                tokenized_docs.append(self.tokenized_documents[idx])
            else:
                own_docs.append(self.documents[idx])
                own_tokenized.append(self.tokenized_documents[idx])
        new_index.documents = documents
        new_index.tokenized_documents = tokenized_docs
        self.documents = own_docs
        self.tokenized_documents = own_tokenized
        return new_index

    def update(self, new_documents: List[str]):
        """Adds new documents to the corpus's index and to the n-gram indices.

        Args:
            new_documents: A list of new documents. The tokenizer used is the same
                tokenizer used to initialize the corpus.
        """
        if len(new_documents) > 0:
            tokenized_docs = [self.tokenizer(doc) for doc in new_documents]
            # for some reason mypy doesn't get that this converts from a list of tuples to a tuple of lists
            words, positions = map(list, zip(*tokenized_docs))  # type: ignore
        else:
            words, positions = [], []
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
        # this tries to figure out the type of result we have
        # whether we're dealing with occurrence searches or document searches
        if all((isinstance(q, PositionResult) for q in subquery_results)):
            # need this check because set() passes the above check
            if len(subquery_results) > 0:
                return {q for q in query_results if q.doc_id in query_docs}  # type: ignore
        return query_docs  # type: ignore

    def search_documents(
        self,
        query: str,
        query_tokenizer: Callable[[str], List[str]] = str.split,
    ) -> Set[int]:
        """Search documents from a query.

        In order to figure out the intracacies of writing queries,
        you should view :py:mod:`text_data.query.Query`. In general,
        standard boolean (AND, OR, NOT) searches work perfectly reasonably.
        You should generally not need to set :code:`query_tokenizer` to anything
        other than the default (string split).

        This produces a set of unique documents, where each document is
        the index of the document. To view the documents by their ranked importance
        (ranked largely using TF-IDF), use :meth:`~text_data.index.Corpus.ranked_search`.

        Example:
            >>> corpus = Corpus(["this is an example", "here is another"])
            >>> assert corpus.search_documents("is") == {0, 1}
            >>> assert corpus.search_documents("example") == {0}

        Args:
            query: A string boolean query (as defined in :class:`text_data.query.Query`)
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
        query_tokenizer: Callable[[str], List[str]] = str.split,
    ) -> Set[PositionResult]:
        """Search for matching positions within a search.

        This allows you to figure out all of the occurrences
        matching your query. In addition, this is used internally
        to display search results.

        Each matching position comes in the form of a tuple where
        the first field :code:`doc_id` refers to the position
        of the document, the second field :code:`first_idx`
        refers to the starting index of the occurrence (among
        the tokenized documents), :code:`last_idx` refers to the
        last index of the occurrence, :code:`raw_start` refers
        to the starting index of the occurrence from
        *within the raw, non-tokenized documents.* :code:`raw_end`
        refers to the *index after the last character of the matching
        result* within the non-tokenized documents. There is not really
        a reason behind this decision.

        Example:
            >>> corpus = Corpus(["this is fun"])
            >>> result = list(corpus.search_occurrences("'this is'"))[0]
            >>> result
            PositionResult(doc_id=0, first_idx=0, last_idx=1, raw_start=0, raw_end=7)
            >>> corpus.documents[result.doc_id][result.raw_start:result.raw_end]
            'this is'
            >>> corpus.tokenized_documents[result.doc_id][result.first_idx:result.last_idx+1]
            ['this', 'is']

        Args:
            query: The string query. See :class:`text_data.query.Query` for details.
            query_tokenizer: The tokenizing function for the query.
                See :class:`text_data.query.Query` or :meth:`~text_data.index.Corpus.search_documents`
                for details.
        """
        return self._search_item(  # type: ignore
            self._yield_subquery_phrase_results,  # type: ignore
            Query(query, query_tokenizer),
        )

    def ranked_search(
        self,
        query_string: str,
        query_tokenizer: Callable[[str], List[str]] = str.split,
    ) -> List[List[PositionResult]]:
        """This produces a list of search responses in ranked order.

        More specifically, the documents are ranked in order of the
        sum of the TF-IDF scores for each word in the query
        (with the exception of words that are negated using a NOT operator).

        To compute the TF-IDF scores, I simply have computed the dot products
        between the raw query counts and the TF-IDF scores of all the unique
        words in the query. This is roughly equivalent to the :code:`ltn.lnn`
        normalization scheme
        `described in Manning <https://nlp.stanford.edu/IR-book/html/htmledition/document-and-query-weighting-schemes-1.html>`_.
        (The catch is that I have normalized the term-frequencies in the document
        to the length of the document.)

        Each item in the resulting list is a list referring to a single item.
        The items inside each of those lists are of the same format you get from
        :meth:`~text_data.index.Corpus.search_occurrences`. The first item in
        each list is either an item having the largest number of words in it
        or is the item that's the nearest to another match within the document.

        Args:
            query: Query string
            query_tokenizer: Function for tokenizing the results.

        Returns:
            A list of tuples, each in the same format as :meth:`~text_data.index.Corpus.search_occurrences`.
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
        query_tokenizer: Callable[[str], List[str]] = str.split,
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
        html, _ = self._show_html_occurrences(
            search_query, query_tokenizer, max_results, window_size
        )
        return display.HTML(html)

    def _show_html_occurrences(
        self,
        search_query: str,
        query_tokenizer: Callable[[str], List[str]] = str.split,
        max_results: Optional[int] = None,
        window_size: Optional[int] = None,
    ) -> Tuple[str, int]:
        """Plots all of the search matches as HTML phrases.

        Each result contains information about the order in which it appeared,
        the document ID, and bolds each matching phrase. The results are centered
        around the match that appears to be closest based on _sort_positions.

        Returns the HTML and the number of documents returned.

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
        search_results = self.ranked_search(search_query, query_tokenizer)
        for count, doc in enumerate(search_results):
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
                        results += core._escape_html(
                            sel_doc[current_idx : item.raw_start]
                        )
                    block_text = core._escape_html(
                        sel_doc[item.raw_start : item.raw_end]
                    )
                    results += f"<b>{block_text}</b>"
                    current_idx = item.raw_end
                raw_end_window = max(current_idx, end_window)
                results += core._escape_html(sel_doc[current_idx:raw_end_window])
                if raw_end_window < len(sel_doc):
                    results += "<b>&hellip;</b>"
                results += "</p>"
        return results, min(max_results, len(search_results))

    def search_document_count(
        self,
        query_string: str,
        query_tokenizer: Callable[[str], List[str]] = str.split,
    ) -> int:
        """Finds the total number of documents matching a query.

        By entering a search, you can get the total number of documents
        that match the query.

        Example:
            >>> corpus = Corpus(["the cow was hungry", "the cow likes the grass"])
            >>> corpus.search_document_count("cow")
            2
            >>> corpus.search_document_count("grass")
            1
            >>> corpus.search_document_count("the")
            2

        Args:
            query_string: The query you're searching for
            query_tokenizer: The tokenizer for the query
        """
        return len(self.search_documents(query_string, query_tokenizer))

    def search_document_freq(
        self,
        query_string: str,
        query_tokenizer: Callable[[str], List[str]] = str.split,
    ) -> float:
        """Finds the percentage of documents that match a query.

        Example:
            >>> corpus = Corpus(["the cow was hungry", "the cow likes the grass"])
            >>> corpus.search_document_freq("cow")
            1.0
            >>> corpus.search_document_freq("grass")
            0.5
            >>> corpus.search_document_freq("the grass")
            0.5
            >>> corpus.search_document_freq("the OR nonsense")
            1.0

        Args:
            query_string: The query you're searching for
            query_tokenizer: The tokenizer for the query
        """
        return self.search_document_count(query_string, query_tokenizer) / len(self)

    def search_occurrence_count(
        self,
        query_string: str,
        query_tokenizer: Callable[[str], List[str]] = str.split,
    ) -> int:
        """Finds the total number of occurrences you have for the given query.

        This just gets the number of items in :meth:`~text_data.Corpus.search_occurrences`.
        As a result, searching for occurrences where two separate words occur will find
        the total number of places where either word occurs within the set of documents
        where both words appear.

        Example:
            >>> corpus = Corpus(["the cow was hungry", "the cow likes the grass"])
            >>> corpus.search_occurrence_count("the")
            3
            >>> corpus.search_occurrence_count("the cow")
            5
            >>> corpus.search_occurrence_count("'the cow'")
            2

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
        query_tokenizer: Callable[[str], List[str]] = str.split,
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
        query_tokenizer: Callable[[str], List[str]] = str.split,
    ):
        """Returns a bar chart (in altair) showing the queries with the largest number of documents.

        Note:
            This method requires that you have :code:`altair` installed. To install,
            type :code:`pip install text_data[display]` or :code:`poetry add text_data -E display`.

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
        query_tokenizer: Callable[[str], List[str]] = str.split,
    ):
        """Displays a bar chart showing the percentages of documents with a given query.

        Note:
            This method requires that you have :code:`altair` installed. To install,
            type :code:`pip install text_data[display]` or :code:`poetry add text_data -E display`.

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
        query_tokenizer: Callable[[str], List[str]] = str.split,
    ):
        """Display a bar chart showing the number of times a query matches.

        Note:
            This method requires that you have :code:`altair` installed. To install,
            type :code:`pip install text_data[display]` or :code:`poetry add text_data -E display`.

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
        content = core._escape_html(self.documents[doc_idx])
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
