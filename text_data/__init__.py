"""Top-level package for Text Data."""
from typing import Callable, List, Optional, Set

from text_data import indexing

__author__ = """Max Lee"""
__email__ = "maxbmhlee@gmail.com"
__version__ = "0.1.0"


class WordIndex:
    """This is a class designed to contain quick lookups of words and phrases.

    It mainly provides convenience lookups on top of `indexing.PositionalIndex`
    and is designed solely for use inside of `Corpus`.

    Args:
        tokenized_documents: A list of documents, where each document is a list of words.
    """

    def __init__(self, tokenized_documents: List[List[str]]):
        self.index = indexing.PositionalIndex(tokenized_documents)

    def __len__(self):
        return len(self.index)

    @property
    def vocab(self) -> Set[str]:
        """Returns all of the unique words in the index."""
        return self.index.vocabulary()


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
        tokenizer: Callable[[str], List[str]] = str.split,
        sep: Optional[str] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ):
        self.documents = documents
        self.tokenizer = tokenizer
        self.tokenized_documents = [tokenizer(doc) for doc in documents]
        self.ngram_indexes = {}
        self.ngram_sep = sep
        self.ngram_prefix = prefix
        self.ngram_suffix = suffix
        super().__init__(self.tokenized_documents)

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
            ngram_words = indexing.ngrams_from_documents(
                self.tokenized_documents,
                n,
                self.ngram_sep,
                self.ngram_prefix,
                self.ngram_suffix,
            )
        else:
            ngram_words = indexing.ngrams_from_documents(
                self.tokenized_documents, n, sep, prefix, suffix
            )
        self.ngram_indexes[n] = WordIndex(ngram_words)

    def update(self, new_documents: List[str]):
        """Updates the indexes for the corpus, given a new set of documents.

        This also updates the n-gram indexes.

        Args:
            new_documents: A list of new documents. The tokenizer used is the same
            tokenizer used to initialize the corpus.
        """
        tokenized_docs = [self.tokenizer(doc) for doc in new_documents]
        self.index.add_documents(tokenized_docs)
        for ngram, index in self.ngram_indexes.items():
            ngram_tok = indexing.ngrams_from_documents(
                tokenized_docs,
                ngram,
                self.ngram_sep,
                self.ngram_prefix,
                self.ngram_suffix,
            )
            index.add_documents(ngram_tok)
