"""Top-level package for Text Data."""
from typing import Callable, List

from text_data_rs import PositionalIndex

__author__ = """Max Lee"""
__email__ = "maxbmhlee@gmail.com"
__version__ = "0.1.0"

class Corpus:
    """This class holds the core data behind text_data.

    It holds the raw text, the index, and the tokenized text.
    Using it, you can compute statistics about the corpus,
    you can query the corpus, or you can visualize some findings.

    Args:
        documents: A list of raw text items (not tokenized)
        tokenizer: A function to tokenize the documents
        n: The number of n-grams (defaults to unigrams)
        keep_original_positions: If true, store the original positions
            within a document (this allows you to display the original text)
    """
    def __init__(
        self, 
        documents: List[str],
        tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
        n: int = 1,
        keep_original_positions: bool = True 
    ):
        self.documents = documents
        self.tokenized_documents = [tokenizer(doc) for doc in documents]
        self.index = PositionalIndex(self.tokenized_documents)