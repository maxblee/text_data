"""Top-level package for Text Data."""

__author__ = """Max Lee"""
__email__ = "maxbmhlee@gmail.com"
__version__ = "0.1.0"

from text_data.index import WordIndex, Corpus  # noqa: 401
from text_data import graphics, multi_corpus, tokenize, query

__all__ = ["graphics", "multi_corpus", "tokenize", "query", "WordIndex", "Corpus"]
