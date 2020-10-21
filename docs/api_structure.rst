.. _api_structure:

How :code:`text_data` is organized
==================================

In the next few parts of the tutorial, I'm going to introduce
a bunch of the tools that :code:`text_data` has to offer
within the context of a mock project. My hope is that they'll
give you a sense of how you might introduce the library
into your analysis.

But before I do that, I want to introduce the structure of the
library.

:code:`text_data` is built upon two classes, :class:`~text_data.index.WordIndex`
and :class:`~text_data.index.Corpus`. The :code:`WordIndex`
class indexes your documents in an efficient data structure
and provides a number of ways for you to perform statistical lookups.
You can see how often words are used across your entire set of documents
or within a single option. And it supports matrix and vector operations
that allow you to see those same statistics at a much broader scale.
Because it uses an efficient data structure and parallelized
Rust code, those matrix and vector operations run rapidly.

In addition, :class:`~text_data.index.WordIndex` offers a number of ways
to split up or concatenate individual indexes you have
so you can compare how a portion of documents compares to another portion
of documents. This can be useful as you're conducting a standalone
analysis, or it can be useful if you're trying to debug
why a machine learning model is incorectly classifying
some of your documents.

:class:`~text_data.index.Corpus` builds upon the :class:`~text_data.index.WordIndex`
to support searching through documents. You can look up arbitrarily
long phrases and conduct boolean :code:`AND`, :code:`NOT`, and :code:`OR`
queries. In addition, :class:`~text_data.index.Corpus` offers an easy
way to index multi-word phrases.

For the most part, the other portions of the library work around these
two classes. :mod:`text_data.query` holds the internal support for
building queries. It's mainly meant as an internal data structure,
although there are cases when you might want to use it to
debug search results that don't seem to be working. :mod:`text_data.tokenize`
provides easy ways to write tokenizers (or functions that split up
a string into a list of words, or "tokens") that you can plug
directly into a :code:`Corpus`. And :mod:`text_data.multi_corpus`
offers two simple functions for building :code:`Corpus` or :code:`WordIndex`
objects from a list of other :code:`Corpus` or :code:`WordIndex` objects.

The only partial exception to this rule is the :mod:`text_data.display`
module, which offers features for displaying data visualizations
and top values along numpy matrixes and arrays. For the most part,
this, too, is designed to work with the :code:`WordIndex` and :code:`Corpus`.
But it's flexible and accepts other numpy matrixes and arrays as its inputs.