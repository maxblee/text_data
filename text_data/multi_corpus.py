"""Tools and displays for handling multiple document sets.

These are primarily designed to provide features for merging
sets of documents so you can easy compute statistics on them.
"""
from text_data import WordIndex


def concatenate(*indexes: WordIndex, ignore_index: bool = True) -> WordIndex:
    """Concatenates an arbitrary number of :class:`text_data.index.WordIndex` objects.

    Args:
        ignore_index: If set to :code:`True`, which is the default, the resulting
            index has a reset index beginning at 0.

    Raises:
        ValueError: If :code:`ignore_index` is set to :code:`False` and there are overlapping
            document indexes.

    Example:
        >>> corpus_1 = WordIndex([["example"], ["document"]])
        >>> corpus_2 = WordIndex([["second"], ["document"]])
        >>> corpus_3 = WordIndex([["third"], ["document"]])
        >>> concatenate().most_common()
        []
        >>> concatenate(corpus_1).most_common()
        [('document', 1), ('example', 1)]
        >>> concatenate(corpus_1, corpus_2).most_common()
        [('document', 2), ('example', 1), ('second', 1)]
        >>> concatenate(corpus_1, corpus_2, corpus_3).most_common()
        [('document', 3), ('example', 1), ('second', 1), ('third', 1)]
    """
    if len(indexes) == 0:
        return WordIndex([])
    elif len(indexes) == 1:
        return indexes[0]
    else:
        first_idx, *rest = indexes
        for index in rest:
            first_idx = first_idx.concatenate(index)
        return first_idx


def flat_concat(*indexes: WordIndex) -> WordIndex:
    """This flattens a sequence of :class:`text_data.index.WordIndex` objects and concatenates them.

    This does not preserve any information about :class:`text_data.index.Corpus` objects.

    Example:
        >>> corpus_1 = WordIndex([["example"], ["document"]])
        >>> corpus_2 = WordIndex([["another"], ["set"], ["of"], ["documents"]])
        >>> len(corpus_1)
        2
        >>> len(corpus_2)
        4
        >>> len(concatenate(corpus_1, corpus_2))
        6
        >>> len(flat_concat(corpus_1, corpus_2))
        2

    Args:
        indexes: A sequence of :class:`text_data.index.Corpus` or :class:`text_data.index.WordIndex`
            objects.
    """
    return concatenate(*[index.flatten() for index in indexes])
