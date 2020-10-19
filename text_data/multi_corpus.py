"""Tools and displays for handling multiple document sets.

Some of these metrics, like the heatmap, can be performed
against a single corpus, with the corpus's documents being
compared to themselves.

The purpose of this module is to allow you to explore similarities
and differences between corpuses or within a single corpus. Like
the rest of this library, this does not do too much work for you;
it's designed as a lightweight set of tools that can be generalized
regardless of the similarity or difference metric you intend to use.
But it does have some simple metrics that are generally useful
across contexts.
"""
import bisect
import functools
import math
from typing import Any, Callable, List, Optional, Tuple

from IPython import display
import numpy as np

from text_data import Corpus, WordIndex
from text_data.core import requires_display_extra


def concatenate(*indexes: WordIndex, ignore_index: bool = True) -> WordIndex:
    """This method concatenates an arbitrary number of :class:`text_data.index.WordIndex` objects.

    Args:
        ignore_index: If set to :code:`True`, which is the default, the resulting
            index has a reset index beginning at 0.

    Raises:
        ValueError: If :code:`ignore_index` is set to :code:`False` and there are overlapping
            document indexes.

    Example:
        >>> corpus_1 = Corpus(["example", "document"])
        >>> corpus_2 = Corpus(["second", "document"])
        >>> corpus_3 = Corpus(["third", "document"])
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


@requires_display_extra
def distance_heatmap(
    distance_matrix: np.array,
    left_indexes: Optional[List[Any]] = None,
    right_indexes: Optional[List[Any]] = None,
    left_name: str = "Left",
    right_name: str = "Right",
):
    """Displays a heatmap mapping the similarities or distances given a distance matrix.

    The purpose of this is to visually gauge which documents
    are closest to each other given two sets of documents. (If you only have one
    set of documents, the left and right can be the same.) The visual rendering
    here is inspired by
    [`tensorflow`'s Universal Sentence Encoder documentation](https://www.tensorflow.org/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder).
    But, while you can use a universal sentence encoder to create the heatmap,
    you can also easily use any of the metrics in scikit's [`pairwise_distances` function](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html#sklearn.metrics.pairwise_distances).
    Or, indeed, any other 2-dimensional matrix of floats will do the trick.

    Note that the `left_name` and `right_name` must be different. In order to account
    for this, this function automatically adds a suffix to both names if they are the same.

    Args:
        distance_matrix: A distance matrix of size M x N where M is the number of
            documents on the left side and N is the number of documents on the right side.
        left_indexes: Labels for the left side (the Y axis)
        right_indexes: Labels for the right side (the X axis)
        left_name: The Y axis label
        right_name: The X axis label

    Raises:
        A `ValueError`, if the side of the indexes doesn't match the shape of the matrix.
    """
    import altair as alt

    rows, cols = np.shape(distance_matrix)
    # raise an error if the size of the indexes don't match
    if left_indexes is not None and len(left_indexes) != rows:
        raise ValueError(
            (
                "The length of the indexes must match the number of rows in the distance matrix."
                f"{rows} does not match {len(left_indexes)}"
            )
        )
    if right_indexes is not None and len(right_indexes) != cols:
        raise ValueError(
            (
                "The length of the columns must match the number of columns in the distance matrix."
                f"{cols} does not match {len(right_indexes)}"
            )
        )
    # because the data is in a dictionary, the left and right names must be different
    if left_name == right_name:
        left_name = f"{left_name}_0"
        right_name = f"{right_name}_1"
    # set the labels for the heatmap
    index = left_indexes if left_indexes is not None else range(rows)
    cols = right_indexes if right_indexes is not None else range(cols)
    data = [
        {left_name: row, right_name: column, "Similarity": distance_matrix[i][j]}
        for i, row in enumerate(index)
        for j, column in enumerate(cols)
    ]
    return (
        alt.Chart(alt.Data(values=data))
        .mark_rect()
        .encode(
            y=f"{left_name}:O",
            x=f"{right_name}:O",
            color="Similarity:Q",
            tooltip=[f"{left_name}:O", f"{right_name}:O", "Similarity:Q"],
        )
    )


def display_multiple_searches(
    corpuses: List[Corpus],
    search_query: str,
    query_tokenizer: Callable[[str], List[str]] = str.split,
    max_results: Optional[int] = None,
    window_size: Optional[int] = None,
    total_max: Optional[int] = None,
) -> display.HTML:
    """Takes a list of corpuses and prints out search results.

    This essentially is an extension of `Corpus.show_html_occurrences`
    for multiple corpuses.

    Args:
        corpuses: A list of `Corpus` objects you want to display results from
        search_query: The query you're searching for
        query_tokenizer: The tokenizer for the query
        max_results: The maximum number of results you want to display over a single corpus.
            If None, returns all results.
        window_size: The number of characters you want to return around the matching phrase.
            If None, returns the entire document.
        total_max: The maximum number of results you want to display over *all* corpuses.
            If None, returns all results.
    """
    num_results = 0
    html = f"<p><b>Showing results for {len(corpuses)} corpuses</b></p>"
    html += "<style>" ".newtab { padding-left: 4ch;}" "</style>" "<div class='newtab'>"
    for count, corpus in enumerate(corpuses):
        html += f"<p><b>Corpus {count}</b></p><div class='newtab'>"
        if max_results is not None and total_max is not None:
            new_results: Optional[int] = min(max_results, total_max - num_results)
        else:
            # for some reason, mypy can't even figure out a simple if/else variable declaration
            new_results: Optional[int] = None  # type: ignore
        cur_html, cur_count = corpus._show_html_occurrences(
            search_query, query_tokenizer, new_results, window_size
        )
        html += f"{cur_html}</div>"
        if num_results is not None:
            num_results += cur_count
            if total_max is not None and num_results >= total_max:
                break
    html += "</div>"
    return display.HTML(html)


def display_documents(
    corpuses: List[Corpus], documents: List[List[int]]
) -> display.HTML:
    """This displays document results over multiple corpuses.

    Args:
        corpuses: A list of `Corpus` objects
        documents: A list of the exact size as `corpuses`
            where each item in the list is a list of indexes. The
            indexes in that item refer to the document indexes of
            the corpus that is at the same position in the corpus list.

    Raises:
        ValueError, if `corpuses` and `documents` are not of the same length
    """
    if len(corpuses) != len(documents):
        raise ValueError(
            (
                "The number of corpuses must be equal to the number of documents.\n"
                f"There are {len(corpuses)} corpuses and {len(documents)} documents."
            )
        )
    html = "<style>" ".newtab { padding-left: 4ch;}" "</style>"
    for count, corpus in enumerate(corpuses):
        html += f"<p><b>Showing corpus {count + 1} Results</b></p>"
        html += (
            "<div class='newtab'>"
            f"{corpus.display_documents(documents[count]).data}"
            "</div>"
        )
    return display.HTML(html)


def get_word_differences(
    left_corpus: Corpus,
    right_corpus: Corpus,
    scoring_func: Callable[[Corpus, Corpus, Optional[Corpus], str], Optional[float]],
    background_corpus: Optional[Corpus] = None,
) -> List[Tuple[float, str]]:
    """Returns a list of word scores, sorted by the word score.

    In cases where you're trying to compare two sets of documents,
    this returns singular scores tied to the combined set of words.

    Args:
        left_corpus: One of the corpuses you're comparing
        right_corpus: The other corpus you're comparing
        scoring_func: A function that takes the two corpuses and a word
            and optionally returns a float. If the result is None, the score
            will not be added to the final list of sorted scores and words.
        background_corpus: A background corpus (for e.g. calculating WordScores)
    """
    scores: List[Tuple[float, str]] = []
    combined_vocabulary = left_corpus.vocab.union(right_corpus.vocab)
    for word in combined_vocabulary:
        word_score = scoring_func(left_corpus, right_corpus, background_corpus, word)
        if word_score is not None:
            bisect.insort(scores, (word_score, word))
    return scores


def risk_ratio(
    left_corpus: Corpus,
    right_corpus: Corpus,
    background_corpus: Optional[Corpus],
    word: str,
) -> Optional[float]:
    """Returns the risk ratio over a list of corpuses.

    This calculates the frequency of a word in one corpus over the frequency
    of the word in the other corpus.
    """
    # adjust frequencies slightly so there are no negative frequencies
    adjusted_left = (left_corpus.word_count(word) + 1) / left_corpus.vocab_size
    adjusted_right = (right_corpus.word_count(word) + 1) / right_corpus.vocab_size
    return adjusted_left / adjusted_right


def odds_ratio(
    left_corpus: Corpus,
    right_corpus: Corpus,
    background_corpus: Optional[Corpus],
    word: str,
) -> Optional[float]:
    """Returns the odds ratio."""
    adjusted_left = (left_corpus.word_count(word) + 1) / (
        left_corpus.num_words + left_corpus.vocab_size
    )
    adjusted_right = (right_corpus.word_count(word) + 1) / (
        right_corpus.num_words + right_corpus.vocab_size
    )
    odds_left = adjusted_left / (1 - adjusted_left)
    odds_right = adjusted_right / (1 - adjusted_right)
    return odds_left / odds_right


def log_odds_ratio(
    left_corpus: Corpus,
    right_corpus: Corpus,
    background_corpus: Optional[Corpus],
    word: str,
) -> Optional[float]:
    """Returns the log-odds ratio."""
    adjusted_left = (left_corpus.word_count(word) + 1) / (
        left_corpus.num_words + left_corpus.vocab_size
    )
    adjusted_right = (right_corpus.word_count(word) + 1) / (
        right_corpus.num_words + right_corpus.vocab_size
    )
    odds_left = adjusted_left / (1 - adjusted_left)
    odds_right = adjusted_right / (1 - adjusted_right)
    return math.log(odds_left) - math.log(odds_right)


def get_word_scores(
    corpuses: List[Corpus],
    scoring_func: Callable[
        [Corpus, List[Corpus], Optional[Corpus], str], Optional[float]
    ],
    background_corpus: Optional[Corpus] = None,
) -> List[List[Tuple[float, str]]]:
    """Returns a list of scores for the total set of words for each corpus in the list.

    For instance, you can use this function to get the TF-IDF scores for each
    corpus in the list of corpuses. The format of this list is a sorted list of
    scores and words for each corpus in the list of corpuses.

    Args:
        corpuses: A list of `Corpus` objects
        scoring_func: A function that takes a corpus, a list of comparison corpuses, and a word
            and optionally returns a float. If the result is None, the score
            will not be added to the final list of sorted scores and words.
        background_corpus: A background corpus (for e.g. calculating WordScores).
    """
    score_list = []
    vocab_sets = [corpus.vocab for corpus in corpuses]
    combined_vocab = functools.reduce(lambda a, b: a.union(b), vocab_sets)
    for i, corpus in enumerate(corpuses):
        corpus_scores: List[Tuple[float, str]] = []
        # get all of the corpuses except for the current one
        remaining_corpuses = corpuses[:i] + corpuses[i + 1 :]
        for word in combined_vocab:
            word_score = scoring_func(
                corpus, remaining_corpuses, background_corpus, word
            )
            if word_score is not None:
                bisect.insort(corpus_scores, (word_score, word))
        score_list.append(corpus_scores)
    return score_list


def tfidf(
    corpus: Corpus,
    other_corpuses: List[Corpus],
    background_corpus: Optional[Corpus],
    word: str,
) -> Optional[float]:
    """Returns the TF-IDF score for the given word, compared to the rest of the scores."""
    # the word is guaranteed to be in the corpus we're looking at
    doc_count = sum([word in c for c in other_corpuses + [corpus]])
    idf = math.log((len(other_corpuses) + 1) / doc_count)
    tf = math.log(1 + corpus.word_frequency(word))
    return tf * idf


@requires_display_extra
def display_difference_graph(
    left_corpus: Corpus,
    right_corpus: Corpus,
    scoring_func: Callable[[Corpus, Corpus, Optional[Corpus], str], Optional[float]],
    background_corpus: Optional[Corpus] = None,
    num_results: Optional[int] = None,
):
    """Displays a scatter plot plotting the overall frequency of a word to its score.

    This is inspired from [Monroe 2008](http://languagelog.ldc.upenn.edu/myl/Monroe.pdf)
    and is useful for exploring the words that most distinguish one corpus from another.

    Args:
        left_corpus: One of the corpuses you're comparing
        right_corpus: The other corpus you're comparing
        scoring_func: A function that takes the two corpuses and a word
            and optionally returns a float. If the result is None, the score
            will not be added to the final list of sorted scores and words.
        background_corpus: A background corpus (for e.g. calculating WordScores)
        num_results: The number of results you want to return. If you set at None, returns all results.
    """
    import altair as alt

    data = []
    word_diffs = get_word_differences(
        left_corpus, right_corpus, scoring_func, background_corpus
    )
    for score, word in word_diffs:
        total_words = left_corpus.num_words + right_corpus.num_words
        word_count = left_corpus.word_count(word) + right_corpus.word_count(word)
        background_freq = word_count / total_words
        data.append({"Score": score, "Frequency": background_freq, "Word": word})
    if num_results is not None:
        top_results = data[: math.ceil(num_results / 2) + 1]
        bottom_results = data[-math.floor(num_results / 2) :]
        data = top_results + bottom_results
    return (
        alt.Chart(alt.Data(values=data))
        .mark_point()
        .encode(
            x=alt.X("Frequency:Q", scale=alt.Scale(type="log", base=10)),
            y="Score:Q",
            tooltip=["Word:N", "Frequency:Q", "Score:Q"],
        )
    )


def display_notable(
    scores: List[List[Tuple[float, str]]],
    class_names: List[str],
    num_results: int = 10,
    reverse: bool = True,
) -> display.HTML:
    """Displays tables of each class's most notable words (words with the highest scores).

    Args:
        scores: A list of lists of (score, word) tuples, where each item in the list refers to
            a class (or corpus)
        class_names: A list of class names. Must be the same length as the scores.
        num_results: The number of words you want to show.
        reverse: Whether you want the results to go descending (highest score to lowest) or ascending.
            True is descending, False is ascending

    Raises:
        A ValueError if the `scores` and `class_names` are not of the same length.
    """
    html = ""
    if len(scores) != len(class_names):
        raise ValueError("`scores` and `class_names` must have the same length.")
    for i, score_set in enumerate(scores):
        if reverse:
            score_set = score_set[::-1]
        score_table = _display_score_table(score_set[:num_results])
        html += f"<p><b>{class_names[i]}</b></p>{score_table}"
    return display.HTML(html)


def display_top_differences(
    left_corpus: Corpus,
    right_corpus: Corpus,
    scoring_func: Callable[[Corpus, Corpus, Optional[Corpus], str], Optional[float]],
    background_corpus: Optional[Corpus] = None,
    num_results: int = 10,
) -> display.HTML:
    """Displays the words that most distinguish the left corpus from the right corpus.

    Args:
        left_corpus: One of the corpuses you're comparing
        right_corpus: The other corpus you're comparing
        scoring_func: A function that takes the two corpuses and a word
            and optionally returns a float. If the result is None, the score
            will not be added to the final list of sorted scores and words.
        background_corpus: A background corpus (for e.g. calculating WordScores)
        num_results: The number of results you want to return from each corpus.
    """
    word_diffs = get_word_differences(
        left_corpus, right_corpus, scoring_func, background_corpus
    )
    top_scores = word_diffs[:num_results]
    bottom_results = word_diffs[-num_results:]
    return display_notable(
        [top_scores, bottom_results], ["Low Scores", "High Scores"], num_results
    )


def _display_score_table(scores: List[Tuple[float, str]]) -> str:
    """Returns the top (or bottom scores) as a table."""
    html = (
        "<table>"
        "<thead>"
        "<tr><th>Order</th><th>Word</th><th>Score</th></tr>"
        "</thead>"
        "<tbody>"
    )
    html += "".join(
        [
            f"<tr><td>{count + 1}.</td><td>{word}</td><td>{score}</td></tr>"
            for count, (score, word) in enumerate(scores)
        ]
    )
    html += "</tbody></table>"
    return html
