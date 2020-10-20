"""Renders data visualizations on :class:`text_data.index.WordIndex` objects.

The graphics in this module are designed to work across different
metrics. You just have to pass them 1- or 2-dimensional numpy arrays.

This enables you to take the outputs from any functions inside of
:class:`text_data.index.WordIndex` and visualize them.
"""
from typing import Any, List, Optional

import numpy as np

# this allows for global imports. ImportError will only be raised from inside
# a function decorated with `requires_display_extra`
try:
    import altair as alt
    import pandas as pd
except Exception:
    pass

from text_data import core, WordIndex


@core.requires_display_extra
def render_bar_chart(
    labels: np.array,
    vector_data: np.array,
    x_label: str = "Score",
    y_label: str = "Word",
):
    """Renders a bar chart given a 1-dimensional numpy array.

    Args:
        vector_data: A 1-dimensional numpy array of floating point
            scores.
        labels: A 1-dimensional numpy array of labels for the bar
            chart (e.g. words)
        x_label: The label for your x-axis (the score).
        y_label: The label for the y-axis (the words).

    Raises:
        ValueError: If the numpy arrays have more than 1 dimension.
    """
    if vector_data.ndim != 1 or labels.ndim != 1:
        raise ValueError("You must pass a numpy array into this function")
    word_score_map = pd.DataFrame({x_label: vector_data, y_label: labels})
    base_chart = (
        alt.Chart(word_score_map)
        .mark_bar()
        .encode(
            x=alt.X(f"{x_label}:Q", title=x_label),
            y=alt.Y(f"{y_label}:O", title=y_label, sort="-x"),
        )
    )
    return base_chart


@core.requires_display_extra
def render_multi_bar_chart(
    labels: np.array,
    matrix_scores: np.array,
    document_names: Optional[List[str]],
    y_label: str = "Score",
):
    """This renders a bar chart, grouped by document, showing word-document statistics.

    It's essentially the 2-dimensional matrix equivalent of :func:`~text_data.graphics.render_bar_chart`.

    Args:
        labels: A 2-dimensional numpy array of words, like those passed from
            :meth:`text_data.index.get_top_scores`.
        matrix_scores: A 2-dimensional numpy array of scores, like those
            passed from :meth:`text_data.index.get_top_scores`.
        document_names: A list of names for the documents. If :code:`None`,
            this will display numbers incrementing from 0.
        y_label: The name for the y label (where the scores go).

    Raises:
        ValueError: If your labels or your axes aren't 2 dimensional or
            aren't of the same size.
    """
    if labels.ndim != 2 or matrix_scores.ndim != 2:
        raise ValueError("You must pass labels and matrixes with 2 dimensions")
    if labels.shape != matrix_scores.shape:
        raise ValueError("Your labels and matrixes must have the same dimensions")
    if document_names is not None and len(document_names) != labels.shape[1]:
        raise ValueError(
            "Your list of document names must be equal to the number of documents"
        )
    base_frame = pd.DataFrame()
    # iterate through columns appending to dataframe
    for col in range(labels.shape[1]):
        doc_name = col if document_names is None else document_names[col]
        col_frame = pd.DataFrame(
            {
                "Word": labels[:, col],
                "Document": doc_name,
                y_label: matrix_scores[:, col],
            }
        )
        base_frame = base_frame.append(col_frame)
    base_chart = (
        alt.Chart(base_frame)
        .mark_bar()
        .encode(y=f"{y_label}:Q", x="Document:O", color="Document:O", column="Word:O")
    )
    return base_chart


@core.requires_display_extra
def heatmap(
    distance_matrix: np.array,
    left_indexes: Optional[List[Any]] = None,
    right_indexes: Optional[List[Any]] = None,
    left_name: str = "Left",
    right_name: str = "Right",
    metric_name: str = "Similarity",
):
    """Displays a heatmap displaying scores across a 2-dimensional matrix.

    The purpose of this is to visually gauge which documents
    are closest to each other given two sets of documents. (If you only have one
    set of documents, the left and right can be the same.) The visual rendering
    here is inspired by
    `tensorflow's Universal Sentence Encoder documentation <https://www.tensorflow.org/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder>`_.
    But, while you can use a universal sentence encoder to create the heatmap,
    you can also easily use any of the metrics in scikit's
    `pairwise_distances function <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html#sklearn.metrics.pairwise_distances>`_.
    Or, indeed, any other 2-dimensional matrix of floats will do the trick.

    Note that the :code:`left_name` and :code:`right_name` must be different. In order to account
    for this, this function automatically adds a suffix to both names if they are the same.

    Args:
        distance_matrix: A distance matrix of size M x N where M is the number of
            documents on the left side and N is the number of documents on the right side.
        left_indexes: Labels for the left side (the Y axis)
        right_indexes: Labels for the right side (the X axis)
        left_name: The Y axis label
        right_name: The X axis label

    Raises:
        ValueError: If the side of the indexes doesn't match the shape of the matrix
            of if there are not 2 dimensions in the distance matrix.
    """
    if distance_matrix.ndim != 2:
        raise ValueError("You must supply a 2-dimensional numpy array.")
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
        {left_name: row, right_name: column, metric_name: distance_matrix[i][j]}
        for i, row in enumerate(index)
        for j, column in enumerate(cols)
    ]
    return (
        alt.Chart(alt.Data(values=data))
        .mark_rect()
        .encode(
            y=f"{left_name}:O",
            x=f"{right_name}:O",
            color=f"{metric_name}:Q",
            tooltip=[f"{left_name}:O", f"{right_name}:O", f"{metric_name}:Q"],
        )
    )


@core.requires_display_extra
def frequency_map(
    index: WordIndex,
    word_vector: np.array,
    x_label: str = "Word Frequency",
    y_label: str = "Score",
):
    """A scatterplot scores over a corpus to their underlying frequencies.

    I cribbed this idea from `Monroe et al 2008 <http://languagelog.ldc.upenn.edu/myl/Monroe.pdf>`_,
    a great paper that uses it to show distributional problems in metrics that are trying to compare
    two things.

    The basic idea is that by creating a scatter plot mapping the frequencies of words
    to scores, you can both figure out which scores are disproportionately high
    or low *and* identify bias in whether your metric is excessively favoring common or rare words.

    In order to render this graphic, your word vector has to conform to the number of words
    in your index. If you feel the need to remove words to make the graphic
    manageable to look at, consider using :meth:`text_data.index.WordIndex.skip_words`.

    Args:
        index: A :class:`text_data.index.WordIndex` object. This is used to get the
            overall frequencies.
        word_vector: A 1-dimensional numpy array with floating point scores.
        x_label: The name of the x label for your graphic.
        y_label: The name of the y label for your graphic.

    Raises:
        ValueError: If the word_vector doesn't have 1 dimension or if the vector
            isn't the same length as your vocabulary.
    """
    if word_vector.shape != (index.vocab_size,):
        raise ValueError("You must enter a 1-dimensional array")
    df_rendering = pd.DataFrame(
        {
            "Word": index.vocab_list,
            x_label: index.word_freq_vector(),
            y_label: word_vector,
        }
    )
    return (
        alt.Chart(df_rendering)
        .mark_point()
        .encode(
            x=alt.X(f"{x_label}:Q", scale=alt.Scale(type="log", base=10)),
            y=f"{y_label}:Q",
            tooltip=list(df_rendering.columns),
        )
    )


def display_score_tables(
    words: np.array, scores: np.array, table_names: Optional[List[str]] = None
):
    """Renders two score tables.

    This is the 2-dimensional equivalent of
    :func:`~text_data.index.display_score_table` for details.

    Args:
        words: A 2-dimensional matrix of words
        scores: A 2-dimensional matrix of scores
        table_names: A list of names for your corresponding tables.

    Raises:
        ValueError: If :code:`words` and :code:`scores` aren't both 2-dimensional
            arrays of the same shape, or if :code:`table_names`
            isn't of the same length as the number of documents.
    """
    if words.ndim != 2 or scores.ndim != 2:
        raise ValueError("The word and score matrixes must be 2-dimensional.")
    if words.shape != scores.shape:
        raise ValueError("Both matrix arguments must be of the same shape.")
    _rows, cols = words.shape
    table_names = (
        list(map(str, range(len(cols)))) if table_names is None else table_names
    )
    if len(table_names) != cols:
        raise ValueError("There must be as many table names as there are columns")
    html = ""
    for doc_words, doc_scores, table_name in zip(words, scores, table_names):
        html += display_score_table(words, scores, table_name)
    return html


def display_score_table(
    words: np.array, scores: np.array, table_name: str = "Top Scores"
) -> str:
    """Returns the top (or bottom scores) as a table.

    It requires a 1-dimensional numpy array of the scores and the words,
    much as you would receive from
    :meth:`text_data.index.WordIndex.get_top_words`. For a 2-dimensional
    equivalent, use :meth:`~text_data.display.display_score_tables`.

    Args:
        words: A 1-dimensional numpy array of words.
        scores: A 1-dimensional numpy array of corresponding scores.
        table_name: The name to give your table.

    Raises:
        ValueError: If you did not use a 1-dimensional array, or if the
            two arrays don't have identical shapes.
    """
    if words.shape != scores.shape:
        raise ValueError("The shape of the words and scores must be the same")
    if words.ndim != 1 or scores.ndim != 1:
        raise ValueError("Both words and scores must have 1 dimension")
    html = (
        f"<p><b>{core._escape_html(table_name)}</b></p>"
        "<table>"
        "<thead>"
        "<tr><th>Order</th><th>Word</th><th>Score</th></tr>"
        "</thead>"
        "<tbody>"
    )
    for count, (word, score) in enumerate(zip(words, scores)):
        html += f"<tr><td>{count + 1}.</td><td>{core._escape_html(word)}</td><td>{score}</td></tr>"
    html += "</tbody></table>"
    return html
