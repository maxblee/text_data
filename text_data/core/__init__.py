"""Core utilities for text_data.

This is mostly where the Rust backend is stored.
"""
import functools
import html
from typing import Any, Callable

from text_data_rs import PositionalIndex, ngrams_from_documents  # noqa: F401


def requires_display_extra(func: Callable[..., Any]) -> Callable[..., Any]:
    """This ensures that `altair` is installed in anything that depends on it."""
    try:
        # flake8 reads this as an unused import
        import altair as alt  # noqa: F401
        import pandas as pd  # noqa: F401
    except ModuleNotFoundError:
        raise ImportError(
            """
    You must install `altair` and `pandas` to display this graphic. To do so, type
    `pip install text_data[display]` or `poetry install text_data -E display`
    or run `pip install altair` inside your environment.
    """
        )

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper

def _escape_html(raw_text: str) -> str:
    """Escapes HTML to get rid of Jupyter rendering problems."""
    # the first part of this is built on the idea that if the content
    # contains html we shouldn't necessarily see that
    # the extra symbols are because of weird pretty printing behavior from
    # jupyter (see https://stackoverflow.com/questions/16089089/escaping-dollar-sign-in-ipython-notebook)
    return (
            html.escape(raw_text)
            # https://blogueun.wordpress.com/2014/01/04/escaping-in-mathjax/
            .replace("$", "<span class='tex2jax_ignore'>$</span>")
    )