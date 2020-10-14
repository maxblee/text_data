"""Core utilities for text_data.

This is mostly where the Rust backend is stored.
"""
import functools
from typing import Any, Callable

from text_data_rs import PositionalIndex, ngrams_from_documents  # noqa: F401


def requires_display_extra(func: Callable[..., Any]) -> Callable[..., Any]:
    """This ensures that `altair` is installed in anything that depends on it."""
    try:
        # flake8 reads this as an unused import
        import altair as alt  # noqa: F401
    except ModuleNotFoundError:
        raise ImportError(
            """
    You must install `altair` to display this graphic. To do so, type
    `pip install text_data[display]` or `poetry install text_data -E display`
    or run `pip install altair` inside your environment.
    """
        )

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper
