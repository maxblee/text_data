"""Tests `text_data.core`."""
from unittest import mock

import pytest

from text_data.core import requires_display_extra


def test_require_decorator():
    """Makes sure that the decorator that requires altair works."""

    def fake_function(x, **kwargs):
        return sum(kwargs.values()) + x

    with mock.patch.dict("sys.modules", {"altair": None}):
        with pytest.raises(ImportError):
            requires_display_extra(fake_function)(1, y=2, z=3)

    try:
        fake_function(1, y=2, z=3)
    except ImportError:
        pytest.fail("This function should not raise an error")
