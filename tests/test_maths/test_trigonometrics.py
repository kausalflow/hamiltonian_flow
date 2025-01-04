"""Tests for the trigonometrics module."""

from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from hamilflow.maths.trigonometrics import acos_with_shift

if TYPE_CHECKING:
    from collections.abc import Collection


class TestAcosWithShift:
    """Tests for arccos with shift."""

    _some_numbers: ClassVar[list[float]] = [x / 2 - 5 for x in range(20)]

    @pytest.fixture(params=[_some_numbers[0], _some_numbers])
    def phi(self, request: pytest.FixtureRequest) -> float | list[float]:
        """Give scalar phi and a list of phi's."""
        return request.param

    def test_acos_with_shift(self, phi: "float | Collection[float]") -> None:
        """Test arccos with shift."""
        phi = np.asarray(phi)
        actual = np.asarray(acos_with_shift(np.cos(phi), phi / 2 / np.pi))
        assert_array_almost_equal(phi, actual)
