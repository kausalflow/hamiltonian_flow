"""Tests for the Kepler numerics module."""

from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from hamilflow.models.kepler_problem.numerics import u_of_tau

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import typing as npt


class TestUOfTau:
    """Tests for calculating u from tau."""

    @pytest.mark.parametrize(
        "method",
        [
            "bisect",
            pytest.param(
                "newton",
                marks=pytest.mark.xfail(
                    reason="Newton method gives nan's, possibly because of inefficient initial estimate",
                ),
            ),
        ],
    )
    def test_u_of_tau(
        self,
        ecc: float,
        method: Literal["bisect", "newton"],
        u_s: "npt.ArrayLike",
        tau_of_u: "Callable[[float, npt.ArrayLike], npt.ArrayLike]",
    ) -> None:
        """Test numeric evaluation of u from tau."""
        u_s = np.asarray(u_s)
        tau = tau_of_u(ecc, u_s)
        actual = u_of_tau(ecc, tau, method)
        assert_array_almost_equal(u_s, actual)
