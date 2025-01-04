"""Tests for the Kepler dynamics module."""

from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.testing import (
    assert_allclose,
    assert_approx_equal,
    assert_array_almost_equal,
)
from scipy.integrate import quad

from hamilflow.models.kepler_problem.dynamics import (
    tau_of_1_plus_u_hyperbolic,
    tau_of_e_minus_u_elliptic,
    tau_of_e_minus_u_hyperbolic,
    tau_of_e_plus_u_elliptic,
    tau_of_u_exact_elliptic,
    tau_of_u_exact_hyperbolic,
    tau_of_u_prime,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import typing as npt

_EPS_SQRT = 1e-16


@pytest.mark.usefixtures("ecc", "u_s", "tau_of_u")
class TestTauOfU:
    """Tests for calculating tau from u."""

    def test_const(
        self,
        ecc: float,
        tau_of_u: "Callable[[float, npt.ArrayLike], npt.ArrayLike]",
    ) -> None:
        """Test that tau at u = ecc should be 0.

        There are dividends sqrt(e**2 - u**2) and (u + 1),
        hence u cannot be too close to +e / -e / -1.
        """
        res = np.asarray(tau_of_u(ecc, ecc * (1 - _EPS_SQRT)))
        desired = np.full(res.shape, _EPS_SQRT)
        assert_array_almost_equal(desired, res)

    def test_tau_of_u(
        self,
        ecc: float,
        u_s: "npt.ArrayLike",
        tau_of_u: "Callable[[float, npt.ArrayLike], npt.ArrayLike]",
    ) -> None:
        """Test that tau from u is consistent with the integral from its first derivative.

        The integral constant has been tested in another test.
        """

        def integrand(u: "npt.ArrayLike") -> "npt.ArrayLike":
            return tau_of_u_prime(ecc, u)

        u_s = np.asarray(u_s).reshape(-1)
        rets = [quad(integrand, ecc, u) for u in u_s]
        integrals = np.array([ret[0] for ret in rets])
        assert_allclose(integrals, np.asarray(tau_of_u(ecc, u_s)))

    @pytest.fixture
    def exact_and_approx_tau_s(
        self,
        ecc: float,
    ) -> """tuple[Callable[[float, npt.NDArray[np.float64]], npt.NDArray[np.float64]], Callable[[float, npt.NDArray[np.float64]], npt.NDArray[np.float64]], Callable[[float, npt.NDArray[np.float64]], npt.NDArray[np.float64]]]""":
        """Give approximate and exact solutions for the elliptic and hyperbolic cases.

        The exact solutions have removable singularities at the boundary of domain,
        hence approximate solutions are needed.
        """
        if ecc in {0.0, 1.0}:
            c = "Parabolic" if ecc else "Circular"
            pytest.skip(f"{c} case is exact")
        elif 0 < ecc < 1:
            return (
                tau_of_u_exact_elliptic,
                tau_of_e_plus_u_elliptic,
                tau_of_e_minus_u_elliptic,
            )
        elif ecc > 1:
            return (
                tau_of_u_exact_hyperbolic,
                tau_of_1_plus_u_hyperbolic,
                tau_of_e_minus_u_hyperbolic,
            )
        else:
            msg = f"Expect ecc >= 0, got {ecc}"
            raise ValueError(msg)

    @pytest.mark.parametrize("epsilon", [3e-7])
    def test_expansion(
        self,
        ecc: float,
        epsilon: float,
        exact_and_approx_tau_s: "tuple[Callable[[float, npt.NDArray[np.float64]], npt.NDArray[np.float64]], Callable[[float, npt.NDArray[np.float64]], npt.NDArray[np.float64]], Callable[[float, npt.NDArray[np.float64]], npt.NDArray[np.float64]]]",
    ) -> None:
        """Test that the approximate solutions near the booundary of domain are close to the exact solution.

        The exact solutions have removable singularities at the boundary of domain,
        hence approximate solutions are needed.
        """
        if ecc in {0.0, 1.0}:
            pytest.skip(f"Test applies to ecc > 0, ecc != 1, got {ecc}")
        factor = 1 - epsilon
        f, g_s = exact_and_approx_tau_s[0], exact_and_approx_tau_s[1:]
        for u, g in zip((max(-1, -ecc) * factor, ecc * factor), g_s, strict=False):
            u_s = np.asarray(u)
            desired, actual = f(ecc, u_s), g(ecc, u_s)
            if not np.isinf(desired):
                assert_approx_equal(actual, desired, err_msg=f"ecc={ecc}, u={u_s}")
