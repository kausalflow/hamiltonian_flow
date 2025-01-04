"""Tests for the pendulum main module."""

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy import typing as npt
from numpy.testing import assert_array_almost_equal

from hamilflow.models.pendulum import Pendulum, PendulumIC, PendulumSystem

if TYPE_CHECKING:
    from collections.abc import Sequence


@pytest.fixture(params=[0.3, 0.6, 1.5])
def omega0(request: pytest.FixtureRequest) -> float:
    """Give a few omega0's."""
    return request.param


@pytest.fixture(params=[-1.5, -0.3, 0.2, 0.4, +1.4])
def theta0(request: pytest.FixtureRequest) -> float:
    """Give a few theta0's."""
    return request.param


@pytest.fixture(params=(-1.2, [-5.5, 0.0, 0.5, 1.0], np.linspace(0, 1, 7)))
def times(request: pytest.FixtureRequest) -> float:
    """Give a scalar time, a list of and a numpy array of times."""
    return request.param


class TestPendulumSystem:
    """Tests for the class PendulumSystem."""

    @pytest.mark.parametrize("omega0", [-1.0, 0])
    def test_omega0_range(self, omega0: float) -> None:
        """Test raises upon illegal omega0's."""
        m = r"\d+ validation error for PendulumSystem\nomega0\n"
        with pytest.raises(
            ValueError,
            match=m,
        ):
            _ = PendulumSystem(omega0=omega0)


class TestPendulumIC:
    """Tests for the class PendulumIC."""

    @pytest.mark.parametrize("theta0", [-2.0, 2.0])
    def test_theta0_range(self, theta0: float) -> None:
        """Test raises upon illegal theta0's."""
        m = r"\d+ validation error for PendulumIC\ntheta0\n"
        with pytest.raises(
            ValueError,
            match=m,
        ):
            _ = PendulumIC(theta0=theta0)

    def test_k(self, theta0: float) -> None:
        """Test the kinematic transformation from theta0 to k."""
        ic = PendulumIC(theta0=theta0)
        assert ic.k == np.sin(ic.theta0 / 2)


class TestPendulum:
    """Tests for the class Pendulum."""

    @pytest.mark.parametrize(
        ("omega0", "theta0", "period"),
        [(2 * math.pi, math.pi / 3, 1.0731820071493643751)],
    )
    def test_period_static(self, omega0: float, theta0: float, period: float) -> None:
        """Test the calculated period from python is the same as the value from Mathematica."""
        p = Pendulum(omega0, theta0)
        assert pytest.approx(p.period) == period

    def test_transf(
        self,
        omega0: float,
        theta0: float,
        times: "Sequence[float] | npt.ArrayLike",
    ) -> None:
        """Test the dynamic transformation from u to theta."""
        p = Pendulum(omega0, theta0)
        arr_times = np.asarray(times)

        sin_u = np.sin(p.u(arr_times))
        theta_terms = np.sin(p.theta(arr_times) / 2) / p._k
        assert_array_almost_equal(theta_terms, sin_u)

    def test_period_dynamic_theta(
        self,
        omega0: float,
        theta0: float,
        times: "Sequence[float] | npt.ArrayLike",
    ) -> None:
        """Test theta of t has the theory-predicted period."""
        p = Pendulum(omega0, theta0)
        arr_times_1 = np.asarray(times) + p.period

        theta, theta_1 = p.theta(times), p.theta(arr_times_1)

        assert_array_almost_equal(theta, theta_1)

    def test_generate_from(self, omega0: float, theta0: float) -> None:
        """Test alternative UI with periods and sample numbers."""
        p = Pendulum(omega0, theta0)

        df = p.generate_from(n_periods=1, n_samples_per_period=10)
        assert len(df) == 10  # noqa: PLR2004
