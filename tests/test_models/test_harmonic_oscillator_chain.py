"""Tests for the harmonic oscillator chain main module."""

from collections.abc import Iterable, Mapping, Sequence
from itertools import chain, product

import numpy as np
import pytest

from hamilflow.models.harmonic_oscillator_chain import HarmonicOscillatorsChain

_wave_params: list[tuple[tuple[int, int], ...]] = [
    ((0, 0),),
    ((1, 0),),
    ((0, 1), (0, 1)),
]
_possible_wave_modes: list[dict[str, tuple[int, int]]] = [
    dict(zip(("amp", "phi"), param, strict=False)) for param in _wave_params
]


class TestHarmonicOscillatorChain:
    """Tests for the class harmonic oscillator chain."""

    @pytest.fixture(params=(1, 2))
    def omega(self, request: pytest.FixtureRequest) -> int:
        """Give omega for the system."""
        return request.param

    @pytest.fixture(params=((0, 0), (0, 1), (1, 0), (1, 1)))
    def free_mode(self, request: pytest.FixtureRequest) -> dict[str, int]:
        """Give initial conditions of the free mode of the system."""
        return dict(zip(("x0", "v0"), request.param, strict=False))

    @pytest.fixture(
        params=chain.from_iterable(
            product(_possible_wave_modes, repeat=r) for r in range(3)
        ),
    )
    def wave_modes(
        self,
        request: pytest.FixtureRequest,
    ) -> list[dict[str, tuple[int, int]]]:
        """Give initial conditions of the wave mode of the system.

        There can be 0, 1, 2 or 3 independent wave modes.
        """
        return request.param

    @pytest.fixture(params=(False, True))
    def odd_dof(self, request: pytest.FixtureRequest) -> bool:
        """Give whether the system has an odd number of DoF."""
        return request.param

    @pytest.fixture
    def legal_wave_modes_and_odd_def(
        self,
        wave_modes: Iterable[Mapping[str, tuple[int, int]]],
        odd_dof: bool,
    ) -> tuple[Iterable[Mapping[str, tuple[int, int]]], bool]:
        """Give legal wave-modes initial conditions, taking the odd_dof parameter into account."""
        return wave_modes if odd_dof else chain(wave_modes, [{"amp": (1, 1)}]), odd_dof

    @pytest.fixture(params=(0, 1, (0, 1)))
    def times(self, request: pytest.FixtureRequest) -> int | tuple[int]:
        """Give a scalar time or a tuple of times."""
        return request.param

    def test_init(
        self,
        omega: int,
        free_mode: Mapping[str, int],
        legal_wave_modes_and_odd_def: tuple[
            Iterable[Mapping[str, tuple[int, int]]],
            bool,
        ],
    ) -> None:
        """Test initialising a HarmonicOscillatorsChain."""
        wave_modes, odd_dof = legal_wave_modes_and_odd_def
        assert HarmonicOscillatorsChain(omega, [free_mode, *wave_modes], odd_dof)

    @pytest.fixture
    def hoc_and_zs(
        self,
        omega: int,
        free_mode: Mapping[str, int],
        legal_wave_modes_and_odd_def: tuple[
            Iterable[Mapping[str, tuple[int, int]]],
            bool,
        ],
        times: int | Sequence[int],
    ) -> tuple[HarmonicOscillatorsChain, np.ndarray, np.ndarray]:
        """Give initialised harmonic oscillators chain and the solutions from times."""
        wave_modes, odd_dof = legal_wave_modes_and_odd_def
        hoc = HarmonicOscillatorsChain(omega, [free_mode, *wave_modes], odd_dof)
        return (hoc, *hoc._z(times))

    def test_real(
        self,
        hoc_and_zs: tuple[HarmonicOscillatorsChain, np.ndarray, np.ndarray],
    ) -> None:
        """Test that the solution are real."""
        _, original_zs, _ = hoc_and_zs
        assert np.all(original_zs.imag == 0.0)

    def test_dof(
        self,
        hoc_and_zs: tuple[HarmonicOscillatorsChain, np.ndarray, np.ndarray],
    ) -> None:
        """Test the number of degrees of freedom is consistent."""
        hoc, original_zs, _ = hoc_and_zs
        assert original_zs.shape[0] == hoc.n_dof

    @pytest.mark.parametrize("wave_mode", [None, *_possible_wave_modes[1:]])
    def test_raise(
        self,
        omega: int,
        free_mode: Mapping[str, int],
        wave_mode: Mapping[str, int],
    ) -> None:
        """Test raises when odd_dof is false but the standing-wave mode is not real."""
        ics = [free_mode, *([wave_mode] if wave_mode else [])]
        m = r"For even degrees of freedom, "
        if wave_mode:
            m += "k == N // 2 must have equal positive and negative amplitudes."
        else:
            m += "at least 1 travelling wave is needed"
        with pytest.raises(ValueError, match=m):
            HarmonicOscillatorsChain(omega, ics, False)
