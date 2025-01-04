"""Tests for the Brownian motion main module."""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from hamilflow.models.brownian_motion import (
    BrownianMotion,
    BrownianMotionIC,
    BrownianMotionSystem,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy import typing as npt


@pytest.mark.parametrize(
    ("x0", "expected"),
    [(0.1, np.array(0.1)), (1, np.array(1.0)), ([1, 2], np.array([1, 2]))],
)
def test_brownian_motion_ic(
    x0: "float | int | Sequence[int]",
    expected: "npt.ArrayLike",
) -> None:
    """Test BrownianMotionIC."""
    brownian_motion_ic = BrownianMotionIC(x0=x0)

    np.testing.assert_equal(brownian_motion_ic.x0, expected)


@pytest.mark.parametrize(
    ("sigma", "delta_t", "gaussian_scale"),
    [(1, 1, 1), (1, 2, 2), (2, 1, 4)],
)
def test_brownian_motion_system(sigma: int, delta_t: int, gaussian_scale: int) -> None:
    """Test BrownianMotionSystem."""
    bms = BrownianMotionSystem(sigma=sigma, delta_t=delta_t)

    assert bms.gaussian_scale == gaussian_scale


@pytest.mark.parametrize(
    ("sigma", "delta_t"),
    [
        (-1, 1),
        (1, -1),
    ],
)
def test_brownian_motion_system_failed_spec(sigma: int, delta_t: int) -> None:
    """Test raises upon illegal parameters for initialising a BrownianMotionSystem."""
    m = r"\d+ validation error for BrownianMotionSystem\n{}\n".format(
        "sigma" if sigma < 0 else "delta_t",
    )
    with pytest.raises(ValueError, match=m):
        BrownianMotionSystem(sigma=sigma, delta_t=delta_t)


@pytest.mark.parametrize(
    ("sigma", "x0", "expected"),
    [
        (
            1,
            0,
            {
                "t": [0.0, 1.0, 2.0, 3.0, 4.0],
                "x_0": [0.0, 0.49671415, 0.35844985, 1.00613839, 2.52916825],
            },
        ),
        (
            1,
            [1, 1],
            {
                "t": [0.0, 1.0, 2.0, 3.0, 4.0],
                "x_0": [1.0, 1.49671415, 2.14440269, 1.91024932, 3.48946213],
                "x_1": [1.0, 0.8617357, 2.38476556, 2.1506286, 2.91806333],
            },
        ),
    ],
)
def test_brownian_motion_generate_from(
    sigma: int,
    x0: "int | Sequence[int]",
    expected: "Mapping[str, Sequence[float]]",
) -> None:
    """Test BrownianMotion values from generate_from, comparing with calculation results of the author."""
    system = {
        "sigma": sigma,
        "delta_t": 1,
    }

    initial_condition = {"x0": x0}

    bm = BrownianMotion(system=system, initial_condition=initial_condition)

    pd.testing.assert_frame_equal(
        bm.generate_from(n_steps=5, seed=42),
        pd.DataFrame(expected),
        check_exact=False,
        check_like=True,
    )


@pytest.mark.parametrize(
    ("sigma", "x0", "t", "expected"),
    [
        pytest.param(
            1,
            0,
            [0.0, 1.0, 2.0, 3.0, 4.0],
            {
                "t": [0.0, 1.0, 2.0, 3.0, 4.0],
                "x_0": [0.0, 0.49671415, 0.35844985, 1.00613839, 2.52916825],
            },
            id="1d-5-steps",
        ),
        pytest.param(
            1,
            [1, 1],
            [0.0, 1.0, 2.0, 3.0, 4.0],
            {
                "t": [0.0, 1.0, 2.0, 3.0, 4.0],
                "x_0": [1.0, 1.49671415, 2.14440269, 1.91024932, 3.48946213],
                "x_1": [1.0, 0.8617357, 2.38476556, 2.1506286, 2.91806333],
            },
            id="2d-5-steps",
        ),
        pytest.param(
            1,
            0,
            0.0,
            {
                "t": [0.0],
                "x_0": [0.0],
            },
            id="1d-scaler-t",
        ),
    ],
)
def test_brownian_motion(
    sigma: int,
    x0: "int | Sequence[int]",
    t: "float | Sequence[float]",
    expected: "Mapping[str, Sequence[float]]",
) -> None:
    """Test BrownianMotion values from __call__, comparing with calculation results of the author."""
    system = {
        "sigma": sigma,
        "delta_t": 1,
    }

    initial_condition = {"x0": x0}

    bm = BrownianMotion(system=system, initial_condition=initial_condition)

    pd.testing.assert_frame_equal(
        bm(t=t, seed=42),
        pd.DataFrame(expected),
        check_exact=False,
        check_like=True,
    )
