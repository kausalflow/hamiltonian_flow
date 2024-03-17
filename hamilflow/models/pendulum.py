import math
from functools import cached_property
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from pydantic import BaseModel, Field, computed_field
from scipy.special import ellipj, ellipk


class PendulumSystem(BaseModel):
    r"""The params for the pendulum

    :param float omega0: $\omega_0 \coloneqq \sqrt{\frac{U}{I}} > 0$, frequency
    parameter
    """

    omega0: float = Field(gt=0)

    def __init__(self, omega0: float, **kwargs: Any) -> None:
        return super().__init__(omega0=omega0, **kwargs)


class PendulumIC(BaseModel):
    r"""The initial condition for a pendulum

    :param float theta0: $-\frac{\pi}{2} \le \theta_0 \le \frac{\pi}{2}$, the
    initial angular displacement
    """

    theta0: float = Field(ge=-math.pi / 2, le=math.pi / 2)

    def __init__(self, theta0: float, **kwargs: Any) -> None:
        return super().__init__(theta0=theta0, **kwargs)

    @computed_field  # type: ignore[misc]
    @cached_property
    def k(self) -> float:
        r"""A convenient number

        :return float: $\sin\frac{\theta_0}{2}$
        """
        return math.sin(self.theta0 / 2)


class Pendulum:
    r"""Generate time series data for a pendulum.

    # Lagrangian action
    The Lagrangian action for a pendulum is
    $$
    S\[\theta(t)\] = \int_{0}^{t_0} \mathbb{d}t
    \left\\{\frac{1}{2} I \dot\theta^2 + U \cos\theta \right\\} \eqqcolon
    \int_{0}^{t_0} \mathbb{d}t\,L_\text{P}(\theta, \dot\theta)\,,
    $$
    where $\theta$ is the angular displacement from the vertical to the
    pendulum; $I$ is an _inertial parameter_, $U$ is a potential parameter;
    $L_\text{P}$ is the Lagrangian.

    This setup contains both the single and the physical pendula. For a single
    pendulum,
    $$
    I = m l^2\,,\qquad U = mgl\,,
    $$
    where $m$ is the mass of the pendulum, $l$ is the length of the rod or cord,
    and $g$ is the gravitational acceleration.
    """

    # # Integral of motion
    # $\mathbb{\delta}S / \mathbb{\delta}{t} = 0$
    # $$
    # \dot\theta\frac{\partial L_\text{P}}{\partial \dot\theta} - L_\text{P}
    # \equiv E = U \cos\theta_0
    # $$

    # $$
    # \left(\frac{\mathbb{d}t}{\mathbb{d}\theta}\right)^2 = \frac{1}{2\omega_0^2}
    # \frac{1}{\cos\theta - \cos\theta_0}
    # $$
    # $\omega_0 \coloneqq \sqrt{\frac{U}{I}}$

    # ## Coordinate transformation
    # $$
    # \sin u \coloneqq \frac{\sin\frac{\theta}{2}}{\sin\frac{\theta_0}{2}}
    # $$

    # $$
    # \left(\frac{\mathbb{d}t}{\mathbb{d}u}\right)^2 = \frac{1}{\omega_0^2}
    # \frac{1}{1-k^2\sin^2 u}
    # $$

    def __init__(
        self,
        system: Union[int, float, Dict[str, Union[int, float]]],
        initial_condition: Union[int, float, Dict[str, Union[int, float]]],
    ) -> None:
        if isinstance(system, (float, int)):
            system = {"omega0": system}
        if isinstance(initial_condition, (float, int)):
            initial_condition = {"theta0": initial_condition}
        self.system = PendulumSystem.model_validate(system)
        self.initial_condition = PendulumIC.model_validate(initial_condition)

    @cached_property
    def definition(self) -> Dict[str, float]:
        """Model params and initial conditions defined as a dictionary."""
        return dict(
            system=self.system.model_dump(),
            initial_condition=self.initial_condition.model_dump(),
        )

    @property
    def omega0(self) -> float:
        return self.system.omega0

    @property
    def _k(self) -> float:
        return self.initial_condition.k

    @property
    def _math_m(self) -> float:
        return self._k**2

    @computed_field  # type: ignore[misc]
    @cached_property
    def freq(self) -> float:
        r"""Frequency.

        :return float: $\frac{\pi}{2K(k^2)}\omega_0$, where
        $K(m)$ is [Legendre's complete elliptic integral of the first kind](https://dlmf.nist.gov/19.2#E8)
        """
        return math.pi * self.omega0 / (2 * ellipk(self._math_m))

    @computed_field  # type: ignore[misc]
    @cached_property
    def period(self) -> float:
        r"""Period.

        :return float: $\frac{4K(k^2)}{\omega_0}$, where
        $K(m)$ is [Legendre's complete elliptic integral of the first kind](https://dlmf.nist.gov/19.2#E8)
        """
        return 4 * ellipk(self._math_m) / self.omega0

    def _math_u(self, t: ArrayLike) -> np.ndarray[float]:
        return self.omega0 * np.asarray(t)

    # defined by $\sin u \coloneqq \frac{\sin\frac{\theta}{2}}{\sin\frac{\theta_0}{2}}$
    def u(self, t: ArrayLike) -> np.ndarray:
        r"""The convenient generalised coordinate $u$.

        :param ArrayLike t: time
        :return np.ndarray: $u(t) = \operatorname{am}{\big(\omega_0 t + K(k^2), k^2\big)}$, where
        $\operatorname{am}{x, k}$ is [Jacobi's amplitude function](https://dlmf.nist.gov/22.16#E1),
        $K(m)$ is [Legendre's complete elliptic integral of the first kind](https://dlmf.nist.gov/19.2#E8)
        """
        _, _, _, ph = ellipj(self._math_u(t) + ellipk(self._math_m), self._math_m)

        return ph

    def theta(self, t: ArrayLike) -> np.ndarray:
        r"""Angular coordinate $\theta$.

        :param ArrayLike t: time
        :return np.ndarray: $\theta(t) = 2\arcsin\big(k\cdot\operatorname{cd}{(\omega_0 t, k^2)}\big)$, where
        $\operatorname{cd}{z, k}$ is a [Jacobian elliptic function](https://dlmf.nist.gov/22.2#E8)
        """
        _, cn, dn, _ = ellipj(self._math_u(t), self._math_m)

        return 2 * np.arcsin(cn / dn * self._k)

    def __call__(self, n_periods: int, n_samples_per_period: int) -> pd.DataFrame:
        time_delta = self.period / n_samples_per_period
        time_steps = np.arange(0, n_periods * n_samples_per_period) * time_delta
        thetas = self.theta(time_steps)

        return pd.DataFrame({"t": time_steps, "x": thetas})
