"""Main module for Brownian motion."""

from collections.abc import Mapping
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, computed_field

from hamilflow.models.utils.typing import TypeTime


class SIRSystem(BaseModel):
    """Definition of the SIR system.

    :cvar beta: Transmission rate
    :cvar alpha: Recovery rate
    :cvar delta_t: Time granularity of the simulation
    """

    beta: float = Field(default=0.3, gt=0, description="Transmission rate", frozen=True)
    alpha: float = Field(default=0.1, gt=0, description="Recovery rate", frozen=True)
    delta_t: float = Field(ge=0.0, default=1.0)


class SIRIC(BaseModel):
    """The initial condition for an SIR model simulation.

    :cvar susceptible_0: Initial number of susceptible individuals
    :cvar infected_0: Initial number of infectious individuals
    :cvar recovered_0: Initial number of recovered individuals
    """

    susceptible_0: int = Field(
        default=999,
        ge=0,
        description="Initial susceptible population",
    )
    infected_0: int = Field(
        default=1,
        ge=0,
        description="Initial infectious population",
    )
    recovered_0: int = Field(
        default=0,
        ge=0,
        description="Initial recovered population",
    )

    @computed_field  # type: ignore[misc]
    @cached_property
    def n(self) -> int:
        """Total population in the simulation."""
        return self.susceptible_0 + self.infected_0 + self.recovered_0


class SIR:
    r"""SIR model simulation.

    The SIR model divides a population into three compartments:

    - $S$ (Susceptible): Individuals who can be infected.
    - $I$ (Infectious): Individuals who are currently infected and can transmit the disease.
    - $R$ (Recovered): Individuals who have recovered and are assumed to have immunity.
    - $N$(Total population): $N = S + I + R$.

    The dynamics of the compartments are governed by the following system of ordinary differential equations:

    $$
    \begin{split}
    \frac{dS(t)}{dt} &= -\beta I(t) S(t) \\
    \frac{dI(t)}{dt} &= \beta S(t) I(t) - \alpha I(t) \\
    \frac{dR(t)}{dt} &= \alpha I(t),
    \end{split}
    $$

    with the constraint

    $$
    N = S(t) + I(t) + R(t).
    $$

    Where:
    - $\beta$ is the transmission rate (probability of infection per contact per unit time).
    - $\alpha$ is the recovery rate (rate at which infected individuals recover per unit time).

    :param system: The parameters of the SIR system, including `beta` and `gamma`.
    :param initial_condition: The initial state of the population, including `S0`, `I0`, and `R0`.
    """

    def __init__(
        self,
        system: Mapping[str, float],
        initial_condition: Mapping[str, int],
    ) -> None:
        self.system = SIRSystem(**system)
        self.initial_condition = SIRIC(**initial_condition)

    @cached_property
    def definition(self) -> dict[str, dict[str, Any]]:
        """Model params and initial conditions defined as a dictionary."""
        return {
            "system": self.system.model_dump(),
            "initial_condition": self.initial_condition.model_dump(),
        }

    def generate_from(self, n_steps: int) -> pd.DataFrame:
        """Simulate the SIR model and return time series data.

        :param n_steps: Number of steps to simulate
        :return: DataFrame with time, S, I, R columns
        """
        time_steps = np.arange(0, n_steps) * self.system.delta_t

        return self(time_steps)

    def _step(
        self,
        susceptible: float,
        infected: float,
    ) -> tuple[int, int, int]:
        """Calculate changes in S, I, R populations for one time step.

        :param susceptible: Current susceptible population
        :param infected: Current infected population
        :param recovered: Current recovered population
        :return: tuple of (dS, dI, dR) changes
        """
        delta_s = -self.system.beta * susceptible * infected * self.system.delta_t
        delta_i = (
            self.system.beta * susceptible * infected - self.system.alpha * infected
        ) * self.system.delta_t
        delta_r = self.system.alpha * infected * self.system.delta_t

        return int(delta_s), int(delta_i), int(delta_r)

    def __call__(self, t: TypeTime) -> pd.DataFrame:
        """Generate the SIR model simulation based on the given time array."""
        susceptible = self.initial_condition.susceptible_0
        infected = self.initial_condition.infected_0
        recovered = self.initial_condition.recovered_0

        results = []

        for t_i in np.array(t):
            results.append(
                {
                    "t": t_i * self.system.delta_t,
                    "S": susceptible,
                    "I": infected,
                    "R": recovered,
                },
            )

            delta_s, delta_i, delta_r = self._step(susceptible, infected)

            susceptible += delta_s
            infected += delta_i
            recovered += delta_r

        return pd.DataFrame(results)
