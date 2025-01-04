"""Tests for the SIR model."""

import numpy as np
import pandas as pd
import pytest

from hamilflow.models.sir import SIR


@pytest.fixture
def sir_system() -> dict[str, float]:
    """Params for the SIR system."""
    return {
        "beta": 0.3,
        "alpha": 0.1,
        "delta_t": 1.0,
    }


@pytest.fixture
def sir_initial_condition() -> dict[str, int]:
    """Generate initial conditions for the SIR model."""
    return {
        "susceptible_0": 999,
        "infected_0": 1,
        "recovered_0": 0,
    }


@pytest.fixture
def sir_model(sir_system: dict, sir_initial_condition: dict) -> SIR:
    """SIR model."""
    return SIR(sir_system, sir_initial_condition)


def test_sir_initialization(sir_model: SIR) -> None:
    """Test the initialization of the SIR model."""
    assert sir_model.system.beta == 0.3
    assert sir_model.system.alpha == 0.1
    assert sir_model.system.delta_t == 1.0
    assert sir_model.initial_condition.susceptible_0 == 999
    assert sir_model.initial_condition.infected_0 == 1
    assert sir_model.initial_condition.recovered_0 == 0


def test_sir_definition(sir_model: SIR) -> None:
    """Test the definition of the SIR model."""
    definition = sir_model.definition
    assert definition["system"]["beta"] == 0.3
    assert definition["system"]["alpha"] == 0.1
    assert definition["system"]["delta_t"] == 1.0
    assert definition["initial_condition"]["susceptible_0"] == 999
    assert definition["initial_condition"]["infected_0"] == 1
    assert definition["initial_condition"]["recovered_0"] == 0


def test_sir_generate_from(sir_model: SIR) -> None:
    """Test the data generation of the SIR model."""
    n_steps = 10
    result = sir_model.generate_from(n_steps)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == n_steps
    assert all(col in result.columns for col in ["t", "S", "I", "R"])


def test_sir_step(sir_model: SIR) -> None:
    """Test the step function of the SIR model."""
    delta_s, delta_i, delta_r = sir_model._step(999, 1)
    assert delta_s < 0
    assert delta_i > 0
    assert delta_r > 0


def test_sir_call(sir_model: SIR) -> None:
    """Test the call function of the SIR model."""
    t = np.arange(0, 10)
    result = sir_model(t)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(t)
    assert all(col in result.columns for col in ["t", "S", "I", "R"])
