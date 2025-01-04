# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # SIR Model
#
# In this tutorial, we will learn how to use the SIR model.

# %%
import plotly.express as px

from hamilflow.models.sir import SIR

# %% [markdown]
# ## Model

# %%
sir_1 = SIR(
    system={
        "beta": 0.03,
        "alpha": 0.1,
        "delta_t": 0.1,
    },
    initial_condition={
        "susceptible_0": 999,
        "infected_0": 1,
        "recovered_0": 0,
    },
)

# %%
n_steps = 100

sir_1_results = sir_1.generate_from(n_steps=n_steps)
sir_1_results.head()

# %%
px.line(
    sir_1_results,
    x="t",
    y=["S", "I", "R"],
)

# %%
