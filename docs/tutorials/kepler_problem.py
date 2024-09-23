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
# # Kepler Orbits
#
# In this tutorial we will generate data for the Kepler problem.
#

# %% [markdown]
# ## Usage

# %%
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from hamilflow.models.kepler_problem import Kepler2D

# %% [markdown]
# To make it easy to use the Kepler system, we implemented an interface `Kepler2D.from_geometry` to specify the configuration using the geometry of the orbit. The geometry of the orbit requires three parameters:
#
# - `positive_angular_mom`: whether the angular momentum is positive
# - `ecc`: the eccentricity of the orbit
# - `parameter`: the semi-latus rectum of the orbits, for circular orbits, this is the radius

# %%
system = {
    "alpha": 1.0,
    "mass": 1.0,
}

k2d = Kepler2D.from_geometry(
    system=system,
    geometries={
        "positive_angular_mom": True,
        "ecc": 0,
        "parameter": 1.0,
    },
)

# %% [markdown]
# We define the time steps for a Kepler system. In this example, we create 401 time steps, also with some some negative timestamps for a more symmetric trajectory.

# %%
t = np.linspace(-20, 20, 401)

# %% [markdown]
# Orbit time series data is generated by calling the object `k2d`, as the data generation is done in the `__call__` method.

# %%
df_p_1 = k2d(t=t)
df_p_1  # noqa: B018


# %%
def visualize_orbit(dataframe: pd.DataFrame) -> go.Figure:
    """Plot out the trajectory in a polar coordinate.

    :param dataframe: dataframe containing the trajectory data
    """
    fig = px.line_polar(
        dataframe.assign(
            phi_degree=dataframe.phi / (2 * np.pi) * 360,
        ),
        r="r",
        theta="phi_degree",
        hover_data=["t", "r", "phi"],
        start_angle=0,
    )

    fig.update_layout(
        polar={
            "angularaxis_thetaunit": "radians",
        },
    )

    return fig


# %%
fig = visualize_orbit(df_p_1)

fig.show()

# %% [markdown]
# We also plot out the ellipse, hyperbolic, parabolic orbits.

# %%
visualize_orbit(
    Kepler2D.from_geometry(
        system=system,
        geometries={
            "positive_angular_mom": True,
            "ecc": 0.5,
            "parameter": 1.0,
        },
    )(t=t),
).show()

# %%
visualize_orbit(
    Kepler2D.from_geometry(
        system=system,
        geometries={
            "positive_angular_mom": True,
            "ecc": 1,
            "parameter": 1.0,
        },
    )(t=t),
).show()

# %%
visualize_orbit(
    Kepler2D.from_geometry(
        system=system,
        geometries={
            "positive_angular_mom": True,
            "ecc": 2,
            "parameter": 1.0,
        },
    )(t=np.linspace(-5, 5, 101)),
).show()

# %% [markdown]
# ## Formalism

# %% [markdown]
# In this section, we briefly discuss the derivations of motion in a central field. Please refer to *Mechanics: Vol 1* by Landau and Lifshitz for more details[^1].
#
# The Lagrangian for an object in a central field is
#
# $$
# \mathcal  L = \frac{1}{2} m ({\dot r}^2 + r^2 {\dot \phi}^2) - V(r),
# $$
#
# where $r$ and $\phi$ are the polar coordinates, $m$ is the mass of the object, and $V(r)$ is the potential energy. The equations of motion are
#
# $$
# \begin{align}
# \frac{\mathrm d r}{\mathrm d t}  &= \sqrt{ \frac{2}{m} (E - V(r)) - \frac{L^2}{m^2 r^2} } \\
# \frac{\mathrm d \phi}{\mathrm d t} &= \frac{L}{m r^2},
# \end{align}
# $$
#
# where $E$ is the total energy and $L$ is the angular momentum,
#
# $$
# \begin{align}
# E =& \frac{1}{2} m \left(\left(\frac{\mathrm d r}{ \mathrm dt} \right)^2 + r^2 \left( \frac{\mathrm d r}{\mathrm dt} \right)^2\right) + V(r) \\
# L =& m r^2 \frac{d\phi}{dt}
# \end{align}
# $$
#
# Both $E$ and $L$ are conserved. We obtain the coordinates as a function of time by solving the two equations.
#
#

# %% [markdown]
# For a inverse-square force, the potential energy is
#
# $$
# V(r) = - \frac{\alpha}{r},
# $$
#
# where $\alpha$ is a constant that specifies the strength of the force. For Newtonian gravity, $\alpha = G m_0$ with $G$ being the gravitational constant.
#
# First of all, we solve $\phi(r)$,
#
# $$
# \phi = \cos^{-1}\left( \frac{L/r - m\alpha/L}{2 m E + m \alpha^2/L^2} \right).
# $$
#
# Define
#
# $$
# p = \frac{L^2}{m\alpha}
# $$
#
# and
#
# $$
# e = \sqrt{ 1 + \frac{2 E L^2}{m \alpha^2}},
# $$
#
# we rewrite the solution $\phi(r)$ as
#
# $$
# r = \frac{p}{1 + e \cos{\phi}}.
# $$
#
# With the above relationship between $r$ and $\phi$, and $\frac{\mathrm d \phi}{\mathrm d} = \frac{L}{mr^2}$, we find that
#
# $$
# \frac{m\alpha^2}{L^3} \mathrm d t = \frac{1}{(1 + e \cos{\phi})^2} \mathrm d \phi.
# $$
#
# The integration on the right hand side depends on the domain of $e$.
#
# $$
# \int\frac{1}{(1 + e \cos{\phi})^2} \mathrm d \phi = \begin{cases}
# \frac{1}{(1 - e^2)^{3/2}} \left( 2 \tan^{-1} \sqrt{\frac{1 - e}{1 + e}} \tan\frac{\phi}{2} - \frac{e\sqrt{1 - e^2}\sin\phi}{1 + e\cos\phi} \right), & \text{if } e<1 \\
# \frac{1}{2}\tan{\frac{\phi}{2}} + \frac{1}{6}\tan^3{\frac{\phi}{2}}, & \text{if } e=1 \\
# \frac{1}{(e^2 - 1)^{3/2}}\left( \frac{e\sqrt{e^2-1}\sin\phi}{1 + e\cos\phi} - \ln \left( \frac{\sqrt{1 + e} + \sqrt{e-1}\tan\frac{\phi}{2}}{\sqrt{1 + e} - \sqrt{e-1}\tan\frac{\phi}{2} } \right) \right), & \text{if } e> 1.
# \end{cases}
# $$
#
# The value of $t(\phi)$ is easily obtained from the above formulae.

# %% [markdown]
# There exists many numerical methods to solve the Kepler orbits as functions of time, $r(t)$ and $\phi(t)$. For our use case of the solutions, we choose to integrate the equation of motion directly.

# %% [markdown]
# References:
#
# 1. Landau LD, Lifshitz EM. Mechanics: Vol 1. 3rd ed. Oxford, England: Butterworth-Heinemann; 1982.
# 2. Klioner SA. Basic Celestial Mechanics. arXiv [astro-ph.IM]. 2016. Available: http://arxiv.org/abs/1609.00915

# %% [markdown]
#
