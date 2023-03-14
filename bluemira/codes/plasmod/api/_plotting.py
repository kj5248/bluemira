# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.
"""
Plotting for PLASMOD.
"""

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.constants import MU_0
from bluemira.display import plot_defaults

__all__ = ["plot_default_profiles"]


def plot_default_profiles(plasmod_solver, show=True, f=None, ax=None):
    """
    Plot a default set of profiles from a PLASMOD solver.

    Parameters
    ----------
    plasmod_solver: plasmod.Solver
        Solver for which to plot profiles
    show: bool
        Whether or not to show the plot

    Returns
    -------
    f: Figure
        Matplotlib figure
    ax: np.ndarray[Axes]
        Array of matplotlib Axes
    """
    plot_defaults()
    if f is None and ax is None:
        f, ax = plt.subplots(2, 3, figsize=(18, 10))

    rho = plasmod_solver.get_profile("x")
    R_0 = plasmod_solver.params.R_0.value

    # Corrected flux function profiles (used as output profiles)
    pprime = plasmod_solver.get_profile("pprime")
    ffprime = plasmod_solver.get_profile("ffprime")

    # Actual flux function profiles

    pprime_true_but_wrong = plasmod_solver.plasmod_outputs().pprime
    pprime_true_but_wrong = plasmod_solver._from_phi_to_psi(pprime_true_but_wrong)
    ffprime_true_but_wrong = plasmod_solver.plasmod_outputs().ffprime
    ffprime_true_but_wrong = plasmod_solver._from_phi_to_psi(ffprime_true_but_wrong)
    # Current density profile reconstruction from flux functions
    jpar_recon = 2 * np.pi * (R_0 * pprime + ffprime / (MU_0 * R_0))
    jpar_recon_true_but_wrong = (
        2 * np.pi * (R_0 * pprime_true_but_wrong + ffprime_true_but_wrong / (MU_0 * R_0))
    )

    # Temperature profiles
    ti = plasmod_solver.get_profile("Ti")
    te = plasmod_solver.get_profile("Te")
    ax[0, 0].plot(rho, ti, label="$T_{i}$")
    ax[0, 0].plot(rho, te, label="$T_{e}$")
    ax[0, 0].set_ylabel("Temperature [keV]")

    # Current profiles
    jpar = plasmod_solver.get_profile("jpar")
    jbs = plasmod_solver.get_profile("jbs")
    jcd = plasmod_solver.get_profile("jcd")
    ax[0, 1].plot(rho, jpar, label="$j_{||}$")
    ax[0, 1].plot(rho, jbs, label="$j_{BS}$")
    ax[0, 1].plot(rho, jcd, label="$j_{CD}$")
    ax[0, 1].plot(rho, jpar_recon_true_but_wrong, linestyle="--", label="$j_{p', FF'}$")
    ax[0, 1].plot(rho, jpar_recon, linestyle="--", label="$j_{p', FF'_{*corr}}$")
    ax[0, 1].set_ylabel("Current density [A/m²]")

    # Density profiles
    ni = plasmod_solver.get_profile("n_ion")
    ne = plasmod_solver.get_profile("n_e")
    ax[1, 0].plot(rho, ni, label="$n_{i}$")
    ax[1, 0].plot(rho, ne, label="$n_{e}$")
    ax[1, 0].set_ylabel("Density [10¹⁹/m³]")

    # q profile
    qprof = plasmod_solver.get_profile("q")
    ax[1, 1].plot(rho, qprof, label="$q$")
    ax[1, 1].set_ylabel("Safety factor")

    # Flux functions
    ax[0, 2].plot(rho, pprime, label="$p'_{*corr}$")
    ax[0, 2].plot(rho, pprime_true_but_wrong, linestyle="--", label="$p'$")
    ax[0, 2].set_ylabel("[Pa/Wb]")
    axi: plt.Axes = ax[0, 2]
    axi.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

    ax[1, 2].plot(rho, ffprime, label="$FF'_{*corr}$")
    ax[1, 2].plot(rho, ffprime_true_but_wrong, linestyle="--", label="$FF'$")
    ax[1, 2].set_ylabel("[T]")

    for axe in ax.flat:
        axe.grid()
        axe.set_xlabel("$\\rho$")
        axe.set_xlim([0.0, 1.0])
        axe.legend(loc="best")

    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    if show:
        plt.show()
    return f, ax
