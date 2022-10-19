# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
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
Wall Silhouette Parameterisations
"""
import copy
from typing import Dict

from bluemira.geometry.parameterisations import PolySpline, PrincetonD


class WallPolySpline(PolySpline):
    """
    Defines the geometry for reactor first wall, without a divertor,
    based on the PolySpline parameterisation.
    """

    _defaults = {
        "x1": {"value": 5.8},
        "x2": {"value": 12.1},
        "z2": {"value": 0},
        "height": {"value": 9.3},
        "top": {"value": 0.4},
        "upper": {"value": 0.3},
        "dz": {"value": -0.5},
        "tilt": {"value": 0},
        "lower": {"value": 0.5},
        "bottom": {"value": 0.2},
    }

    def __init__(self, var_dict: Dict = None):
        if var_dict is None:
            var_dict = {}
        defaults = copy.deepcopy(self._defaults)
        defaults.update(var_dict)
        super().__init__(defaults)

        ib_radius = self.variables["x1"].value
        ob_radius = self.variables["x2"].value
        z2 = self.variables["z2"].value
        height = self.variables["height"].value
        top = self.variables["top"].value
        upper = self.variables["upper"].value
        dz = self.variables["dz"].value
        tilt = self.variables["tilt"].value
        lower = self.variables["lower"].value
        bottom = self.variables["bottom"].value

        if not self.variables["x1"].fixed:
            self.adjust_variable(
                "x1",
                ib_radius,
                lower_bound=ib_radius - 2,
                upper_bound=ib_radius * 1.1,
            )
        if not self.variables["x2"].fixed:
            self.adjust_variable(
                "x2",
                value=ob_radius,
                lower_bound=ob_radius * 0.9,
                upper_bound=ob_radius + 2,
            )
        self.adjust_variable("z2", z2, lower_bound=-0.9, upper_bound=0.9)
        self.adjust_variable(
            "height", height, lower_bound=height - 0.001, upper_bound=50
        )
        self.adjust_variable("top", top, lower_bound=0.05, upper_bound=0.75)
        self.adjust_variable("upper", upper, lower_bound=0.2, upper_bound=0.7)
        self.adjust_variable("dz", dz, lower_bound=-5, upper_bound=5)
        self.adjust_variable("tilt", tilt, lower_bound=-25, upper_bound=25)
        self.adjust_variable("lower", lower, lower_bound=0.2, upper_bound=0.7)
        self.adjust_variable("bottom", bottom, lower_bound=0.05, upper_bound=0.75)

        # Fix 'flat' to avoid drawing the PolySpline's outer straight.
        # The straight is often optimised to near-zero length, which
        # causes an error when CAD tries to draw it
        self.fix_variable("flat", 0)


class WallPrincetonD(PrincetonD):
    """
    Defines the geometry for reactor first wall, without a divertor,
    based on the PrincetonD parameterisation.
    """

    _defaults = {
        "x1": {"value": 5.8},
        "x2": {"value": 12.1},
        "dz": {"value": -0.5},
    }

    def __init__(self, var_dict: Dict = None):
        if var_dict is None:
            var_dict = {}
        defaults = copy.deepcopy(self._defaults)
        defaults.update(var_dict)
        super().__init__(defaults)

        ib_radius = self.variables["x1"].value
        ob_radius = self.variables["x2"].value
        if not self.variables["x1"].fixed:
            self.adjust_variable(
                "x1", ib_radius, lower_bound=ib_radius - 2, upper_bound=ib_radius * 1.02
            )

        if not self.variables["x2"].fixed:
            self.adjust_variable(
                "x2", ob_radius, lower_bound=ob_radius * 0.98, upper_bound=ob_radius + 2
            )
        self.adjust_variable(
            "dz", self.variables["dz"].value, lower_bound=-3, upper_bound=3
        )