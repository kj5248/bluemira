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
Tests for the design module.
"""

from bluemira.base.design import Design

import tests


class TestDesign:
    def test_builders(self):
        build_config = {
            "Plasma": {
                "class": "MakeParameterisedShape",
                "param_class": "bluemira.equilibria.shapes::JohnerLCFS",
                "variables_map": {
                    "r_0": "R_0",
                    "a": "A",
                },
                "label": "Shape",
            },
            "TF Coils": {
                "class": "MakeParameterisedShape",
                "param_class": "PrincetonD",
                "variables_map": {
                    "x1": "r_tf_in_centre",
                    "x2": {
                        "value": "r_tf_out_centre",
                        "lower_bound": 8.0,
                    },
                    "dz": 0.0,
                },
                "label": "Shape",
            },
        }
        params = {
            "R_0": (9.0, "Input"),
            "A": (3.5, "Input"),
            "r_tf_in_centre": (5.0, "Input"),
            "r_tf_out_centre": (15.0, "Input"),
        }
        design = Design(params, build_config)
        component = design.run()

        assert component is not None

        assert [child.name for child in component.children] == ["Plasma", "TF Coils"]

        plasma_component = component.get_component("Plasma")
        assert [child.name for child in plasma_component.children] == ["Shape"]

        tf_coils_component = component.get_component("TF Coils")
        assert [child.name for child in tf_coils_component.children] == ["Shape"]

        if tests.PLOTTING:
            component.plot_2d()