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

import pytest

from bluemira.base.builder import ComponentManager
from bluemira.base.reactor import Reactor, ReactorError
from bluemira.builders.plasma import Plasma, PlasmaBuilder
from bluemira.geometry.tools import make_polygon

REACTOR_NAME = "My Reactor"


class TFCoil(ComponentManager):
    """
    This component manager is purely for testing the reactor is still
    valid if this is not set, so we don't need to implement it.
    """


class MyReactor(Reactor):
    SOME_CONSTANT: str = "not a component"

    plasma: Plasma
    tf_coil: TFCoil


class TestReactor:
    @classmethod
    def setup_class(cls):
        cls.reactor = cls._make_reactor()

    def test_name_set_on_root_component(self):
        assert self.reactor.component().name == REACTOR_NAME

    def test_unset_components_are_not_part_of_component_tree(self):
        # Only the plasma component is in the tree, as we haven't set the
        # TF coil
        assert len(self.reactor.component().children) == 1

    def test_component_tree_built_from_class_properties(self):
        assert self.reactor.plasma.component().name == "Plasma"

    def test_show_cad_displays_all_components(self):
        self.reactor.show_cad()

    @pytest.mark.parametrize("bad_dim", ["not_a_dim", 1, ["x"]])
    def test_ReactorError_given_invalid_plotting_dimension(self, bad_dim):
        with pytest.raises(ReactorError):
            self.reactor.show_cad(dim=bad_dim)

    @staticmethod
    def _make_reactor() -> MyReactor:
        reactor = MyReactor(REACTOR_NAME)
        # use a square plasma, as circle causes topological naming issue
        lcfs = make_polygon({"x": [1, 1, 5, 5], "z": [-2, 2, 2, -2]}, closed=True)
        reactor.plasma = Plasma(PlasmaBuilder({}, lcfs).build())
        return reactor
