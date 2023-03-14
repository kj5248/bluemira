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
Wrapper for FreeCAD Part.Face objects
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

import bluemira.codes._freecadapi as cadapi

# import from bluemira
from bluemira.geometry.base import BluemiraGeo
from bluemira.geometry.coordinates import Coordinates

# import from error
from bluemira.geometry.error import DisjointedFace, NotClosedWire
from bluemira.geometry.wire import BluemiraWire

__all__ = ["BluemiraFace"]


class BluemiraFace(BluemiraGeo):
    """Bluemira Face class."""

    def __init__(self, boundary, label: str = ""):
        boundary_classes = [BluemiraWire]
        super().__init__(boundary, label, boundary_classes)

    @staticmethod
    def _converter(func):
        def wrapper(*args, **kwargs):
            output = func(*args, **kwargs)
            if isinstance(output, cadapi.apiWire):
                output = BluemiraWire(output)
            if isinstance(output, cadapi.apiFace):
                output = BluemiraFace._create(output)
            return output

        return wrapper

    def copy(self):
        """Make a copy of the BluemiraFace"""
        return BluemiraFace(self.boundary, self.label)

    def deepcopy(self, label=None):
        """Make a copy of the BluemiraFace"""
        boundary = []
        for o in self.boundary:
            boundary += [o.deepcopy(o.label)]
        geo_copy = BluemiraFace(boundary)
        if label is not None:
            geo_copy.label = label
        else:
            geo_copy.label = self.label
        return geo_copy

    def _check_boundary(self, objs):
        """Check if objects in objs are of the correct type for this class"""
        if objs is None:
            return objs

        if not hasattr(objs, "__len__"):
            objs = [objs]
        check = False
        for c in self._boundary_classes:
            for o in objs:
                check = check or isinstance(o, c)
            if check:
                if all(o.is_closed() for o in objs):
                    return objs
                else:
                    raise NotClosedWire("Only closed BluemiraWire are accepted.")
        raise TypeError(
            f"Only {self._boundary_classes} objects can be used for {self.__class__}"
        )

    def _create_face(self, check_reverse=True):
        """Create the primitive face"""
        external: BluemiraWire = self.boundary[0]
        face = cadapi.apiFace(external._create_wire(check_reverse=False))

        if len(self.boundary) > 1:
            fholes = [cadapi.apiFace(h.shape) for h in self.boundary[1:]]
            face = face.cut(fholes)
            if len(face.Faces) == 1:
                face = face.Faces[0]
            else:
                raise DisjointedFace("Any or more than one face has been created.")

        if check_reverse:
            return self._check_reverse(face)
        else:
            return face

    def _create_shape(self) -> cadapi.apiFace:
        """Part.Face: shape of the object as a primitive face"""
        return self._create_face()

    @classmethod
    def _create(cls, obj: cadapi.apiFace, label="") -> BluemiraFace:
        if isinstance(obj, cadapi.apiFace):
            bmwires = []
            for w in obj.Wires:
                w_orientation = w.Orientation
                bm_wire = BluemiraWire(w)
                bm_wire._orientation = w_orientation
                if cadapi.is_closed(w):
                    bm_wire.close()
                bmwires += [bm_wire]

            bmface = cls(None, label=label)
            bmface._set_shape(obj)
            bmface._boundary = bmwires
            bmface._orientation = obj.Orientation

            return bmface

        raise TypeError(f"Only Part.Face objects can be used to create a {cls} instance")

    def discretize(
        self, ndiscr: int = 100, byedges: bool = False, dl: float = None
    ) -> np.ndarray:
        """
        Make an array of the geometry.

        Parameters
        ----------
        ndiscr: int
            Number of points in the array
        byedges: bool
            Whether or not to discretise by edges

        Returns
        -------
        xyz: np.ndarray
            (M, (3, N)) array of point coordinates where M is the number of boundaries
            and N the number of discretization points.
        """
        points = []
        for w in self.shape.Wires:
            if byedges:
                points.append(cadapi.discretize_by_edges(w, ndiscr=ndiscr, dl=dl))
            else:
                points.append(cadapi.discretize(w, ndiscr=ndiscr, dl=dl))
        return points

    @property
    def vertexes(self) -> Coordinates:
        """
        The vertexes of the wire.
        """
        return Coordinates(cadapi.vertexes(self.shape))

    @property
    def edges(self) -> Tuple[BluemiraWire]:
        """
        The edges of the wire.
        """
        return tuple([BluemiraWire(cadapi.apiWire(o)) for o in cadapi.edges(self.shape)])

    @property
    def wires(self) -> Tuple[BluemiraWire]:
        """
        The wires of the wire. By definition a list of itself.
        """
        return tuple([BluemiraWire(o) for o in cadapi.wires(self.shape)])

    @property
    def faces(self) -> Tuple[BluemiraFace]:
        """
        The faces of the wire. By definition an empty list.
        """
        return tuple(self)

    @property
    def shells(self) -> tuple:
        """
        The shells of the wire. By definition an empty list.
        """
        return ()

    @property
    def solids(self) -> tuple:
        """
        The solids of the wire. By definition an empty list.
        """
        return ()
