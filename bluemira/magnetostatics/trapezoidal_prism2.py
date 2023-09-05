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
Analytical expressions for the field inside an arbitrarily shaped winding pack
of rectangular cross-section, following equations as described in:

https://onlinelibrary.wiley.com/doi/epdf/10.1002/jnm.594?saml_referrer=
including corrections from:
https://onlinelibrary.wiley.com/doi/abs/10.1002/jnm.675
"""
import math

import numba as nb
import numpy as np

from bluemira.base.constants import MU_0_4PI
from bluemira.magnetostatics.baseclass import RectangularCrossSectionCurrentSource
from bluemira.magnetostatics.tools import process_xyz_array

__all__ = ["TrapezoidalPrismCurrentSource2"]


def cuboid_field(point, breadth, depth, height, current):
    x = [0, point[0] - depth / 2, point[0] + depth / 2]
    y = [0, point[1] - breadth / 2, point[1] + breadth / 2]
    z = [0, point[2] - height / 2, point[2] + height / 2]
    Hx = 0
    Hy = 0
    Hz = 0
    for i in range(1, 3):
        for j in range(1, 3):
            for k in range(1, 3):
                A = x[i] * math.log(z[k] + np.sqrt(x[i] ** 2 + y[j] ** 2 + z[k] ** 2))
                B = z[k] * math.log(x[i] + np.sqrt(x[i] ** 2 + y[j] ** 2 + z[k] ** 2))
                C = -y[j] * np.arctan(
                    x[i] * z[k] / (y[j] * np.sqrt(x[i] ** 2 + y[j] ** 2 + z[k] ** 2))
                )
                Hx += -current / 4 / np.pi * (-1) ** (i + j + k) * (A + B + C)
                D = y[j] * math.log(z[k] + np.sqrt(x[i] ** 2 + y[j] ** 2 + z[k] ** 2))
                E = z[k] * math.log(y[j] + np.sqrt(x[i] ** 2 + y[j] ** 2 + z[k] ** 2))
                F = -x[i] * np.arctan(
                    y[j] * z[k] / (x[i] * np.sqrt(x[i] ** 2 + y[j] ** 2 + z[k] ** 2))
                )
                Hy += current / 4 / np.pi * (-1) ** (i + j + k) * (D + E + F)
    return Hx, Hy, Hz


class TrapezoidalPrismCurrentSource2(RectangularCrossSectionCurrentSource):
    """
    3-D trapezoidal prism current source with a retangular cross-section and
    uniform current distribution.

    The current direction is along the local y coordinate.

    Parameters
    ----------
    origin: np.array(3)
        The origin of the current source in global coordinates [m]
    ds: np.array(3)
        The direction vector of the current source in global coordinates [m]
    normal: np.array(3)
        The normalised normal vector of the current source in global coordinates [m]
    t_vec: np.array(3)
        The normalised tangent vector of the current source in global coordinates [m]
    breadth: float
        The breadth of the current source (half-width) [m]
    depth: float
        The depth of the current source (half-height) [m]
    alpha: float
        The first angle of the trapezoidal prism [rad]
    beta: float
        The second angle of the trapezoidal prism [rad]
    current: float
        The current flowing through the source [A]
    """

    def __init__(self, origin, ds, normal, t_vec, breadth, depth, alpha, beta, current):
        self.origin = origin

        length = np.linalg.norm(ds)
        self._halflength = 0.5 * length
        # Normalised direction cosine matrix
        self.dcm = np.array([t_vec, ds / length, normal])
        self.length = 0.5 * (length - breadth * np.tan(alpha) - breadth * np.tan(beta))
        self.breadth = breadth
        self.depth = depth
        self.alpha = alpha
        self.beta = beta
        # Current density
        self.current = current
        self.area = depth * breadth
        self.J = self.current / self.area
        # self.rho = current / (4 * breadth * depth)
        self.points = self._calculate_points()

    @process_xyz_array
    def field(self, x, y, z):
        """
        Calculate the magnetic field at a point due to the current source.

        Parameters
        ----------
        x: Union[float, np.array]
            The x coordinate(s) of the points at which to calculate the field
        y: Union[float, np.array]
            The y coordinate(s) of the points at which to calculate the field
        z: Union[float, np.array]
            The z coordinate(s) of the points at which to calculate the field

        Returns
        -------
        field: np.array(3)
            The magnetic field vector {Bx, By, Bz} in [T]
        """
        point = np.array([x, y, z])
        # Convert to local coordinates
        point = self._global_to_local([point])[0]
        # Evaluate field in local coordinates
        if (self.alpha == np.pi) and (self.beta == np.pi):
            b_local = cuboid_field(point, self.breadth, self.depth, self.length, self.J)
        else:
            if (self.alpha == np.pi) and (self.beta < np.pi):
                length = self.length - self.breadth * np.tan(self.beta)
                b_cuboid = cuboid_field(point, self.breadth, self.depth, length, self.J)
            elif (self.beta == np.pi) and (self.alpha < np.pi):
                length = self.length - self.breadth * np.tan(self.alpha)
                b_cuboid = cuboid_field(point, self.breadth, self.depth, length, self.J)
            elif (self.alpha < np.pi) and (self.beta < np.pi):
                length = (
                    self.length
                    - self.breadth * np.tan(self.alpha)
                    - self.breadth * np.tan(self.beta)
                )
                b_cuboid = cuboid_field(point, self.breadth, self.depth, length, self.J)
            elif (self.alpha == np.pi) and (self.beta > np.pi):
                length = self.length + self.breadth * np.tan(self.beta)
                b_cuboid = cuboid_field(point, self.breadth, self.depth, length, self.J)
            elif (self.beta == np.pi) and (self.alpha > np.pi):
                length = self.length + self.breadth * np.tan(self.alpha)
                b_cuboid = cuboid_field(point, self.breadth, self.depth, length, self.J)
            elif (self.alpha > np.pi) and (self.beta > np.pi):
                length = (
                    self.length
                    + self.breadth * np.tan(self.alpha)
                    + self.breadth * np.tan(self.beta)
                )
                b_cuboid = cuboid_field(point, self.breadth, self.depth, length, self.J)
            elif (self.alpha < np.pi) and (self.beta > np.pi):
                length = (
                    self.length
                    - self.breadth * np.tan(self.alpha)
                    + self.breadth * np.tan(self.beta)
                )
                b_cuboid = cuboid_field(point, self.breadth, self.depth, length, self.J)
            elif (self.alpha > np.pi) and (self.beta < np.pi):
                length = (
                    self.length
                    + self.breadth * np.tan(self.alpha)
                    - self.breadth * np.tan(self.beta)
                )
                b_cuboid = cuboid_field(point, self.breadth, self.depth, length, self.J)
            b_local = b_cuboid

        # Convert vector back to global coordinates
        return self.dcm.T @ b_local

    def _calculate_points(self):
        """
        Calculate extrema points of the current source for plotting and debugging.
        """
        b = self._halflength
        c = self.depth / 2
        d = self.breadth / 2
        # Lower rectangle
        p1 = np.array([-c, -d, -b + d * np.tan(self.beta)])
        p2 = np.array([-c, d, -b - d * np.tan(self.beta)])
        p3 = np.array([c, d, -b - d * np.tan(self.beta)])
        p4 = np.array([c, -d, -b + d * np.tan(self.beta)])
        # Upper rectangle
        p5 = np.array([-c, -d, b - d * np.tan(self.alpha)])
        p6 = np.array([-c, d, b + d * np.tan(self.alpha)])
        p7 = np.array([c, d, b + d * np.tan(self.alpha)])
        p8 = np.array([c, -d, b - d * np.tan(self.alpha)])

        points_array = []
        points = [
            np.vstack([p1, p2, p3, p4, p1]),
            np.vstack([p5, p6, p7, p8, p5]),
            # Lines between rectangle corners
            np.vstack([p1, p5]),
            np.vstack([p2, p6]),
            np.vstack([p3, p7]),
            np.vstack([p4, p8]),
        ]

        for p in points:
            points_array.append(self._local_to_global(p))

        return np.array(points_array, dtype=object)
