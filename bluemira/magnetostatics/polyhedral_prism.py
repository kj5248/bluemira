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
with arbitrarily shaped cross-section, following equations as described in:


"""
import math
import sys

import numpy as np

from bluemira.base.constants import MU_0
from bluemira.base.look_and_feel import bluemira_error
from bluemira.magnetostatics.baseclass import ArbitraryCrossSectionCurrentSource
from bluemira.magnetostatics.tools import process_xyz_array

__all__ = ["PolyhedralPrismCurrentSource"]


def h_field_x(j, nprime, nprime2, d, fpoint, lam, gam, psi):
    """
    Function to calculate magnetic field strength in x direction from working parameters
    """
    hx = (
        j
        / (4 * np.pi)
        * (
            (nprime2 * d - nprime * (nprime * fpoint[0] + nprime2 * fpoint[1])) * lam
            + (nprime * d - nprime * (-nprime2 * fpoint[0] + nprime * fpoint[1])) * gam
            - nprime * nprime2 * psi
        )
    )
    return hx


def h_field_y(j, nprime, nprime2, d, fpoint, lam, gam, psi):
    """
    Function to calculate magnetic field strength in y direction from working parameters
    """
    hy = (
        j
        / (4 * np.pi)
        * (
            -(nprime * d - nprime * (-nprime2 * fpoint[0] + nprime * fpoint[1])) * lam
            + (nprime2 * d - nprime * (nprime * fpoint[0] + nprime2 * fpoint[1])) * gam
            - nprime * nprime * psi
        )
    )
    return hy


def h_field_z(j, nprime, d, eta, zeta):
    """
    Function to calculate magnetic field strength in z direction from working parameters
    """
    hz = j / (4 * np.pi) * (nprime * d * eta - nprime**2 * zeta)
    return hz


class PolyhedralPrismCurrentSource(ArbitraryCrossSectionCurrentSource):
    """
    3-D polyhedral prism current source with arbitrary cross-section and
    uniform current distribution, cross section is in x-y and prism extends
    in z.

    The current direction is along the local z coordinate.

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
    n: int
        The number of sides of the prism cross section
    width: float, np.array(n-1)
        The width of the prism sides (same value for all or array of values
        for the sides) [m]
    length: float
        The length of the current source at the starting edge [m]
    alpha: float
        The first angle of the trapezoidal prism [rad]
    beta: float
        The second angle of the trapezoidal prism [rad]
    theta: float, np.array(n-1)
        Internal cross section angles (same value for all or array of values
        for the sides) [rad]
    angle_type: str
        Determine whether theta is an internal angle of the prism cross section
        or external from the vector direction of the line
    current: float
        The current flowing through the source [A]
    """

    def __init__(
        self,
        origin,
        ds,
        normal,
        t_vec,
        n,
        width,
        alpha,
        beta,
        theta,
        angle_type,
        current,
    ):
        self.origin = origin
        self.n = n
        self.length = np.linalg.norm(ds)
        self.alpha = alpha
        self.beta = beta
        self.current = current

        # direction vector for current
        # this is along direction vector
        self.j_hat = ds / np.linalg.norm(ds)
        # direction vector for magnetisation
        # perp to J (set to be perp to tvec
        # but anything perp to J would work)
        j_cross_tvec = np.cross(self.j_hat, t_vec)
        self.mc_hat = j_cross_tvec / -np.linalg.norm(j_cross_tvec)
        # direction vector for magnetisation value
        # perp to J and Mc
        j_cross_m = np.cross(self.j_hat, self.mc_hat)
        self.d_hat = j_cross_m / np.linalg.norm(j_cross_m)

        # Normalised direction cosine matrix
        self.dcm = np.array([t_vec, ds / self.length, normal])

        # setting up arrays for width and theta
        if isinstance(width, float) or isinstance(width, int):
            self.width = width * np.ones(self.n - 1)
        else:
            self.width = width
        if angle_type == "ext":
            if isinstance(theta, float):
                self.theta = theta * np.ones(self.n - 1)
            else:
                self.theta = theta
        elif angle_type == "int":
            if isinstance(theta, float):
                # converts from internal to external angle
                self.theta = (np.pi - theta) * np.ones(self.n - 1)
            else:
                # converts from internal to external angle
                self.theta = np.pi * np.ones(self.n - 1) - theta
        # points of the prism
        self.points = self._calculate_points()
        self.facepoints = self._calculate_face_points()
        # set X location as center of start face
        sum = np.array([0.0, 0.0, 0.0])
        for k in range(self.n):
            x = self.points[k, 0, 0]
            y = self.points[k, 0, 1]
            z = self.points[k, 0, 2]
            sum += np.array([x, y, z])
        self.x = sum / self.n
        # set up normals to all sides of prism
        self.normals = self._calculate_side_normals()
        # set up cosine angles nprime and nprime2
        self.angles = self._calculate_angles()

    def _calculate_points(self):
        """
        Function to calculate all the points of the prism.
        Points are calculated in local coordinates and then transformed to global.
        """
        points = []
        # sum of rotational angles
        sum_theta = 0
        dl = 0
        # starting length
        len = self.length
        # loop through sides but not last side as already will have those points
        for k in range(self.n - 1):
            # set width as w for ease of use
            w = self.width[k]
            # add rotational angle
            sum_theta += self.theta[k]
            # variation in x and y using width and theta
            dxz = np.array([w * math.cos(sum_theta), -w * math.sin(sum_theta)])
            # midpoint results in no z increase/decrease
            if np.abs(sum_theta) == math.pi:
                # difference betwen point 1, 2, 3, 4
                d_12 = [dxz[0], 0, 0]
                d_23 = [0, len, 0]
                d_34 = [-dxz[0], 0, 0]
            # first half of shape z decreases, depending on alpha, on front face
            # or depending on beta for rear face
            elif np.abs(sum_theta) < math.pi:
                # change in z at front and rear face respectively
                dy1 = dxz[1] / math.tan(self.alpha)
                dy2 = dxz[1] / math.tan(self.beta)
                dl += dy1 + dy2
                # difference betwen point 1, 2, 3, 4
                d_12 = np.array([dxz[0], dy1, dxz[1]])
                d_23 = np.array([0, len - dy1 - dy2, 0])
                d_34 = np.array([-dxz[0], dy2, -dxz[1]])
            # second half of shape z increases, depending on alpha, on front face
            # or depending on beta for rear face
            else:
                # change in z at front and rear face respectively
                dy1 = np.abs(dxz[1]) / math.tan(self.alpha)
                dy2 = np.abs(dxz[1]) / math.tan(self.beta)
                dl += dy1 + dy2
                # difference betwen point 1, 2, 3, 4
                d_12 = np.array([dxz[0], dy1, dxz[1]])
                d_23 = np.array([0, len - dy1 - dy2, 0])
                d_34 = np.array([-dxz[0], dy2, -dxz[1]])
            # check for if shape gets inverted ie front and rear face overlap
            if (dl) > len:
                bluemira_error(
                    "length at maximum shape depth becomes negative. Change either the angles (alpha, beta) or reduce shape depth."
                )
                sys.exit()
            # points of side set
            if k == 0:
                p1 = np.array([0, 0, 0])
                p2 = p1 + d_12
                p3 = p2 + d_23
                p4 = p3 + d_34
            else:
                p1 = p2
                p4 = p3
                p2 = p1 + d_12
                p3 = p2 + d_23
            # set edge length to new starting edge
            len = math.dist(p3, p2)
            # final side from other points that already exist and adding points to array
            if k == (self.n - 2):
                points += [np.vstack([p1, p2, p3, p4, p1])]
                p = np.array([p2[0] - p1[0], p2[1] - p1[1]])
                p1 = p2
                p4 = p3
                p2 = np.array([0, 0, 0])
                p3 = np.array([0, len, 0])
                points += [np.vstack([p1, p2, p3, p4, p1])]
                q = np.array([p2[0] - p1[0], p2[1] - p1[1]])
                w = math.sqrt((dxz[0] * dxz[0] + dxz[1] * dxz[1]))
                self.width = np.append(self.width, w)
                theta = math.acos(np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q)))
                self.theta = np.append(self.theta, theta)
            # if not final side just adds points to array
            else:
                points += [np.vstack([p1, p2, p3, p4, p1])]
        # reorganise array and convert points to global
        points_array = []
        for p in points:
            points_array.append(self._local_to_global(p))

        return np.array(points_array)

    def _calculate_face_points(self):
        """
        Takes the prism vertex points and extracts the faces from them
        """
        face1 = []
        face2 = []
        # loop through all sides to extract end points
        for k in range(self.n):
            face1 += [self.points[k, 0, :]]
            face2 += [self.points[k, 3, :]]
        # add face 1
        points = [np.vstack([np.array(face1)])]
        # add face 2
        points += [np.vstack([np.array(face2)])]
        points_array = []
        # reorganise array
        for p in points:
            points_array.append(p)
        return np.array(points_array)

    def _calculate_mc(self, p):
        """
        Provides value of Mc at point p
        """

        if self._inside_outside(p) == "inside":
            # if p inside shape or on edge:
            mc = self.current * self._calculate_vector_distance(p, self.d_hat)
        else:
            # if p outside shape
            mc = 0.0
        return mc * self.mc_hat

    def _inside_outside(self, p):
        """
        Function to determine if a point p is inside or outside the prism,
        on the surface counts as inside.
        """
        pos = 0
        neg = 0
        # loop through sides
        for k in range(self.n):
            n = self.normals[k, :]
            v = p - self.points[k, 0, :]
            # value to determine where p is with respect to side
            dp = np.dot(v, n)
            if dp > 0:
                pos += 1
            elif dp < 0:
                neg += 1
            else:
                pos += 1
                neg += 1
        # loop through faces
        for k in range(2):
            n = self.normals[k + self.n, :]
            v = p - self.facepoints[k, 0, :]
            # value to determine where p is with respect to side
            dp = np.dot(v, n)
            if dp > 0:
                pos += 1
            elif dp < 0:
                neg += 1
            else:
                pos += 1
                neg += 1
        if (pos == self.n + 2) or (neg == self.n + 2):
            # if p inside shape or on edge:
            pos = "inside"
        else:
            # if p outside shape
            pos = "outside"

        return pos

    def _calculate_side_normals(self):
        """
        Calculate the normals for all the sides and the two faces of the prism.
        These are all returned as normalised vectors
        """
        normals = []
        # normals from sides
        for k in range(self.n):
            p = self.points[k, :, :]
            # vectors for l1 and l2
            u = [p[0, 0] - p[3, 0], p[0, 1] - p[3, 1], p[0, 2] - p[3, 2]]
            v = [p[1, 0] - p[0, 0], p[1, 1] - p[0, 1], p[1, 2] - p[0, 2]]
            # cross product to determine normal to plane
            u_cross_v = np.cross(u, v)
            # normalise cross product vector
            n_hat = np.linalg.norm(u_cross_v)
            normals += [u_cross_v / n_hat]
        # normals from faces
        for k in range(2):
            p = self.facepoints[k, :, :]
            # vectors for l1 and l2
            u = [p[0, 0] - p[3, 0], p[0, 1] - p[3, 1], p[0, 2] - p[3, 2]]
            v = [p[1, 0] - p[0, 0], p[1, 1] - p[0, 1], p[1, 2] - p[0, 2]]
            # cross product to determine normal to plane
            u_cross_v = np.cross(u, v)
            # normalise cross product vector
            n_hat = np.linalg.norm(u_cross_v)
            normals += [u_cross_v / n_hat]
        # reorganise into array
        normals_array = []
        for n in normals:
            normals_array.append(n)
        return np.array(normals_array)

    def _calculate_angles(self):
        """
        Calculate the cosine of the angles between the normal with Mc and Mc cross J
        (n' and n'' respectively)
        """
        angle_arr = []
        # iterate through all the sides and faces
        for k in range(self.n + 2):
            n = self.normals[k, :]
            o = np.array([0, 0, 0])
            # cosine of the angle between side normal and magnetisation direction
            n_prime = np.dot(n, self.mc_hat) / (
                math.dist(n, o) * math.dist(self.mc_hat, o)
            )
            # sets value to zero if small
            if np.abs(n_prime) < 1e-8:
                n_prime = 0.0
            # cosine of angle between side normal and magnetisation value direction
            n_prime2 = np.dot(n, self.d_hat) / (
                math.dist(n, o) * math.dist(self.d_hat, o)
            )
            # sets value to zero if small
            if np.abs(n_prime2) < 1e-8:
                n_prime2 = 0.0
            angle_arr.append(np.vstack([n_prime, n_prime2]))
        return np.array(angle_arr)

    def _calculate_vector_distance(self, p, v):
        """
        Calculate distance along vector v between point X and p
        """
        # new point fpr p that is on plane of X
        p_prime = p - (np.dot(np.dot(p - self.x, v), v))
        # calculates distance between p' and X along vector Dhat
        d = np.dot(p_prime - self.x, self.d_hat)
        return d

    def _face_cutting(self, p):
        """
        Splits shape made up of points p into triangles and trapezoids
        """
        shapes = []
        for i in range(self.n - 4):
            if i == 0:
                # makes the first shape and loops back to start point
                cut = np.append(p[:3, :], p[:1, :], axis=0)
            else:
                # makes remaining shapes and loops back to start point
                cut = np.append(p[(0, i + 1, i + 2), :], p[:1, :], axis=0)
            # adds new shapes to list
            shapes += [np.vstack([cut])]
        # creates final shape
        cut = np.append(p[:1, :], p[-3:, :], axis=0)
        # adds start point to final shape
        cut = np.append(cut, p[:1, :], axis=0)
        # adds final shape to list
        shapes += [np.vstack([cut])]
        # rearranges shapes into array
        shape_arr = []
        for s in shapes:
            shape_arr += [s]
        return shape_arr

    def _position_vector(self, point, fpoint):
        """
        Creates a vector R from the shape vertices to the fieldpoint
        """
        r1 = fpoint - point[0, :]
        r2 = fpoint - point[1, :]
        r3 = fpoint - point[2, :]
        r4 = fpoint - point[3, :]
        return np.array([r1, r2, r3, r4])

    def _length_vector(self, p, q, r):
        """
        Creates a set of vectors that go from point p to q on an existing shape
        """
        # position vector for point p
        rp = r[p - 1, :]
        # position vector for point q
        rq = r[q - 1, :]
        # length vector from p to q using position vectors
        len = rp - rq
        return len

    def _vector_coordinates(self, point, fpoint):
        """
        Creates a set of vector working coordinates that are used to calculate the
        magnetic field for the prism. Does this for a side of the prism at a time.
        """
        # vector R between vertices and fieldpoint
        r = self._position_vector(point, fpoint)
        # vector that sets z direction
        zhat = self._length_vector(1, 4, r) / np.linalg.norm(
            self._length_vector(1, 4, r)
        )
        # value of z change between vertex 1 and p (=2,3,4)
        zp = np.array(
            [
                [np.dot(zhat, self._length_vector(1, 2, r))],
                [np.dot(zhat, self._length_vector(1, 3, r))],
                [np.dot(zhat, self._length_vector(1, 4, r))],
            ]
        )
        # value d which is width of shape
        d = np.sqrt(np.linalg.norm(self._length_vector(1, 2, r)) ** 2 - zp[0] ** 2)
        # vector that sets x direction
        xhat = np.cross(self._length_vector(1, 2, r), zhat) / d
        # vector that sets y direction
        yhat = np.cross(zhat, xhat)
        # x value (between vertex and fieldpoint)
        x = np.dot(xhat, r[0, :])
        # x = fpoint[0]
        # if x suitably small set to 0
        if x < 1e-12:
            x = 0.0
        # y value (between vertex and fieldpoint)
        y = np.dot(yhat, r[0, :])
        # y = fpoint[1]
        # if y suitably small set to 0
        if y < 1e-12:
            y = 0.0
        # z value (between vertex and fieldpoint)
        z = np.dot(zhat, r[0, :])
        # z = fpoint[2]
        # if z suitably small set to 0
        if z < 1e-12:
            z = 0.0
        # projections of area vectors along normal to trapezoid side
        p12 = np.dot(xhat, np.cross(r[0, :], r[1, :]))
        p34 = np.dot(-xhat, np.cross(r[2, :], r[3, :]))
        # scalar products of area vector
        q12 = np.dot(r[0, :], self._length_vector(1, 2, r))
        q34 = -np.dot(r[3, :], self._length_vector(3, 4, r))
        # calculates some parameters to simplify later equations ie lam, psi, eta...
        lambda_12 = np.log(
            (
                np.linalg.norm(r[0, :]) * np.linalg.norm(self._length_vector(1, 2, r))
                - q12
            )
            / (
                (np.linalg.norm(r[1, :]) + np.linalg.norm(self._length_vector(1, 2, r)))
                * np.linalg.norm(self._length_vector(1, 2, r))
                - q12
            )
        )
        lambda_34 = np.log(
            (
                (np.linalg.norm(r[2, :]) + np.linalg.norm(self._length_vector(3, 4, r)))
                * np.linalg.norm(self._length_vector(3, 4, r))
                - q34
            )
            / (
                np.linalg.norm(r[3, :]) * np.linalg.norm(self._length_vector(3, 4, r))
                - q34
            )
        )
        # major term used in calculation of h field
        lam = (
            np.log(
                ((np.linalg.norm(r[0, :]) + z) * (np.linalg.norm(r[2, :]) + z - zp[1]))
                / (
                    (np.linalg.norm(r[1, :]) + z - zp[0])
                    * (np.linalg.norm(r[3, :]) + z - zp[2])
                )
            )
            + zp[0] * lambda_12 / np.linalg.norm(self._length_vector(1, 2, r))
            + (zp[1] - zp[2]) * lambda_34 / np.linalg.norm(self._length_vector(3, 4, r))
        )
        # components of gamma equation to simplify equation
        a1 = z * q12 - zp[0] * np.linalg.norm(r[0, :]) ** 2
        a2 = x * np.linalg.norm(r[0, :]) * d
        b1 = (z - zp[0]) * (
            q12 - np.linalg.norm(self._length_vector(1, 2, r)) ** 2
        ) - zp[0] * np.linalg.norm(r[1, :]) ** 2
        b2 = x * np.linalg.norm(r[1, :]) * d
        c1 = (z - zp[1]) * (q34 - np.linalg.norm(self._length_vector(3, 4, r)) ** 2) - (
            zp[1] - zp[2]
        ) * np.linalg.norm(r[2, :]) ** 2
        c2 = x * np.linalg.norm(r[2, :]) * d
        d1 = (z - zp[2]) * q34 - (zp[1] - zp[2]) * np.linalg.norm(r[3, :]) ** 2
        d2 = x * np.linalg.norm(r[3, :]) * d
        # major term used in calculation of h field
        gam = (
            np.arctan2(a1, a2)
            - np.arctan2(b1, b2)
            + np.arctan2(c1, c2)
            - np.arctan2(d1, d2)
        )
        # major term used in calculation of h field
        psi = (
            d
            * zp[0]
            * (np.linalg.norm(r[0, :]) - np.linalg.norm(r[1, :]))
            / (np.linalg.norm(self._length_vector(1, 2, r)) ** 2)
            + d
            * (zp[1] - zp[2])
            * (np.linalg.norm(r[2, :]) - np.linalg.norm(r[3, :]))
            / (np.linalg.norm(self._length_vector(3, 4, r)) ** 2)
            - d
            * d
            * (
                p12 * lambda_12 / (np.linalg.norm(self._length_vector(1, 2, r)) ** 3)
                + p34 * lambda_34 / (np.linalg.norm(self._length_vector(3, 4, r)) ** 3)
            )
        )
        # major term used in calculation of h field
        eta = d * (
            lambda_12 / np.linalg.norm(self._length_vector(1, 2, r))
            + lambda_34 / np.linalg.norm(self._length_vector(3, 4, r))
        )
        # major term used in calculation of h field
        zeta = (
            d
            * d
            * (
                (np.linalg.norm(r[0, :]) - np.linalg.norm(r[1, :]))
                / (np.linalg.norm(self._length_vector(1, 2, r)) ** 2)
                + (np.linalg.norm(r[2, :]) - np.linalg.norm(r[3, :]))
                / (np.linalg.norm(self._length_vector(3, 4, r)) ** 2)
                + q12 * lambda_12 / (np.linalg.norm(self._length_vector(1, 2, r)) ** 3)
                + q34 * lambda_34 / (np.linalg.norm(self._length_vector(3, 4, r)) ** 3)
            )
        )
        # coordinates for x,y,z from perspective of shape
        coords = np.array([x, y, z])
        # returns parameters needed in h field calculation
        return coords, lam, gam, psi, eta, zeta

    def _hxhyhz(self, fpoint):
        """
        Produces h field at fieldpoint using h field functions and vector coordinates
        """
        h_array = []
        j = self.current
        # h field contribution from sides
        for k in range(self.n):
            # get pre-calculated values
            points = self.points[k, :, :]
            nprime = self.angles[k, 0]
            nprime2 = self.angles[k, 1]
            # get vector values from functions
            d = self._calculate_vector_distance(points[0, :], self.normals[k, :])
            coords, lam, gam, psi, eta, zeta = self._vector_coordinates(points, fpoint)
            # calculate h field in all directions
            hx = h_field_x(j, nprime, nprime2, d, coords, lam, gam, psi)
            hy = h_field_y(j, nprime, nprime2, d, coords, lam, gam, psi)
            hz = h_field_z(j, nprime, d, eta, zeta)
            # add outputs to array
            h_array.append(np.hstack([hx, hy, hz]))

        # h field contribution from faces
        for k in range(2):
            # get pre-calculated values
            points = self.facepoints[k, :, :]
            nprime = self.angles[k + self.n, 0]
            # if there are more than 4 sides to a face
            # it must be broken down into 3 or 4 sided shapes
            if self.n > 4:
                # splits face into smaller shapes
                shapes = self._face_cutting(points)
                # cycle through shapes to calculate field
                for i in range(self.n - 4):
                    shape = shapes[i]
                    dist = []
                    for p in shape:
                        dist += [self._calculate_vector_distance(p, self.normals[k, :])]
                    # takes distance as the maximum from all the points
                    d = np.max(dist)
                    # value always zero for face
                    nprime2 = 0
                    # get working vector coordinates
                    coords, lam, gam, psi, eta, zeta = self._vector_coordinates(
                        points, fpoint
                    )
                    # calcaulte h field
                    hx = h_field_x(j, nprime, nprime2, d, coords, lam, gam, psi)
                    hy = h_field_y(j, nprime, nprime2, d, coords, lam, gam, psi)
                    hz = h_field_z(j, nprime, d, eta, zeta)
                    # add outputs to array
                    h_array.append(np.hstack([hx, hy, hz]))
            # has few enough sides so can just straight calculate fields
            else:
                dist = []
                # calculate the distance for all points of face
                for p in points:
                    dist += [self._calculate_vector_distance(p, self.normals[k, :])]
                # take the distance to be maximum value
                d = np.max(dist)
                # value always zero for face
                nprime2 = 0
                # get working vector coordinates
                coords, lam, gam, psi, eta, zeta = self._vector_coordinates(
                    points, fpoint
                )
                # calcaulte h field
                hx = h_field_x(j, nprime, nprime2, d, coords, lam, gam, psi)
                hy = h_field_y(j, nprime, nprime2, d, coords, lam, gam, psi)
                hz = h_field_z(j, nprime, d, eta, zeta)
                # add outputs to array
                h_array.append(np.hstack([hx, hy, hz]))

        # reorganise array and include magnetisation before returning
        hx = 0
        hy = 0
        hz = 0
        for h in h_array:
            hx += h[0]
            hy += h[1]
            hz += h[2]
        h = [hx, hy, hz]
        h = h + self._calculate_mc(fpoint)
        return h

    def _bxbybz(self, point):
        """
        Calculate the b field at a fieldpoint using h field calculator and
        converting to b field
        """
        h = self._hxhyhz(point)
        b = (h + self._calculate_mc(point)) * MU_0
        b_field = []
        # convert negligible values to 0
        for b_xyz in b:
            if np.abs(b_xyz) < 1e-12:
                b_xyz = 0
            b_field += [b_xyz]
        b_field = np.array([b_field[0], b_field[1], b_field[2]])
        return b_field

        # def _bxbybz_in(point):
        """
        Calculate the b field at a point inside (or on the surface of) the prism
        """

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
        if self._inside_outside == "inside":
            # new magnetic field calculation not setup yet
            # b = self._bxbybz_in(point)
            b = self._bxbybz(point)
        else:
            b = self._bxbybz(point)
        return b
