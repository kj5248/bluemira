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
from bluemira.geometry.coordinates import rotation_matrix_v1v2
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
        # this is along prism normal
        # self.J_hat = normal
        self.j_hat = ds / np.linalg.norm(ds)
        # direction vector for magnetisation
        # perp to J
        j_cross_tvec = np.cross(self.j_hat, t_vec)
        self.mc_hat = j_cross_tvec / -np.linalg.norm(j_cross_tvec)
        # direction vector for magnetisation value
        # perp to J and Mc
        j_cross_m = np.cross(self.j_hat, self.mc_hat)
        self.D_hat = j_cross_m / np.linalg.norm(j_cross_m)

        # Normalised direction cosine matrix
        self.dcm = np.array([normal, t_vec, ds / self.length])

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
        self.normals = self._calculate_side_normals()
        self.angles = self._calculate_angles()
        self.geometry = self._calculate_geometric_parameters()

    def _calculate_points(self):
        # points
        points = []
        # sum of rotational angles
        sum_theta = 0
        dl = 0
        # starting length
        len = self.length
        for k in range(self.n - 1):
            w = self.width[k]
            sum_theta += self.theta[k]
            dxy = np.array([w * math.cos(sum_theta), -w * math.sin(sum_theta)])
            if np.abs(sum_theta) == math.pi:
                # difference betwen point 1, 2, 3, 4
                d_12 = [dxy[0], 0, 0]
                d_23 = [0, 0, len]
                d_34 = [-dxy[0], 0, 0]
            elif np.abs(sum_theta) < math.pi:
                dz1 = dxy[1] / math.tan(self.alpha)
                dz2 = dxy[1] / math.tan(self.beta)
                dl += dz1 + dz2
                # difference betwen point 1, 2, 3, 4
                d_12 = np.array([dxy[0], dxy[1], dz1])
                d_23 = np.array([0, 0, len - dz1 - dz2])
                d_34 = np.array([-dxy[0], -dxy[1], dz2])
            else:
                dz1 = np.abs(dxy[1]) / math.tan(self.alpha)
                dz2 = np.abs(dxy[1]) / math.tan(self.beta)
                dl += dz1 + dz2
                # difference betwen point 1, 2, 3, 4
                d_12 = np.array([dxy[0], dxy[1], dz1])
                d_23 = np.array([0, 0, len - dz1 - dz2])
                d_34 = np.array([-dxy[0], -dxy[1], dz2])
            if (dl) > len:
                bluemira_error(
                    "length at maximum shape depth becomes negative. Change either the angles (alpha, beta) or reduce shape depth."
                )
                sys.exit()
            # points of side
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
            if k == (self.n - 2):
                points += [np.vstack([p1, p2, p3, p4, p1])]
                p = np.array([p2[0] - p1[0], p2[1] - p1[1]])
                p1 = p2
                p4 = p3
                p2 = np.array([0, 0, 0])
                p3 = np.array([0, 0, len])
                points += [np.vstack([p1, p2, p3, p4, p1])]
                q = np.array([p2[0] - p1[0], p2[1] - p1[1]])
                w = math.sqrt((dxy[0] * dxy[0] + dxy[1] * dxy[1]))
                self.width = np.append(self.width, w)
                theta = math.acos(np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q)))
                self.theta = np.append(self.theta, theta)
            else:
                points += [np.vstack([p1, p2, p3, p4, p1])]

        points_array = []
        for p in points:
            points_array.append(self._local_to_global(p))

        return np.array(points_array)

    def _calculate_face_points(self):
        # face points
        face1 = []
        face2 = []
        for k in range(self.n):
            face1 += [self.points[k, 0, :]]
            face2 += [self.points[k, 3, :]]
        points = [np.vstack([np.array(face1)])]
        points += [np.vstack([np.array(face2)])]
        points_array = []
        for p in points:
            points_array.append(p)
        return np.array(points_array)

    def _calculate_mc(self, p):
        """
        Provides value of Mc at point p
        """
        pos = 0
        neg = 0
        for k in range(self.n):
            n = self.normals[k, :]
            v = p - self.points[k, 0, :]
            dp = np.dot(v, n)
            if dp > 0:
                pos += 1
            elif dp < 0:
                neg += 1
            else:
                pos += 1
                neg += 1
        for k in range(2):
            n = self.normals[k + self.n, :]
            v = p - self.facepoints[k, 0, :]
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
            mc = self.current * self._calculate_vector_distance(p, self.d_hat)
        else:
            # if p outside shape
            mc = 0.0
        return mc * self.mc_hat

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
        for k in range(self.n + 2):
            n = self.normals[k, :]
            o = np.array([0, 0, 0])
            n_prime = np.dot(n, self.mc_hat) / (
                math.dist(n, o) * math.dist(self.mc_hat, o)
            )
            if np.abs(n_prime) < 1e-8:
                n_prime = 0.0
            n_prime2 = np.dot(n, self.d_hat) / (
                math.dist(n, o) * math.dist(self.d_hat, o)
            )
            if np.abs(n_prime2) < 1e-8:
                n_prime2 = 0.0
            angle_arr.append(np.vstack([n_prime, n_prime2]))
        return np.array(angle_arr)

    def _calculate_vector_distance(self, p, v):
        """
        Calculate distance along vector v between point X and p
        """
        p_prime = p - (np.dot(np.dot(p - self.x, v), v))
        d = np.dot(p_prime - self.x, self.d_hat)
        return d

    def _rotational_matrix(self, k):
        """
        Creates rotational matrix that would rotate points of a side
        k such that the side normal becomes parallel to x_hat and the length
        direction of the side (along J) is parallel to z_hat
        """
        n = self.normals[k, :]
        n2 = np.cross(n, self.J_hat)
        x_hat = np.array([1.0, 0.0, 0.0])
        r = rotation_matrix_v1v2(x_hat, n)
        n2 = np.matmul(r, n2)
        x_hat2 = np.array([0.0, -1.0, 0.0])
        r2 = rotation_matrix_v1v2(x_hat2, n2)
        rt = np.matmul(r2, r)
        return np.transpose(rt)

    def _translation_matrix(self, v):
        """
        Translation matrix for vector v
        """
        t = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [v[0], v[1], v[2], 1]])
        return t

    def _face_cutting(self, p):
        """
        Splits shape made up of points p into triangles and trapezoids
        """
        shapes = []
        for i in range(self.n - 4):
            if i == 0:
                cut = np.append(p[:3, :], p[0, :])
            else:
                cut = np.append(p[(0, i + 1, i + 2), :], p[0, :])
            shapes += [np.vstack([cut])]
        cut = np.append(p[0, :], p[-3:, :])
        cut = np.append(cut, p[0, :])
        shapes += [np.vstack([cut])]
        shape_arr = []
        for s in shapes:
            shape_arr.append(s)
        return np.array([shape_arr], dtype=object)

    def _working_coordinates(self, p, fpoint):
        """
        Get all the working parameters of a side/ face with points p
        for a field point fpoint. p and fpoint must already be rotated
        when inputted
        """
        if self.n == 3:
            p = np.append(p[:2, :], p[1:, :])
        # array for distance between source and field point
        r = np.array(
            [
                math.dist(fpoint, p[0, :]),
                math.dist(fpoint, p[1, :]),
                math.dist(fpoint, p[2, :]),
                math.dist(fpoint, p[3, :]),
            ]
        )
        # length 1->2 and 3->4
        l_12 = math.dist(p[1, :], p[0, :])
        l_34 = math.dist(p[3, :], p[2, :])
        # set needed values from points
        d = np.abs(p[0, 1] - p[2, 1])
        z2 = p[1, 2]
        z3 = p[2, 2]
        z4 = p[3, 2]
        p_12 = -z2 * fpoint[1] + d * fpoint[2]
        p_34 = -(z3 - z4) * fpoint[1] + d * (fpoint[2] - z4)
        q_12 = d * fpoint[1] + z2 * fpoint[2]
        q_34 = d * fpoint[1] + (z3 - z4) * (fpoint[2] - z4)
        lambda_12 = np.log((r[0] * l_12 - q_12) / ((r[1] + l_12) * l_12 - q_12))
        lambda_34 = np.log(((r[2] + l_34) * l_34 - q_34) / (r[3] * l_34 - q_34))
        lam = (
            np.log(
                ((r[0] + fpoint[2]) * (r[2] + fpoint[2] - z3))
                / ((r[1] + fpoint[2] - z2) * (r[3] + fpoint[2] - z4))
            )
            + z2 * lambda_12 / l_12
            + (z3 - z4) * lambda_34 / l_34
        )
        gam = (
            math.atan((fpoint[2] * q_12 - z2 * r[0] * r[0]) / (fpoint[0] * r[0] * d))
            - math.atan(
                ((fpoint[2] - z2) * (q_12 - l_12 * l_12) - z2 * r[1] * r[1])
                / (fpoint[0] * r[1] * d)
            )
            + math.atan(
                ((fpoint[2] - z3) * (q_34 - l_34 * l_34) - (z3 - z4) * r[2] * r[2])
                / (fpoint[0] * r[2] * d)
            )
            - math.atan(
                ((fpoint[2] - z4) * q_34 - (z3 - z4) * r[3] * r[3])
                / (fpoint[0] * r[3] * d)
            )
        )
        psi = (
            d * z2 * (r[0] - r[1]) / (l_12 * l_12)
            + d * (z3 - z4) * (r[2] - r[3]) / (l_34 * l_34)
            - d * d * (p_12 * lambda_12 / (l_12**3) + p_34 * lambda_34 / (l_34**3))
        )
        eta = d * (lambda_12 / l_12 + lambda_34 / l_34)
        zeta = (
            d
            * d
            * (
                (r[0] - r[1]) / (l_12 * l_12)
                + (r[2] - r[3]) / (l_34 * l_34)
                + q_12 * lambda_12 / (l_12**3)
                + q_34 * lambda_34 / (l_34**3)
            )
        )
        return fpoint, lam, gam, psi, eta, zeta

    def _position_vector(self, point, fpoint):
        r1 = fpoint - point[0, :]
        r2 = fpoint - point[1, :]
        r3 = fpoint - point[2, :]
        r4 = fpoint - point[3, :]
        return np.array([r1, r2, r3, r4])

    def _length_vector(self, p, q, r):
        rp = r[p - 1, :]
        rq = r[q - 1, :]
        len = rp - rq
        return len

    def _vector_coordinates(self, point, fpoint):
        r = self._position_vector(point, fpoint)
        zhat = self._length_vector(1, 4, r) / np.linalg.norm(
            self._length_vector(1, 4, r)
        )
        zp = np.array(
            [
                [np.dot(zhat, self._length_vector(1, 2, r))],
                [np.dot(zhat, self._length_vector(1, 3, r))],
                [np.dot(zhat, self._length_vector(1, 4, r))],
            ]
        )
        d = np.sqrt(np.linalg.norm(self._length_vector(1, 2, r)) ** 2 - zp[0] ** 2)
        xhat = np.cross(self._length_vector(1, 2, r), zhat) / d
        yhat = np.cross(zhat, xhat)
        x = np.dot(xhat, r[0, :])
        if x < 1e-8:
            x = 0.0
        # x = fpoint[0]
        y = np.dot(yhat, r[0, :])
        if y < 1e-8:
            y = 0.0
        # y = fpoint[1]
        z = np.dot(zhat, r[0, :])
        if z < 1e-8:
            z = 0.0
        # z = fpoint[2]
        p12 = np.dot(xhat, np.cross(r[0, :], r[1, :]))
        p34 = np.dot(-xhat, np.cross(r[2, :], r[3, :]))
        q12 = np.dot(r[0, :], self._length_vector(1, 2, r))
        q34 = -np.dot(r[3, :], self._length_vector(3, 4, r))
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
        gam = (
            np.arctan2(a1, a2)
            - np.arctan2(b1, b2)
            + np.arctan2(c1, c2)
            - np.arctan2(d1, d2)
        )
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
        eta = d * (
            lambda_12 / np.linalg.norm(self._length_vector(1, 2, r))
            + lambda_34 / np.linalg.norm(self._length_vector(3, 4, r))
        )
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
        coords = np.array([x, y, z])
        return coords, lam, gam, psi, eta, zeta

    def _hxhyhz(self, fpoint):
        h_array = []
        j = self.current
        for k in range(self.n):
            points = self.points[k, :, :]
            nprime = self.angles[k, 0]
            nprime2 = self.angles[k, 1]
            # print("side", k)
            # print("angles", nprime, nprime2)
            d = self._calculate_vector_distance(points[0, :], self.normals[k, :])
            # print("D", D)
            coords, lam, gam, psi, eta, zeta = self._vector_coordinates(points, fpoint)
            # print("coords", coords)
            # print("lambda", lam)
            # print("gamma", gam)
            # print("eta", eta)
            # print("zeta", zeta)
            hx = h_field_x(j, nprime, nprime2, d, coords, lam, gam, psi)
            hy = h_field_y(j, nprime, nprime2, d, coords, lam, gam, psi)
            hz = h_field_z(j, nprime, d, eta, zeta)
            h_array.append(np.hstack([hx, hy, hz]))

        for k in range(2):
            points = self.facepoints[k, :, :]
            nprime = self.angles[k + self.n, 0]
            if self.n > 4:
                shapes = self._face_cutting(points)
                for i in range(self.n - 4):
                    shape = shapes[i, :, :]
                    dist = []
                    for p in shape:
                        dist += [self._calculate_vector_distance(p, self.normals[k, :])]
                    d = np.max(dist)

                    nprime2 = 0
                    coords, lam, gam, psi, eta, zeta = self._vector_coordinates(
                        points, fpoint
                    )
                    hx = h_field_x(j, nprime, nprime2, d, coords, lam, gam, psi)
                    hy = h_field_y(j, nprime, nprime2, d, coords, lam, gam, psi)
                    hz = h_field_z(j, nprime, d, eta, zeta)
                    h_array.append(np.hstack([hx, hy, hz]))

            else:
                dist = []
                for p in points:
                    dist += [self._calculate_vector_distance(p, self.normals[k, :])]
                d = np.max(dist)
                nprime2 = 0
                coords, lam, gam, psi, eta, zeta = self._vector_coordinates(
                    points, fpoint
                )
                hx = h_field_x(j, nprime, nprime2, d, coords, lam, gam, psi)
                hy = h_field_y(j, nprime, nprime2, d, coords, lam, gam, psi)
                hz = h_field_z(j, nprime, d, eta, zeta)
                h_array.append(np.hstack([hx, hy, hz]))
        # print(H_array)

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

    def _hxhyhz2(self, point):
        """
        Calculate the magnetic field strength at a point in local coordinates.
        """
        h_array = []
        r_matrices = []
        t_matrices = []
        for k in range(self.n):
            # rotation matrix
            rot = self._rotational_matrix(k)
            r_matrices += [rot]
            points = self.points[k, :, :]
            # rotate side to working location
            p = np.matmul(points, rot)
            # translate side to origin
            p_t = -p[0, :]
            tmat = self._translation_matrix(p_t)
            t_matrices += [tmat]
            p = np.append(p, np.ones((self.n + 1, 1)), axis=1)
            p = np.matmul(p, tmat)[:, :3]

            # rotate fpoint to working location and translate to origin
            fpoint = np.matmul(point, rot)
            fpoint = np.append(fpoint, 1)
            fpoint = np.matmul(fpoint, tmat)[:3]
            nprime = self.angles[k, 0]
            nprime2 = self.angles[k, 1]
            j = self.current
            d = self._calculate_vector_distance(p[0, :], self.normals[k, :])
            fpoint, lam, gam, psi, eta, zeta = self._working_coordinates(p, fpoint)
            hx = h_field_x(j, nprime, nprime2, d, fpoint, lam, gam, psi)
            hy = h_field_y(j, nprime, nprime2, d, fpoint, lam, gam, psi)
            hz = h_field_z(j, nprime, d, eta, zeta)
            h_array.append(np.hstack([hx, hy, hz]))

        for k in range(2):
            """
            for faces need to seperate into triangles and trapezoids
            before distance D and working coords can be calculated
            """
            # rotation matrix
            rot = self._rotational_matrix(k)
            r_matrices += [rot]
            points = self.facepoints[k, :, :]
            # rotate side to working location
            p = np.matmul(points, rot)
            # translate side to origin
            p_t = -p[0, :]
            tmat = self._translation_matrix(p_t)
            t_matrices += [tmat]
            p = np.append(p, np.ones((self.n + 1, 1)), axis=1)
            p = np.matmul(p, tmat)[:, :3]
            # rotate fpoint to working location and translate to origin
            fpoint = np.matmul(point, rot)
            fpoint = np.append(fpoint, 1)
            fpoint = np.matmul(fpoint, tmat)[:3]
            nprime = self.angles[k, 0]
            nprime2 = self.angles[k, 1]
            j = self.current

            if self.n > 4:
                shapes = self._face_cutting(p)
                d = []
                for i in range(self.n - 4):
                    shape = shapes[i, :, :]
                    dist = []
                    for p in shape:
                        dist += [self._calculate_vector_distance(p, self.normals[k, :])]
                    dist = np.max(d)

                    d += [dist]
                    nprime2 = 0
                    fpoint, lam, gam, psi, eta, zeta = self._working_coordinates(
                        p, fpoint
                    )
                    hx = h_field_x(j, nprime, nprime2, d, fpoint, lam, gam, psi)
                    hy = h_field_y(j, nprime, nprime2, d, fpoint, lam, gam, psi)
                    hz = h_field_z(j, nprime, d, eta, zeta)
                    h_array.append(np.hstack([hx, hy, hz]))

                else:
                    shapes = p
                    for p in shapes:
                        dist += [self._calculate_vector_distance(p, self.normals[k, :])]
                    dist = np.max(d)
                    nprime2 = 0
                    fpoint, lam, gam, psi, eta, zeta = self._working_coordinates(
                        p, fpoint
                    )
                    hx = h_field_x(j, nprime, nprime2, d, fpoint, lam, gam, psi)
                    hy = h_field_y(j, nprime, nprime2, d, fpoint, lam, gam, psi)
                    hz = h_field_z(j, nprime, d, eta, zeta)
                    h_array.append(np.hstack([hx, hy, hz]))

        hx = 0
        hy = 0
        hz = 0
        counter = 0
        for h in h_array:
            h = np.append(h, 1)
            h = np.matmul(h, np.linalg.inv(t_matrices[counter]))[:3]
            h = np.matmul(h, np.transpose(r_matrices[counter]))
            counter += 1
            hx += h[0]
            hy += h[1]
            hz += h[2]

        h = [hx, hy, hz]
        h = h + self._calculate_mc(point)
        return h

    def _bxbybz(self, point):
        """
        Calculate the field at a point in local coordinates.
        """
        h = self._hxhyhz(point)
        b = (h + self._calculate_mc(point)) * MU_0
        return b

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
        # point = self._global_to_local([point])[0]
        b = self._bxbybz(point)
        return b