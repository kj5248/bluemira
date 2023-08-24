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

import matplotlib.collections as col
import matplotlib.path as pltpath
import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.constants import MU_0, MU_0_2PI
from bluemira.base.look_and_feel import bluemira_error
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import make_polygon
from bluemira.magnetostatics.baseclass import (
    ArbitraryCrossSectionCurrentSource,
    SourceGroup,
)
from bluemira.magnetostatics.biot_savart import BiotSavartFilament
from bluemira.magnetostatics.tools import process_xyz_array
from bluemira.utilities.plot_tools import Plot3D

__all__ = ["PolyhedralPrismCurrentSource"]


def trap_dist(theta, pos, min_pos, vec):
    """
    func to produce distance between centre and trap end
    """
    dy = np.dot((pos - min_pos), vec)
    dz = dy * np.tan(theta)
    return dz


class amp_wire:
    def __init__(self, origin_l, origin_u, current_direction, current, radius):
        self.lower = origin_l
        self.upper = origin_u
        self.Ihat = current_direction
        self.current = current
        self.radius = radius

    @process_xyz_array
    def field(self, x, y, z):
        point = np.array([x, y, z])
        if z > self.upper[2]:
            f = self.upper
        if z < self.lower[2]:
            f = self.lower
        else:
            f = np.array([self.lower[0], self.lower[1], z])
        dist = math.dist(point, f)
        r_vec = (point - f) / dist
        b_vec = np.cross(self.Ihat, r_vec)
        if (z <= self.upper[2]) and (z >= self.lower[2]) and (dist <= self.radius):
            Bmag = MU_0_2PI * self.current * dist / self.radius**2
        else:
            Bmag = MU_0_2PI * self.current / dist
        B = Bmag * b_vec

        return B[0], B[1], B[2]


class PolyhedralPrismCurrentSource(ArbitraryCrossSectionCurrentSource):
    """
    prism current source
    """

    def __init__(self, origin, ds, normal, t_vec, n, length, width, current, r):
        """
        initialisation
        """
        self.origin = origin
        self.n = n
        self.length = np.linalg.norm(ds)
        self.current = current
        self.dcm = np.array([t_vec, ds / self.length, normal])
        self.length = length
        self.width = width
        self.theta = 2 * np.pi / self.n
        self.trap_vec = np.array([1, 1, 0])
        self.theta_l = np.pi / 4
        self.theta_u = np.pi / 4
        self.points = self._calc_points()
        self.r = r
        self.area = self._cross_section_area()
        self.filsl, self.filsu = self._filament_setup()
        # self.filament_current = self.current*np.pi*self.r**2/self.area
        self.I_vec = normal

    def _cross_section_area(self):
        """
        Function to calculate cross sectional area of prism
        """

        points = self.points[0]
        wire = make_polygon(points, closed=True)
        face = BluemiraFace(boundary=wire)
        return face.area

    def _calc_points(self):
        """
        Function to calculate all the points of the prism in local coords
        """
        c_points = []
        for i in range(self.n + 1):
            c_points += [
                np.array(
                    [
                        round(self.width * np.sin(i * self.theta), 10),
                        round(self.width * np.cos(i * self.theta), 10),
                        0,
                    ]
                )
            ]
        c_points = np.vstack([np.array(c_points)])

        vals = []
        for p in c_points:
            vals += [np.dot(p, self.trap_vec)]
        self.boundl = c_points[vals.index(min(vals)), :]
        self.boundu = c_points[vals.index(max(vals)), :]

        l_points = []
        u_points = []

        for p in c_points:
            dz_l = trap_dist(self.theta_l, p, self.boundl, self.trap_vec)
            l_points += [
                np.array([p[0], p[1], round(p[2] - 0.5 * self.length - dz_l, 10)])
            ]
            dz_u = trap_dist(self.theta_u, p, self.boundl, self.trap_vec)
            u_points += [
                np.array([p[0], p[1], round(p[2] + 0.5 * self.length + dz_u, 10)])
            ]
        l_points = np.vstack([np.array(l_points)])
        u_points = np.vstack([np.array(u_points)])
        points = [c_points] + [l_points] + [u_points]
        # add lines between cuts
        for i in range(self.n):
            points += [np.vstack([l_points[i], u_points[i]])]
        p_array = []
        for p in points:
            p_array.append(self._local_to_global(p))

        return np.array(p_array, dtype=object)

    def _filament_setup(self):
        """
        Function to provide the placement of the filaments for magnetostatic calculations
        r is the radius of the filaments
        s is the 2d shape filaments fit into (3d coords)
        Method centres the filament on the prism edge rather than having edge of filament align with
        prism edge
        """
        # nrows = 1 + (np.dot(self.boundu - self.boundl, self.trap_vec) - r) / r
        s = self.points[0]
        pmin = np.array([np.min(s[:, 0]), np.min(s[:, 1])])
        pmax = np.array([np.max(s[:, 0]), np.max(s[:, 1])])
        grid = []
        f_grid = []
        nx = int((pmax[0] - pmin[0]) / (2 * self.r))
        ny = int((pmax[1] - pmin[1]) / (2 * self.r))
        x = np.linspace(pmin[0], pmax[0], nx)
        y = np.linspace(pmin[1], pmax[1], ny)
        # setup of filament array
        for i in x:
            for j in y:
                p = np.array([i, j])
                grid += [np.array([i, j])]
                if self._inside_2d(s, p, self._normals(s)) == True:
                    f_grid += [np.array([i, j])]

        fil_arr1 = []
        fil_arr2 = []
        # adding z lower and upper to array
        for xy in f_grid:
            p = np.array([xy[0], xy[1], s[0, 2]])
            zl = (
                p[2]
                - 0.5 * self.length
                - trap_dist(self.theta_l, p, self.boundl, self.trap_vec)
            )
            zu = (
                p[2]
                + 0.5 * self.length
                + trap_dist(self.theta_u, p, self.boundl, self.trap_vec)
            )
            arr1 = np.array([p[0], p[1], zl])
            arr2 = np.array([p[0], p[1], zu])
            fil_arr1 += [arr1]
            fil_arr2 += [arr2]
        fil_arr1 = np.vstack([np.array(fil_arr1)])
        fil_arr2 = np.vstack([np.array(fil_arr2)])
        return fil_arr1, fil_arr2

    def _normals(self, s):
        """
        Function to create normals for shape s in direction of origin
        """
        normals = []
        for i in range(len(s[:, 0]) - 1):
            p1 = s[i, :]
            p2 = s[i + 1, :]
            p3 = self.origin[:2]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            n1 = np.array([-dy, dx])
            if np.dot(n1, p3) >= 0:
                norm = n1
            else:
                norm = np.array([dy, -dx])
            normals += [norm]
        return normals

    def _inside_2d(self, s, p, n):
        """
        Function to determine if point p is inside shape s
        Shape s will be made of points in 2d coords
        Point p will be a single point in 2d
        Array of normals n for shape s
        Output will be either True if p is in s or False otherwise
        """
        s = s[:, :2]
        inside = True
        while inside == True:
            for i in range(len(s) - 1):
                val = np.dot((p - s[i, :]), -n[i])
                if val >= 0:
                    continue
                else:
                    inside = False
            break
        return inside

    @process_xyz_array
    def field(self, x, y, z):
        point = np.array([x, y, z])
        fl = self.filsl
        fu = self.filsu
        self.n_fil = np.shape(fl)[0]
        sources = SourceGroup([])
        area = np.pi * self.r**2
        self.N = self.area / area
        # current = self.current*np.pi*self.r**2/self.area
        # current = self.current/N
        for f1, f2 in zip(fl, fu):
            x = f1[0] * np.ones(2)
            y = f1[1] * np.ones(2)
            z = np.linspace(f1[2], f2[2], 2)
            source = BiotSavartFilament(
                np.array([x, y, z]).T, radius=self.r, current=self.filament_current
            )
            sources.add_to_group([source])
        Bx, By, Bz = sources.field(*point)
        # B = np.sqrt(Bx**2 + By**2 + Bz**2)
        # B *= 4/np.pi # factor for circle area out of square area
        return Bx, By, Bz

    @process_xyz_array
    def field2(self, x, y, z):
        point = np.array([x, y, z])
        filsl = self.filsl
        filsu = self.filsu
        self.n_fil = np.shape(filsl)[0]
        self.filament_current = self.current / self.n_fil
        area = np.pi * self.r**2
        self.N = self.area / area
        B = np.array([0.0, 0.0, 0.0])
        for fl, fu in zip(filsl, filsu):
            if z > fu[2]:
                f = fu
            if z < fl[2]:
                f = fl
            else:
                f = np.array([fl[0], fl[1], z])
            dist = math.dist(point, f)
            r_vec = (point - f) / dist
            # print("point", point)
            # print("f", f)
            # print("rvec", r_vec)

            b_vec = np.cross(self.I_vec, r_vec)
            # print("bvec", b_vec)
            # norm = math.dist(b_vec)
            # print("norm", norm)
            if (z <= fu[2]) and (z >= fl[2]) and (dist <= self.r):
                Bmag = MU_0_2PI * self.filament_current * dist / self.r**2
            elif z > fu[2]:
                Bmag = 0
            elif z < fl[2]:
                Bmag = 0
            else:
                Bmag = MU_0_2PI * self.filament_current / dist
            # print("bmag", Bmag)
            # print("B", Bmag*b_vec)
            B += Bmag * b_vec
        # print("Bfinal", B)

        return B[0], B[1], B[2]

    def filament_plot(self):
        fig, ax = plt.subplots()
        s = self.points[0]
        ax.plot(s[:, 0], s[:, 1], "r-")
        fils = self.filsl
        print("Number of filaments =", np.shape(fils)[0])
        xs = fils[:, 0]
        ys = fils[:, 1]
        circles = [
            plt.Circle((xi, yi), radius=self.r, linewidth=1) for xi, yi in zip(xs, ys)
        ]
        c = col.PatchCollection(circles)
        ax.add_collection(c)
        # plt.plot(xs,ys,'ro')
        plt.show()
