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
A collection of tools used in the EU-DEMO design.
"""

import copy
from typing import List, Tuple, Union

import numpy as np

import bluemira.base.components as bm_comp
import bluemira.geometry as bm_geo
from bluemira.base.components import PhysicalComponent
from bluemira.base.constants import EPS
from bluemira.base.error import BuilderError
from bluemira.builders._varied_offset import varied_offset
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.plane import BluemiraPlane
from bluemira.geometry.tools import (
    boolean_cut,
    boolean_fuse,
    circular_pattern,
    extrude_shape,
    make_circle,
    make_polygon,
    revolve_shape,
    slice_shape,
    sweep_shape,
)

__all__ = [
    "get_n_sectors",
    "circular_pattern_component",
    "pattern_revolved_silhouette",
    "pattern_lofted_silhouette",
    "varied_offset",
    "find_xy_plane_radii",
    "make_circular_xy_ring",
    "build_sectioned_xy",
    "build_sectioned_xyz",
]


def get_n_sectors(no_obj, degree=360):
    """
    Get sector count and angle size for a given number of degrees of the reactor.

    Parameters
    ----------
    no_obj: int
        total number of components (eg TF coils)
    degree: float
        angle to view of reactor

    Returns
    -------
    sector_degree: float
        number of degrees per sector
    n_sectors: int
        number of sectors
    """
    sector_degree = 360 / no_obj
    n_sectors = max(1, int(degree // int(sector_degree)))
    return sector_degree, n_sectors


def circular_pattern_component(
    component: Union[bm_comp.Component, List[bm_comp.Component]],
    n_children: int,
    parent_prefix: str = "Sector",
    *,
    origin=(0.0, 0.0, 0.0),
    direction=(0.0, 0.0, 1.0),
    degree=360.0,
):
    """
    Pattern the provided Component equally spaced around a circle n_children times.

    The resulting components are assigned to a set of common parent Components having
    a name with the structure "{parent_prefix} {idx}", where idx runs from 1 to
    n_children. The Components produced under each parent are named according to the
    original Component with the corresponding idx value appended.

    Parameters
    ----------
    component: Union[Component, List[Component]],
        The original Component to use as the template for copying around the circle.
    n_children: int
        The number of children to produce around the circle.
    parent_prefix: str
        The prefix to provide to the new parent component, having a name of the form
        "{parent_prefix} {idx}", by default "Sector".
    origin: Tuple[float, float, float]
        The origin of the circle to pattern around, by default (0., 0., 0.).
    direction: Tuple[float, float, float]
        The surface normal of the circle to pattern around, by default (0., 0., 1.) i.e.
        the positive z axis, resulting in a counter clockwise circle in the x-y plane.
    degree: float
        The angular extent of the patterning in degrees, by default 360.
    """
    sectors = [
        bm_comp.Component(f"{parent_prefix} {idx+1}") for idx in range(n_children)
    ]

    def assign_component_to_sector(
        comp: bm_comp.Component,
        sector: bm_comp.Component,
        shape: bm_geo.base.BluemiraGeo = None,
    ):
        idx = int(sector.name.replace(f"{parent_prefix} ", ""))

        if shape is not None and not shape.label:
            shape.label = f"{comp.name} {idx}"

        comp = copy.deepcopy(comp)
        comp.name = f"{comp.name} {idx}"

        comp.children = []
        orig_parent: bm_comp.Component = comp.parent
        if orig_parent is not None:
            comp.parent = sector.get_component(f"{orig_parent.name} {idx}")
        if comp.parent is None:
            comp.parent = sector

        if isinstance(comp, bm_comp.PhysicalComponent):
            comp.shape = shape

    def assign_or_pattern(comps: Union[bm_comp.Component, List[bm_comp.Component]]):
        if not isinstance(comps, list):
            comps = [comps]

        for comp in comps:
            if isinstance(comp, bm_comp.PhysicalComponent):
                shapes = bm_geo.tools.circular_pattern(
                    comp.shape,
                    n_shapes=n_children,
                    origin=origin,
                    direction=direction,
                    degree=degree,
                )
                for sector, shape in zip(sectors, shapes):
                    assign_component_to_sector(comp, sector, shape)
            else:
                for sector in sectors:
                    assign_component_to_sector(comp, sector)

    def process_children(comps: Union[bm_comp.Component, List[bm_comp.Component]]):
        if not isinstance(comps, list):
            comps = [comps]
        for comp in comps:
            for child in comp.children:
                assign_or_pattern(child)
                process_children(child)

    assign_or_pattern(component)
    process_children(component)

    return sectors


def pattern_revolved_silhouette(face, n_seg_p_sector, n_sectors, gap):
    """
    Pattern a silhouette with revolutions about the z-axis, inter-spaced with parallel
    gaps between solids.

    Parameters
    ----------
    face: BluemiraFace
        x-z silhouette of the geometry to revolve and pattern
    n_seg_p_sector: int
        Number of segments per sector
    n_sectors: int
        Number of sectors
    gap: float
        Absolute distance between segments (parallel)

    Returns
    -------
    shapes: List[BluemiraSolid]
        List of solids for each segment (ordered anti-clockwise)
    """
    sector_degree = 360 / n_sectors

    if gap <= 0.0:
        # No gaps; just touching solids
        segment_degree = sector_degree / n_seg_p_sector
        shape = revolve_shape(
            face, base=(0, 0, 0), direction=(0, 0, 1), degree=segment_degree
        )
        shapes = circular_pattern(
            shape, origin=(0, 0, 0), degree=sector_degree, n_shapes=n_seg_p_sector
        )
    else:
        volume = revolve_shape(
            face, base=(0, 0, 0), direction=(0, 0, 1), degree=sector_degree
        )
        gaps = _generate_gap_volumes(face, n_seg_p_sector, n_sectors, gap)
        shapes = boolean_cut(volume, gaps)
    return _order_shapes_anticlockwise(shapes)


def pattern_lofted_silhouette(face, n_seg_p_sector, n_sectors, gap):
    """
    Pattern a silhouette with lofts about the z-axis, inter-spaced with parallel
    gaps between solids.

    Parameters
    ----------
    face: BluemiraFace
        x-z silhouette of the geometry to loft and pattern
    n_seg_p_sector: int
        Number of segments per sector
    n_sectors: int
        Number of sectors
    gap: float
        Absolute distance between segments (parallel)

    Returns
    -------
    shapes: List[BluemiraSolid]
        List of solids for each segment (ordered anti-clockwise)
    """
    sector_degree = 360 / n_sectors

    degree = sector_degree * (1 + 1 / n_seg_p_sector)
    faces = circular_pattern(
        face,
        origin=(0, 0, 0),
        direction=(0, 0, 1),
        degree=degree,
        n_shapes=n_seg_p_sector + 1,
    )
    shapes = []
    for i, r_face in enumerate(faces[:-1]):
        com_1 = r_face.center_of_mass
        com_2 = faces[i + 1].center_of_mass

        wire = make_polygon(
            {
                "x": [com_1[0], com_2[0]],
                "y": [com_1[1], com_2[1]],
                "z": [com_1[2], com_2[2]],
            }
        )
        volume = sweep_shape([r_face.boundary[0], faces[i + 1].boundary[0]], wire)
        shapes.append(volume)

    if gap > 0.0:
        if len(shapes) > 1:
            full_volume = boolean_fuse(shapes)
        else:
            full_volume = shapes[0]

        gaps = _generate_gap_volumes(face, n_seg_p_sector, n_sectors, gap)
        shapes = boolean_cut(full_volume, gaps)

    return _order_shapes_anticlockwise(shapes)


def _generate_gap_volumes(face, n_seg_p_sector, n_sectors, gap):
    """
    Generate the gap volumes
    """
    bb = face.bounding_box
    delta = 1.0
    x = np.array(
        [bb.x_min - delta, bb.x_max + delta, bb.x_max + delta, bb.x_min - delta]
    )
    z = np.array(
        [bb.z_min - delta, bb.z_min - delta, bb.z_max + delta, bb.z_max + delta]
    )
    poly = make_polygon({"x": x, "y": 0, "z": z}, closed=True)
    bb_face = BluemiraFace(poly)
    bb_face.translate((0, -0.5 * gap, 0))
    gap_volume = extrude_shape(bb_face, (0, gap, 0))
    degree = 360 / n_sectors
    degree += degree / n_seg_p_sector
    gap_volumes = circular_pattern(
        gap_volume, degree=degree, n_shapes=n_seg_p_sector + 1
    )
    return gap_volumes


def _order_shapes_anticlockwise(shapes):
    """
    Order shapes anti-clockwise about (0, 0, 1) by center of mass
    """
    x, y = np.zeros(len(shapes)), np.zeros(len(shapes))

    for i, shape in enumerate(shapes):
        com = shape.center_of_mass
        x[i] = com[0]
        y[i] = com[1]

    r = np.hypot(x, y)
    angles = np.where(y > 0, np.arccos(x / r), 2 * np.pi - np.arccos(x / r))
    indices = np.argsort(angles)
    return list(np.array(shapes)[indices])


def find_xy_plane_radii(wire, plane):
    """
    Get the radial coordinates of a wire's intersection points with a plane.

    Parameters
    ----------
    wire: BluemiraWire
        Wire to get the radii for in the plane
    plane: BluemiraPlacement
        Plane to slice with

    Returns
    -------
    radii: list
        The array of radii of intersections, sorted from smallest to largest
    """
    intersections = slice_shape(wire, plane)
    return sorted(intersections[:, 0])


def make_circular_xy_ring(r_inner, r_outer):
    """
    Make a circular annulus in the x-y plane (z=0)
    """
    centre = (0, 0, 0)
    axis = (0, 0, 1)
    if np.isclose(r_inner, r_outer, rtol=0, atol=2 * EPS):
        raise BuilderError(f"Cannot make an annulus where r_inner = r_outer = {r_inner}")

    if r_inner > r_outer:
        r_inner, r_outer = r_outer, r_inner

    inner = make_circle(r_inner, center=centre, axis=axis)
    outer = make_circle(r_outer, center=centre, axis=axis)
    return BluemiraFace([outer, inner])


def build_sectioned_xy(
    face: BluemiraFace, plot_colour: Tuple[float]
) -> List[PhysicalComponent]:
    """
    Build the x-y components of sectioned component

    Parameters
    ----------
    face: BluemiraFace
        xz face to build xy component
    plot_colour: Tuple[float]
        colour tuple for component

    """
    xy_plane = BluemiraPlane.from_3_points([0, 0, 0], [1, 0, 0], [1, 1, 0])

    r_ib_out, r_ob_out = find_xy_plane_radii(face.boundary[0], xy_plane)
    r_ib_in, r_ob_in = find_xy_plane_radii(face.boundary[1], xy_plane)

    sections = []
    for name, r_in, r_out in [
        ["inboard", r_ib_in, r_ib_out],
        ["outboard", r_ob_in, r_ob_out],
    ]:
        board = make_circular_xy_ring(r_in, r_out)
        section = PhysicalComponent(name, board)
        section.plot_options.face_options["color"] = plot_colour
        sections.append(section)

    return sections


def build_sectioned_xyz(
    face: BluemiraFace,
    name: str,
    n_TF: int,
    plot_colour: Tuple[float],
    degree: float = 360,
    enable_sectioning: bool = True,
) -> List[PhysicalComponent]:
    """
    Build the x-y-z components of sectioned component

    Parameters
    ----------
    face: BluemiraFace
        xz face to build xyz component
    name: str
        PhysicalComponent name
    n_TF: int
        number of TF coils
    plot_colour: Tuple[float]
        colour tuple for component
    degree: float
        angle to sweep through
    enable_sectioning: bool
        Switch on/off sectioning (#1319 Topology issue)

    Notes
    -----
    When `enable_sectioning=False` a list with a single component rotated a maximum
    of 359 degrees will be returned. This is a workaround for two issues
    from the topology naming issue #1319:

        - Some objects fail to be rebuilt when rotated
        - Some objects cant be rotated 360 degrees due to DisjointedFaceError

    """
    sector_degree, n_sectors = get_n_sectors(n_TF, degree)

    shape = revolve_shape(
        face,
        base=(0, 0, 0),
        direction=(0, 0, 1),
        degree=sector_degree if enable_sectioning else min(359, degree),
    )
    body = PhysicalComponent(name, shape)
    body.display_cad_options.color = plot_colour

    # this is currently broken in some situations
    # because of #1319 and related Topological naming issues
    return (
        circular_pattern_component(body, n_sectors, degree=sector_degree * n_sectors)
        if enable_sectioning
        else [body]
    )
