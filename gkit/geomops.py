from typing import Union, Tuple, List

import numpy as np
from shapely.geometry import LineString, Point, MultiLineString, mapping


def cut(line: LineString, dist):
    assert (
        0.0 < dist < line.length
    ), f"Given {dist} > {line.length}, Expected the distance of split to be less than than line length"

    coords = list(line.coords)
    for i, p in enumerate(coords):
        pd = line.project(Point(p))
        if pd == dist:
            return LineString(coords[: i + 1]), LineString(coords[i:])

        if pd > dist:
            cp = line.interpolate(dist)
            cp_cords = (cp.x, cp.y, cp.z) if cp.has_z else (cp.x, cp.y)

            return (
                LineString(coords[:i] + [cp_cords]),
                LineString([cp_cords] + coords[i:]),
            )


def line_referencing(
    line: Union[LineString, MultiLineString], point: Point
) -> Tuple[Union[int, float], Point]:
    # https://stackoverflow.com/questions/24415806/coordinates-of-the-closest-points-of-two-geometries-in-shapely

    assert type(line) in [LineString, MultiLineString], (
        "Expected type of 'line' to be in ['LineString', 'MultiLineString']" "got %s",
        (type(line),),
    )
    assert type(point) == Point, (
        "Expected type of 'point' to be 'Point'" "got %s",
        (type(point),),
    )
    fraction = line.project(point, normalized=True)
    project_point = line.interpolate(fraction, normalized=True)
    return fraction, project_point


def get_reference_shift(center_line_points: List[Point], translated_line: LineString):
    proj_pts = list()
    for pts in center_line_points:
        _, pt1 = line_referencing(line=translated_line, point=pts)
        proj_pts.append(pt1)
    return proj_pts


def interpolate_with_delta(line: LineString, delta, distances=None) -> List:
    distances = (
        np.arange(0, round(line.length, 1), delta) if distances is None else distances
    )
    geom = [line.interpolate(distance) for distance in distances]
    boundary = [line.boundary[1]]
    # if len(geom) < len(distances) else geom
    return geom + boundary


def get_pts_on_orthogonal_line_and_orthogonal_line(
    pt1: Point, pt2: Point, road_width: float
) -> LineString:
    ab = LineString([pt1, pt2])
    return LineString(
        [
            ab.parallel_offset(road_width / 2, "left").boundary[1],
            ab.parallel_offset(road_width / 2, "right").boundary[0],
        ]
    )


def extrapolate(p1: Point, p2: Point, ratio=15):
    p1 = mapping(p1)["coordinates"]
    p2 = mapping(p2)["coordinates"]

    a = p1
    b = (
        (
            p1[0] + ratio * (p2[0] - p1[0]),
            p1[1] + ratio * (p2[1] - p1[1]),
            p1[2] + ratio * (p2[2] - p1[2]),
        )
        if (p1.has_z() and p2.has_z())
        else (
            p1[0] + ratio * (p2[0] - p1[0]),
            p1[1] + ratio * (p2[1] - p1[1]),
        )
    )
    return LineString([a, b])


def is_orientation_clockwise(co_ords: np.ndarray):
    # https://github.com/shapely/shapely/blob/main/shapely/algorithms/cga.py
    xs, ys = co_ords[:, 0].tolist(), co_ords[:, 1].tolist()
    xs.append(xs[1])
    ys.append(ys[1])
    return (
        sum(xs[i] * (ys[i + 1] - ys[i - 1]) for i in range(1, len(co_ords))) / 2.0
    ) < 0
