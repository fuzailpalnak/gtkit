from itertools import tee
from typing import Union, Tuple, List, Dict

import numpy as np
import rasterio
from affine import Affine
from rasterio.warp import transform_bounds
from shapely import wkb
from shapely.geometry import LineString, Point, MultiLineString, mapping, Polygon


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def cut(line: LineString, dist: float) -> Tuple[LineString, LineString]:
    """
    Split a LineString at a specified distance.

    Args:
        line (LineString): The LineString to be split.
        dist (float): The distance at which to split the LineString.

    Returns:
        Tuple[LineString, LineString]: Two LineString segments resulting from the split.
    """
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
    """
    Calculate the fraction of the given Point's position along the LineString and the projected Point on the LineString.

    Args:
        line (Union[LineString, MultiLineString]): The LineString or MultiLineString geometry.
        point (Point): The Point to be projected onto the LineString.

    Returns:
        Tuple[Union[int, float], Point]: The fraction of the Point's position and the projected Point on the LineString.
    """

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


def get_reference_shift(
    center_line_points: List[Point], translated_line: LineString
) -> List[Point]:
    """
    Get the projected points on a translated LineString from a list of center LineString points.

    Args:
        center_line_points (List[Point]): List of center LineString points.
        translated_line (LineString): The translated LineString.

    Returns:
        List[Point]: List of projected points on the translated LineString.
    """

    proj_pts = list()
    for pts in center_line_points:
        _, pt1 = line_referencing(line=translated_line, point=pts)
        proj_pts.append(pt1)
    return proj_pts


def interpolate_with_delta(
    line: LineString, delta: float, distances: List[float] = None
) -> List[Point]:
    """
    Interpolate points along a LineString at specified intervals or distances.

    Args:
        line (LineString): The LineString to interpolate points along.
        delta (float): The interval or distance between interpolated points.
        distances (List[float], optional): List of specific distances for interpolation. Defaults to None.

    Returns:
        List[Point]: List of interpolated points along the LineString.
    """

    distances = (
        np.arange(0, round(line.length, 1), delta) if distances is None else distances
    )
    geom = [line.interpolate(distance) for distance in distances]
    boundary = [line.boundary[1]]
    # if len(geom) < len(distances) else geom
    return geom + boundary


def get_pts_on_orthogonal_line_and_orthogonal_line(
    pt1: Point, pt2: Point, width: float
) -> LineString:
    """
    Get a LineString representing points on an orthogonal line offset by a given road width.

    Args:
        pt1 (Point): The first Point to define the orthogonal line.
        pt2 (Point): The second Point to define the orthogonal line.
        width (float): The road width for offset.

    Returns:
        LineString: A LineString representing points on an orthogonal line offset by the road width.
    """

    ab = LineString([pt1, pt2])
    return LineString(
        [
            ab.parallel_offset(width / 2, "left").boundary[1],
            ab.parallel_offset(width / 2, "right").boundary[0],
        ]
    )


def extrapolate(p1: Point, p2: Point, ratio: int = 15) -> LineString:
    """
    Extrapolate a LineString between two points based on a given ratio.

    Args:
        p1 (Point): The first Point.
        p2 (Point): The second Point.
        ratio (int, optional): The extrapolation ratio. Defaults to 15.

    Returns:
        LineString: The extrapolated LineString.
    """

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


def is_orientation_clockwise(co_ords: np.ndarray) -> bool:
    """
    Check if the orientation of coordinates is clockwise.

    Args:
        co_ords (np.ndarray): Array of coordinates.

    Returns:
        bool: True if the orientation is clockwise, False otherwise.
    """

    # https://github.com/shapely/shapely/blob/main/shapely/algorithms/cga.py
    xs, ys = co_ords[:, 0].tolist(), co_ords[:, 1].tolist()
    xs.append(xs[1])
    ys.append(ys[1])
    return (
        sum(xs[i] * (ys[i + 1] - ys[i - 1]) for i in range(1, len(co_ords))) / 2.0
    ) < 0


def geo_transform_to_26190(
    width: int, height: int, bounds: Tuple[float, float, float, float], crs: Dict
) -> Affine:
    """
    Transform geographic coordinates to EPSG:26910 (NAD83 UTM Zone 10N) coordinates.

    Parameters:
        width (int): Width of the array.
        height (int): Height of the array.
        bounds (tuple): A tuple representing the bounds (xmin, ymin, xmax, ymax).
        crs (dict): Coordinate Reference System of the input coordinates.

    Returns:
        Affine: The affine transformation matrix for the EPSG:26910 coordinates.
    """

    west, south, east, north = transform_bounds(crs, {"init": "epsg:26910"}, *bounds)
    return rasterio.transform.from_bounds(west, south, east, north, width, height)


def re_project_crs_to_26190(
    bounds: Tuple[float, float, float, float], from_crs: Dict
) -> Tuple[float, float, float, float]:
    """
    Reproject bounds from a given CRS to EPSG:26910 (NAD83 UTM Zone 10N).

    Parameters:
        bounds (tuple): A tuple representing the bounds (xmin, ymin, xmax, ymax).
        from_crs (dict): Source Coordinate Reference System.

    Returns:
        tuple: Reprojected bounds in EPSG:26910 coordinates as (west, south, east, north).
    """

    west, south, east, north = transform_bounds(
        from_crs, {"init": "epsg:26910"}, *bounds
    )
    return west, south, east, north


def re_project_from_26190(
    bounds: Tuple[float, float, float, float], to_crs: Dict
) -> Tuple[float, float, float, float]:
    """
    Reproject bounds from EPSG:26910 (NAD83 UTM Zone 10N) to a target CRS.

    Parameters:
        bounds (tuple): A tuple representing the bounds (xmin, ymin, xmax, ymax).
        to_crs (dict): Target Coordinate Reference System.

    Returns:
        tuple: Reprojected bounds in the target CRS as (west, south, east, north).
    """

    west, south, east, north = transform_bounds({"init": "epsg:26910"}, to_crs, *bounds)
    return west, south, east, north


def centroid_with_z(geom: Polygon) -> Point:
    """
    Calculate the centroid of a 3D polygon.

    Args:
        geom (Polygon): The input 3D polygon geometry.

    Returns:
        Point: The 3D centroid point.
    """
    co_ords = mapping(geom)["coordinates"]
    x = np.average(np.array(co_ords)[0][:, 0])
    y = np.average(np.array(co_ords)[0][:, 1])
    z = np.average(np.array(co_ords)[0][:, 2])

    return Point(x, y, z)


def reverse_geom(geom):
    """
    Reverse the coordinates of a geometry.

    Args:
        geom: The input geometry.

    Returns:
        Tuple: The reversed coordinates tuple.
    """

    def _reverse(x, y, z=None):
        if z:
            return x[::-1], y[::-1], z[::-1]
        return x[::-1], y[::-1]


def convert_3d_2d(geometry):
    """
    Convert a 3D geometry to a 2D geometry.

    Args:
        geometry: The input geometry.

    Returns:
        Geometry: The converted 2D geometry.
    """

    return wkb.loads(wkb.dumps(geometry, output_dimension=2))


def midpoint(line: LineString) -> Point:
    """
    Calculate the midpoint of a LineString.

    Args:
        line (LineString): The input LineString geometry.

    Returns:
        Point: The midpoint point of the LineString.
    """
    return line.interpolate(line.length / 2)
