from typing import Tuple, Dict

import numpy as np
import math
import rasterio

from affine import Affine
from rasterio.transform import rowcol
from rasterio.warp import transform_bounds


def get_window(
    extent: Tuple[float, float, float, float], transform: Affine
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Calculate the row and column window indices for the given extent and affine transform.

    Parameters:
        extent (tuple): A tuple representing the extent (xmin, ymin, xmax, ymax) of the window.
        transform (Affine): An affine transformation matrix.

    Returns:
        tuple: A tuple containing the row and column window indices as (row_indices, col_indices).
    """

    row_start, col_start = rowcol(transform, extent[0], extent[-1], op=int)
    row_stop, col_stop = rowcol(transform, extent[2], extent[1], op=int)

    return (row_start, row_stop), (col_start, col_stop)


def get_mesh_transform(width: int, height: int, transform: Affine) -> Affine:
    """
    Generate an affine transformation matrix for a mesh grid within a given extent.

    Parameters:
        width (int): Width of the mesh grid.
        height (int): Height of the mesh grid.
        transform (Affine): An affine transformation matrix representing the extent.

    Returns:
        Affine: The affine transformation matrix for the mesh grid.
    """

    bounds = compute_bounds(width, height, transform)
    mesh_transform = get_affine_transform(
        bounds[0], bounds[-1], *get_pixel_resolution(transform)
    )
    return mesh_transform


def get_affine_transform(
    min_x: float, max_y: float, pixel_width: float, pixel_height: float
) -> Affine:
    """
    Generate an affine transformation matrix based on translation and scaling.

    Parameters:
        min_x (float): Minimum x-coordinate value.
        max_y (float): Maximum y-coordinate value.
        pixel_width (float): Pixel width.
        pixel_height (float): Pixel height.

    Returns:
        Affine: The generated affine transformation matrix.
    """

    return Affine.translation(min_x, max_y) * Affine.scale(pixel_width, -pixel_height)


def compute_bounds(
    width: int, height: int, transform: Affine
) -> Tuple[float, float, float, float]:
    """
    Compute the bounds of an array using its dimensions and affine transformation.

    Parameters:
        width (int): Width of the array.
        height (int): Height of the array.
        transform (Affine): An affine transformation matrix.

    Returns:
        tuple: A tuple containing the computed bounds (xmin, ymin, xmax, ymax).
    """

    bounds = rasterio.transform.array_bounds(height, width, transform)
    return bounds


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


def get_pixel_resolution(transform: Affine) -> Tuple[float, float]:
    """
    Get the pixel resolution from an affine transformation matrix.

    Parameters:
        transform (Affine): An affine transformation matrix.

    Returns:
        tuple: The pixel resolution as (pixel_width, pixel_height).
    """

    return transform[0], -transform[4]


def compute_num_of_col_and_rows(
    grid_size: Tuple[int, int], mesh_size: Tuple[int, int]
) -> Tuple[int, int]:
    """
    Compute the number of columns and rows in a mesh grid based on grid size and mesh size.

    Parameters:
        grid_size (tuple): A tuple representing the grid size (grid_width, grid_height).
        mesh_size (tuple): A tuple representing the mesh size (mesh_width, mesh_height).

    Returns:
        tuple: The number of columns and rows in the mesh grid as (num_col, num_row).
    """

    num_col = int(np.ceil(mesh_size[0] / grid_size[0]))
    num_row = int(np.ceil(mesh_size[1] / grid_size[1]))

    return num_col, num_row


def compute_dimension(
    bounds: Tuple[float, float, float, float], pixel_resolution: Tuple[float, float]
) -> Tuple[int, int]:
    """
    Compute the output dimensions based on bounds and pixel resolution.

    Parameters:
        bounds (tuple): A tuple representing the bounds (xmin, ymin, xmax, ymax).
        pixel_resolution (tuple): Pixel resolution as (pixel_width, pixel_height).

    Returns:
        tuple: The output dimensions as (output_width, output_height).
    """

    output_width = int(math.ceil((bounds[2] - bounds[0]) / pixel_resolution[0]))
    output_height = int(math.ceil((bounds[3] - bounds[1]) / pixel_resolution[1]))
    return output_width, output_height

