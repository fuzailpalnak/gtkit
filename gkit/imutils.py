from typing import Tuple

import affine
import rasterio
from rasterio.transform import rowcol


def get_window(
    extent: Tuple[float, float, float, float], transform: affine.Affine
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


def compute_bounds(
    width: int, height: int, transform: affine.Affine
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


def get_affine_transform(
    min_x: float, max_y: float, pixel_width: float, pixel_height: float
) -> affine.Affine:
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

    return affine.Affine.translation(min_x, max_y) * affine.Affine.scale(
        pixel_width, -pixel_height
    )


def get_pixel_resolution(transform: affine.Affine) -> Tuple[float, float]:
    """
    Get the pixel resolution from an affine transformation matrix.

    Parameters:
        transform (Affine): An affine transformation matrix.

    Returns:
        tuple: The pixel resolution as (pixel_width, pixel_height).
    """

    return transform[0], -transform[4]
