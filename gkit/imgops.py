from dataclasses import dataclass
from typing import Union, Dict, Generator, Tuple, List

import affine
import cv2
import numpy as np
import rasterio
from osgeo import gdal, osr
from rasterio.features import shapes
from rasterio.io import BufferedDatasetWriter, DatasetWriter
from rasterio.transform import rowcol
from shapely.geometry import shape

from gkit.mesh import create_mesh_using_img_param


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

    return affine.Affine.translation(min_x, max_y) * Affine.scale(
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


@dataclass
class Bitmap:
    """
    Represents an image bitmap with its associated transformation.

    Attributes:
        array (np.ndarray): The image data as a NumPy array.
        transform (affine.Affine): The affine transformation matrix.
        crs (str)
    """

    array: np.ndarray
    transform: affine.Affine
    crs: str


def bitmap_gen(
    bounds: Tuple[float, float, float, float],
    output_image_size: Tuple[int, int],
    pixel_resolution: Tuple[float, float],
    geometry_collection: List,
    crs: str,
    allow_output_to_overlap: bool = True,
) -> Generator[Bitmap, None, None]:
    """
    Generate bitmaps within specified bounds using a mesh grid.

    Parameters:
        bounds (Tuple[float, float, float, float]): The bounds (xmin, ymin, xmax, ymax).
        output_image_size (Tuple[int, int]): Output image size (width, height).
        pixel_resolution (Tuple[float, float]): Pixel resolution (pixel_width, pixel_height).
        geometry_collection (List): List of geometries to include in the bitmaps.
        crs (str)
        allow_output_to_overlap (bool): Flag indicating if output can overlap (default True).

    Yields:
        Generator[Bitmap, None, None]: A generator yielding Bitmap instances.
    """
    mesh = create_mesh_using_img_param(
        bounds,
        output_image_size,
        pixel_resolution,
        is_overlap=allow_output_to_overlap,
    )

    intermediate_bitmap = np.empty((0,))
    intermediate_bitmap = np.concatenate(
        (intermediate_bitmap, geometry_collection), axis=0
    )

    for grid in mesh.mesh():
        transform = get_affine_transform(
            grid["extent"][0],
            grid["extent"][-1],
            *get_pixel_resolution(mesh.mesh_transform),
        )
        bitmap_array = rasterio.features.rasterize(
            ((g, 255) for g in intermediate_bitmap),
            out_shape=mesh.grid_size,
            transform=transform,
        )
        yield Bitmap(bitmap_array, transform, crs)


def geomultiband(
    image: np.ndarray,
    transform: affine.Affine,
    gdal_unit,
    epsg: int,
    output_file_name: str,
):
    """
    Save a multi-band image with 16-bit data to a GeoTIFF file.

    Parameters:
        image (np.ndarray): The image data as a NumPy array.
        geo_transform (affine.Affine): The affine transformation matrix.
        epsg (int): The EPSG code for the coordinate reference system.
        output_file_name (str): The output GeoTIFF file name.
    """
    assert image.ndim == 3, f"Expected to have 3 dim got {image.ndim}"

    x, y, z = image.shape
    dst_ds = gdal.GetDriverByName("GTiff").Create(output_file_name, y, x, z, gdal_unit)

    dst_ds.SetGeoTransform(transform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    dst_ds.SetProjection(srs.ExportToWkt())

    for idx in range(0, z):
        dst_ds.GetRasterBand(idx + 1).WriteArray(image[:, :, idx])

    dst_ds.FlushCache()


def geowrite(
    save_path: str, image: np.ndarray, transform: affine.Affine, crs: str = None
):
    """
    Write a GeoTIFF file using the rasterio library.

    Parameters:
        save_path (str): The path to save the GeoTIFF file.
        image (np.ndarray): The image data as a NumPy array.
        transform (affine.Affine): The affine transformation matrix.
        crs (str)
    """

    bands = 1 if image.ndim == 2 else image.shape[-1]
    with rasterio.open(
        save_path,
        "w",
        driver="GTiff",
        dtype=rasterio.uint8,
        count=bands,
        width=image.shape[0],
        height=image.shape[1],
        transform=transform,
        crs=crs,
    ) as dst:
        dst.write(image, indexes=1)


def georead(
    path: str,
) -> Union[BufferedDatasetWriter, DatasetWriter]:
    """
    Read a GeoTIFF file using the rasterio library.

    Parameters:
        path (str): The path to the GeoTIFF file.

    Returns:
        Union[BufferedDatasetWriter, DatasetWriter]: The opened rasterio dataset.
    """

    return rasterio.open(path, "r+")


def shp_gen(
    image: np.ndarray, transform: affine.Affine, is_shape: bool = False, **kwargs
) -> Generator[Dict, None, None]:
    """
    Generate shape data from an image using rasterio's shapes function.

    Parameters:
        image (np.ndarray): The image data as a NumPy array.
        transform (affine.Affine): The affine transformation matrix.
        is_shape (bool): Flag indicating if the geometry should be represented as Shapely shapes (default False).
        **kwargs: Additional keyword arguments to be included as properties.

    Yields:
        Generator[Dict, None, None]: A generator yielding dictionaries with geometry and properties.
    """
    for i, (s, v) in enumerate(
        shapes(
            image.astype(rasterio.uint8),
            mask=(image > 0),
            connectivity=8,
            transform=transform,
        )
    ):
        yield {
            "geometry": shape(s) if is_shape else s,
            "properties": {"id": v, **kwargs},
        }


def obj_to_shp_gen(
    image: Union[BufferedDatasetWriter, DatasetWriter], **kwargs
) -> Generator[Dict, None, None]:
    """
    Convert a rasterio dataset to a shape generator.

    Parameters:
        image (Union[BufferedDatasetWriter, DatasetWriter]): A rasterio dataset.
        **kwargs: Additional keyword arguments to be included as properties.

    Yields:
        Generator[Dict, None, None]: A generator yielding dictionaries with geometry and properties.
    """
    return shp_gen(
        image.read(), image.transform, is_shape=True, crs=image.crs, **kwargs
    )


def path_to_shp_gen(image_path: str, **kwargs) -> Generator[Dict, None, None]:
    """
    Convert an image file to a shape generator.

    Parameters:
        image_path (str): Path to the image file.
        **kwargs: Additional keyword arguments to be included as properties.

    Yields:
        Generator[Dict, None, None]: A generator yielding dictionaries with geometry and properties.
    """
    return obj_to_shp_gen(georead(image_path), **kwargs)


def copy_reference_from_obj(
    copy_from: Union[BufferedDatasetWriter, DatasetWriter],
    copy_to: np.ndarray,
    save_to: str,
):
    """
    Copy reference information from one rasterio dataset to another.

    Parameters:
        copy_from (Union[BufferedDatasetWriter, DatasetWriter]): Source rasterio dataset.
        copy_to (np.ndarray): The target image data as a NumPy array.
        save_to (str): Path to save the output rasterio dataset.
    """

    bands = copy_to.ndim if copy_to.ndim > 2 else 1
    geo_referenced_image = rasterio.open(
        save_to,
        mode="w",
        driver=copy_from.driver,
        width=copy_from.width,
        height=copy_from.height,
        crs=copy_from.crs,
        transform=copy_from.transform,
        dtype=copy_to.dtype,
        count=bands,
    )
    if bands > 2:
        for band in range(copy_to.shape[2]):
            geo_referenced_image.write(copy_to[:, :, band], band + 1)
    else:
        geo_referenced_image.write(copy_to, 1)

    copy_from.close()
    geo_referenced_image.close()


def copy_reference_from_pth(
    copy_from_path: str, copy_to_path: str, save_to: str, is_binary: bool = True
):
    """
    Copy reference information from one GeoTIFF to another.

    Parameters:
        copy_from_path (str): Path to the source GeoTIFF file.
        copy_to_path (str): Path to the target GeoTIFF file.
        save_to (str): Path to save the output GeoTIFF file.
        is_binary (bool): Flag indicating if the image is binary (default True).
    """

    destination_img = (
        cv2.imread(copy_to_path, cv2.IMREAD_GRAYSCALE)
        if is_binary
        else cv2.cvtColor(cv2.imread(copy_to_path), cv2.COLOR_BGR2RGB)
    )

    copy_reference_from_obj(
        georead(copy_from_path),
        destination_img,
        save_to,
    )


def save_8bit_multi_band(
    image: np.ndarray, geo_transform: affine.Affine, epsg: int, output_file_name: str
):
    """
    Save a multi-band image with 8-bit data to a GeoTIFF file.

    Parameters:
        image (np.ndarray): The image data as a NumPy array.
        geo_transform (affine.Affine): The affine transformation matrix.
        epsg (int): The EPSG code for the coordinate reference system.
        output_file_name (str): The output GeoTIFF file name.
    """

    geomultiband(image, geo_transform, gdal.GDT_Byte, epsg, output_file_name)


def save_16bit_multi_band(
    image: np.ndarray, geo_transform: affine.Affine, epsg: int, output_file_name: str
):
    geomultiband(image, geo_transform, gdal.GDT_UInt16, epsg, output_file_name)
