from dataclasses import dataclass
from typing import Union, Dict, Generator, Tuple, List

import affine
import cv2
import numpy as np
import rasterio
from image_fragment.fragment import Fragment, ImageFragment
from osgeo import gdal, osr
from rasterio.features import shapes
from rasterio.io import BufferedDatasetWriter, DatasetWriter
from shapely.geometry import shape

from gtkit.imutils import get_pixel_resolution, get_affine_transform
from gtkit.mesh import create_mesh_using_img_param


class StitchNSplit:
    """
    Class for splitting images into smaller fragments.

    Attributes:
    split_size (tuple): Size of the fragments to split the image into.
    img_size (tuple): Size of the original image.
    image_fragment (ImageFragment): Image fragment object for managing fragments.
    """

    def __init__(self, split_size: tuple, img_size: tuple):
        """
        Initialize the Split class.

        Parameters:
        split_size (tuple): Size of the fragments to split the image into.
        img_size (tuple): Size of the original image.

        Raises:
        ValueError: If the split size is greater than the image size.
        """

        if split_size[0] > img_size[0] or split_size[1] > img_size[1]:
            raise ValueError(
                "Size to Split Can't Be Greater than Image, Given {},"
                " Expected <= {}".format(split_size, (img_size[0], img_size[1]))
            )
        self.split_size = split_size
        self.img_size = img_size

        self.image_fragment = ImageFragment.image_fragment_3d(
            fragment_size=self.split_size, org_size=self.img_size
        )

    def __len__(self):
        """
        Get the number of fragments.

        Returns:
        int: Number of fragments.
        """

        return len(self.image_fragment.collection)

    def __getitem__(self, index):
        """
        Get a fragment by index.

        Parameters:
        index (int): Index of the fragment.

        Returns:
        tuple: Index and the corresponding fragment.
        """
        return index, self.image_fragment.collection[index]

    def split(
        self, image: Union[BufferedDatasetWriter, DatasetWriter], fragment: Fragment
    ):
        """
        Split the image using a windowing approach.

        Parameters:
        image (rasterio.io.DatasetReader): Input image dataset reader object.
        fragment (Fragment): Fragment object specifying the region of interest.

        Returns:
        tuple: A tuple containing the extracted image data and additional keyword arguments.
        """
        raise NotImplementedError

    def stitch(self, image: np.ndarray, stitched_image: np.ndarray, fragment: Fragment):
        """
        Stitch an image fragment onto a larger stitched image.

        This method transfers the data from the provided image fragment onto the specified location
        in the larger stitched image.

        Parameters:
        image (np.ndarray): The image fragment data to be stitched onto the larger image.
        stitched_image (np.ndarray): The larger stitched image onto which the fragment will be stitched.
        fragment (Fragment): The fragment specifying the region in the larger image where the fragment will be placed.

        Returns:
        np.ndarray: The stitched image with the fragment transferred onto it.
        """
        raise NotImplementedError


class StitchNSplitGeo(StitchNSplit):
    """
    Subclass of Split specialized for geospatial image splitting.
    """

    def __init__(self, split_size: tuple, img_size: tuple):
        """
        Initialize the SplitGeo class.

        Parameters:
        split_size (tuple): Size of the fragments to split the image into.
        img_size (tuple): Size of the original image.
        """
        super().__init__(split_size, img_size)

    def split(
        self, image: Union[BufferedDatasetWriter, DatasetWriter], fragment: Fragment
    ) -> (np.ndarray, dict):
        """
        Internal method to extract data from a fragment of a geospatial image.

        Parameters:
        image (rasterio.io.DatasetReader): Input image dataset reader object.
        fragment (Fragment): Fragment object specifying the region of interest.

        Returns:
        tuple: A tuple containing the extracted image data and additional keyword arguments.
        """

        split_image = image.read(window=fragment.position)

        kwargs_split_image = image.meta.copy()
        kwargs_split_image.update(
            {
                "height": self.split_size[0],
                "width": self.split_size[1],
                "transform": image.window_transform(fragment.position),
            }
        )

        return split_image.swapaxes(0, 1).swapaxes(1, 2), kwargs_split_image

    def stitch(self, image: np.ndarray, stitched_image: np.ndarray, fragment: Fragment):
        """
        Stitch an image fragment onto a larger stitched image.

        This method transfers the data from the provided image fragment onto the specified location
        in the larger stitched image.

        Parameters:
        image (np.ndarray): The image fragment data to be stitched onto the larger image.
        stitched_image (np.ndarray): The larger stitched image onto which the fragment will be stitched.
        fragment (Fragment): The fragment specifying the region in the larger image where the fragment will be placed.

        Returns:
        np.ndarray: The stitched image with the fragment transferred onto it.
        """
        return fragment.transfer_fragment(
            transfer_from=image, transfer_to=stitched_image
        )


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
    assert image.ndim == 3, f"Input Image must of shape HxWxC"
    bands = image.shape[-1]

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
        dst.write(np.rollaxis(image, axis=2))


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
