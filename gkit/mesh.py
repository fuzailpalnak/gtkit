import numpy as np
import math
import rasterio
import affine

from dataclasses import dataclass

from shapely.geometry import Point, LineString, MultiLineString, Polygon
from shapely.ops import polygonize, linemerge, unary_union

from affine import Affine
from rasterio.transform import rowcol
from rasterio.warp import transform_bounds

from gkit.geomops import get_reference_shift
from typing import Tuple, Dict, Union, Tuple, Generator, List


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


class Mesh:
    def mesh(self) -> Generator[dict, None, None]:

        raise NotImplementedError

    def collate_data(self, **kwargs):
        raise NotImplementedError


class ImageMesh(Mesh):
    def _compute_step(self) -> Tuple[int, int]:
        """
        Compute Step in X and Y direction
        :return:
        """

        raise NotImplementedError

    @staticmethod
    def _step_in_x(
        bound: Tuple[float, float, float, float], normalizer: int = 1
    ) -> int:
        """
        Step Size to take in X
        :param bound:
        :param normalizer: How small the step to take, Larger the value smaller and smaller step it will take
        :return:
        """
        return int(((bound[2] - bound[0]) / normalizer))

    @staticmethod
    def _step_in_y(
        bound: Tuple[float, float, float, float], normalizer: int = 1
    ) -> int:
        """
        Step Size to take in Y
        :param bound:
        :param normalizer: How small the step to take, Larger the value smaller and smaller step it will take
        :return:
        """
        return int(((bound[-1] - bound[1]) / normalizer))

    def mesh(self) -> Generator[dict, None, None]:
        """
        Compute Mesh

        :return:
        """

        raise NotImplementedError

    def collate_data(self, extent: Tuple[float, float, float, float]) -> dict:
        raise NotImplementedError


@dataclass
class ImageNonOverLapMesh(ImageMesh):
    """
    The Class will compute Grid bounded within complete_size to provide non overlapping grid,
    The class will adjust the grid to evenly fit the number of tiles

    Working of this class depends on the geo reference information of the image which acts as the starting point

    The geo reference information to be present in the image is source_min_x, source_max_y and pixel resolution

    Based on the geo reference information present in the image, compute grid of size
    complete_size // int(np.ceil(dst_img_size / src_img_size) over complete_size

    Given an starting image size, final size and its transform this will find all the grid of size
    complete_size // int(np.ceil(dst_img_size / src_img_size) between the given complete size

    The start position of grid and the step size of grid is computed from the transform info provided, usually
    present in geo referenced image

    NOTE - The COORDINATES MUST BE IN `EPSG:26910`
    """

    grid_size: tuple
    mesh_size: tuple
    sections: tuple
    mesh_transform: affine.Affine
    mesh_bound: tuple

    def _compute_step(self) -> Tuple[int, int]:
        """
        Compute Step in X and Y direction
        :return:
        """

        step_in_x = self._step_in_x(self.mesh_bound, self.sections[0])
        step_in_y = self._step_in_y(self.mesh_bound, self.sections[1])

        return step_in_x, step_in_y

    def mesh(self) -> Generator[dict, None, None]:
        """
        Compute non overlapping grid bounded within complete_size

        :return:
        """

        (step_in_x, step_in_y) = self._compute_step()

        for y in range(self.sections[1]):

            for x in range(self.sections[0]):
                tx_start = x * step_in_x + self.mesh_bound[0]

                ty_start = y * step_in_y + self.mesh_bound[1]
                tx_end = tx_start + step_in_x - 1
                ty_end = ty_start + step_in_y - 1

                yield self.collate_data((tx_start, ty_start, tx_end, ty_end))

    def collate_data(self, extent: Tuple[float, float, float, float]) -> dict:
        data = {
            "extent": extent,
            "window": get_window(extent, self.mesh_transform),
            "mesh_size": self.mesh_size,
        }
        return data


@dataclass
class ImageOverLapMesh(ImageMesh):
    """
    The Class will compute Grid bounded within complete_size and if the provided grid size overlaps, the the class will
    tune accordingly to provide overlapping grid, The class wont hamper the grid size in any manner, it will find all
    the possible grid of size provided that could fit in complete_size

    Working of this class depends on the geo reference information of the image which acts as the starting point

    The geo reference information to be present in the image is source_min_x, source_max_y and pixel resolution

    Based on the geo reference information present in the image, compute grid of size grid_size over complete_size

    Given an starting image size, final size and its transform this will find all the grid of size image size
    between the given complete size


    The start position of grid and the step size of grid is computed from the transform info provided, usually
    present in geo referenced image

    NOTE - The COORDINATES MUST BE IN `EPSG:26910`

    """

    grid_size: tuple
    mesh_size: tuple
    sections: tuple
    mesh_transform: affine.Affine
    mesh_bound: tuple
    overlap_mesh_bound: Tuple[float, float, float, float]
    buffer_mesh_bound: tuple

    def _is_overlap_in_col_direction(self) -> bool:
        """
        Check if there is any overlap in X direction
        :return:
        """
        return True if self.mesh_size[0] % self.grid_size[0] else False

    def _is_overlap_in_row_direction(self) -> bool:
        """
        Check if there is any overlap in Y direction
        :return:
        """
        return True if self.mesh_size[1] % self.grid_size[1] else False

    def _compute_buffer_step(self) -> Tuple[int, int]:
        """
        To Compute overlapping steps it is essential to compute the the max_x and min_y not based on the complete
        size but to extrapolate the grid size by number of tiles.

        i.e grid_size = grid_size * tiles

        :return:
        """

        buffered_step_in_x = self._step_in_x(self.buffer_mesh_bound, self.sections[0])
        buffered_step_in_y = self._step_in_y(self.buffer_mesh_bound, self.sections[1])

        return buffered_step_in_x, buffered_step_in_y

    def _compute_overlap_step(self) -> Tuple[Union[int, None], Union[int, None]]:
        """
        The overlapping step is nothing but keeping the coordinates in the bound provided in the form of
        complete_img_size, the overlapping step is difference between complete_size  and grid_size

        :return:
        """
        overlap_step_in_x = None
        overlap_step_in_y = None

        if self._is_overlap_in_col_direction():
            overlap_step_in_x = self._step_in_x(self.overlap_mesh_bound)
        if self._is_overlap_in_row_direction():
            overlap_step_in_y = self._step_in_y(self.overlap_mesh_bound)
        return overlap_step_in_x, overlap_step_in_y

    def _compute_step(
        self,
    ) -> Tuple[Tuple[int, int], Tuple[Union[int, None], Union[int, None]]]:
        return self._compute_buffer_step(), self._compute_overlap_step()

    def mesh(self) -> Generator[dict, None, None]:
        """
        Compute Overlapping Grid
        :return:
        """

        (
            (buffered_step_in_x, buffered_step_in_y),
            (overlap_step_in_x, overlap_step_in_y),
        ) = self._compute_step()

        for y in range(self.sections[1]):
            for x in range(self.sections[0]):
                if (x == self.sections[0] - 1) and self._is_overlap_in_col_direction():

                    tx_start = overlap_step_in_x + self.mesh_bound[0]
                else:
                    tx_start = x * buffered_step_in_x + self.mesh_bound[0]
                if y == (self.sections[1] - 1) and self._is_overlap_in_row_direction():
                    ty_start = overlap_step_in_y + self.mesh_bound[1]
                else:
                    ty_start = y * buffered_step_in_y + self.mesh_bound[1]
                tx_end = tx_start + buffered_step_in_x - 1
                ty_end = ty_start + buffered_step_in_y - 1

                yield self.collate_data((tx_start, ty_start, tx_end, ty_end))

    def collate_data(self, extent: Tuple[float, float, float, float]) -> dict:
        data = {
            "extent": extent,
            "window": get_window(extent, self.mesh_transform),
            "mesh_size": self.mesh_size,
        }
        return data


@dataclass
class ShpMesh(Mesh):
    geom: LineString
    grid_width: float
    mesh_width: float

    @property
    def total_grid(self):
        return math.ceil(self.mesh_width / self.grid_width)

    @staticmethod
    def _get_horizontal_lines(grid_lines: np.ndarray) -> List[LineString]:
        horizontal_line_geom = list()
        _, pts_count, _ = grid_lines.shape
        for j in range(pts_count):
            horizontal_line_geom.append(LineString(list(grid_lines[:, j, :])))
        return horizontal_line_geom

    @staticmethod
    def _get_vertical_lines(grid_lines: np.ndarray) -> List[LineString]:
        vertical_line_geom = list()
        pts_count, _, _ = grid_lines.shape
        for j in range(pts_count):
            vertical_line_geom.append(LineString(list(grid_lines[j, :, :])))
        return vertical_line_geom

    def _generate_grid_line(self, pts, distance: float, side: str) -> np.ndarray:
        line = LineString(
            get_reference_shift(
                center_line_points=pts,
                translated_line=self.geom.parallel_offset(distance, side),
            )
        )
        return np.concatenate([line])

    def mesh(self):

        grid_lines = self._get_grid_lines()

        v_line = self._get_vertical_lines(grid_lines)
        h_line = self._get_horizontal_lines(grid_lines)

        # geoms = list()
        for grid in polygonize(MultiLineString(v_line + h_line)):

            line_split_collection = v_line + h_line
            line_split_collection.append(grid.boundary)
            merged_lines = linemerge(line_split_collection)
            border_lines = unary_union(merged_lines)
            decomposition = polygonize(border_lines)
            for d in decomposition:
                # geoms.append(d)
                yield self.collate_data(geom=d)

    def _get_grid_lines(self) -> np.ndarray:
        left_collection = list()
        right_collection = list()

        _pts = [Point(coord) for coord in self.geom.coords]
        for i in range(math.ceil(self.total_grid / 2)):
            left_collection.append(
                self._generate_grid_line(_pts, self.grid_width * (i + 1), "left")
            )
            right_collection.append(
                self._generate_grid_line(_pts, self.grid_width * (i + 1), "right")
            )

        return np.vstack(
            [
                np.array(left_collection[::-1]),
                np.expand_dims(np.array(np.concatenate([self.geom])), axis=0),
                np.array(right_collection),
            ]
        )

    def collate_data(self, geom: Polygon) -> dict:
        return {"geom": geom}


def mesh_from_img_param(
    grid_size: Tuple[int, int] = None,
    mesh_size: Tuple[int, int] = None,
    transform: affine.Affine = None,
    mesh_bounds: Tuple[float, float, float, float] = None,
    overlap: bool = True,
) -> Union[ImageNonOverLapMesh, ImageOverLapMesh]:

    if transform is None:
        raise ValueError("grid_transform can't be None")
    pixel_resolution = get_pixel_resolution(transform)

    if mesh_size is None:
        if mesh_bounds is None:
            raise ValueError("Mesh Bounds and Mesh Size Both can't be None")
        mesh_size = compute_dimension(mesh_bounds, pixel_resolution)
    if grid_size[0] > mesh_size[0] or grid_size[1] > mesh_size[1]:
        raise ValueError(
            "Size Of Grid Can't Be Greater than Mesh, Given {},"
            " Expected less than equal to {}".format(grid_size, mesh_size)
        )
    sections = compute_num_of_col_and_rows(grid_size, mesh_size)

    if overlap:
        buffer_mesh_bound = compute_bounds(
            grid_size[0] * sections[0],
            grid_size[1] * sections[1],
            transform=transform,
        )

        overlap_mesh_bound = compute_bounds(
            mesh_size[0] - grid_size[0],
            mesh_size[1] - grid_size[1],
            transform=transform,
        )

        mesh_bound = compute_bounds(mesh_size[0], mesh_size[1], transform=transform)

        mesh_transform = get_mesh_transform(mesh_size[0], mesh_size[1], transform)

        grid_data = ImageOverLapMesh(
            grid_size,
            mesh_size,
            sections,
            mesh_transform,
            mesh_bound,
            overlap_mesh_bound,
            buffer_mesh_bound,
        )
    else:
        mesh_bound = compute_bounds(mesh_size[0], mesh_size[1], transform=transform)

        mesh_transform = get_mesh_transform(mesh_size[0], mesh_size[1], transform)

        grid_data = ImageNonOverLapMesh(
            grid_size, mesh_size, sections, mesh_transform, mesh_bound
        )
    return grid_data


def create_mesh_using_img_param(
    mesh_bounds: Tuple[float, float, float, float],
    grid_size: Tuple[int, int],
    pixel_resolution: Tuple[float, float],
    is_overlap: bool = False,
) -> Union[ImageNonOverLapMesh, ImageOverLapMesh]:
    assert len(mesh_bounds) == 4, (
        f"Expected mesh_bounds to be in format (minx, miny, maxx, maxy) but got "
        f"{mesh_bounds} of size {len(mesh_bounds)}"
    )

    assert len(grid_size) == 2, (
        f"Expected grid_size to be in format (h x w) but got "
        f"{grid_size} of size {len(grid_size)}"
    )

    assert (
        len(pixel_resolution) == 2
    ), f"Expected pixel_resolution to have size 2 but got {len(grid_size)}"

    mesh = mesh_from_img_param(
        grid_size=grid_size,
        transform=get_affine_transform(
            mesh_bounds[0], mesh_bounds[-1], *pixel_resolution
        ),
        mesh_bounds=mesh_bounds,
        overlap=is_overlap,
    )
    return mesh


def mesh_from_line(line: LineString, grid_width: float, mesh_width: float):
    return ShpMesh(line, grid_width, mesh_width)
