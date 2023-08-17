import numpy as np
import math
import affine

from dataclasses import dataclass

from shapely.geometry import Point, LineString, MultiLineString, Polygon
from shapely.ops import polygonize, linemerge, unary_union

from affine import Affine

from gtkit.gops import get_reference_shift
from typing import Union, Tuple, Generator, List

from gtkit.imutils import (
    get_pixel_resolution,
    get_affine_transform,
    compute_bounds,
    get_window,
)


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


@dataclass
class Mesh:
    """
    Base class for generating mesh data.

    Attributes:
        None

    Methods:
        mesh(): Abstract method to compute mesh.
        collate_data(**kwargs): Abstract method to collate data.
    """

    def mesh(self) -> Generator[dict, None, None]:
        """
        Abstract method to compute mesh.

        Yields:
            dict: Data describing each mesh element.
        """
        raise NotImplementedError

    def collate_data(self, **kwargs):
        """
        Abstract method to collate data.

        Args:
            **kwargs: Arbitrary keyword arguments.

        Returns:
            dict: Collated data.
        """
        raise NotImplementedError


class ImageMesh(Mesh):
    """
    Class for generating image-based mesh data.

    Attributes:
        None

    Methods:
        _compute_step(): Abstract method to compute step in X and Y direction.
        _step_in_x(bound: Tuple[float, float, float, float], normalizer: int = 1) -> int: Compute step size in X.
        _step_in_y(bound: Tuple[float, float, float, float], normalizer: int = 1) -> int: Compute step size in Y.
        mesh(): Abstract method to compute mesh.
        collate_data(extent: Tuple[float, float, float, float]) -> dict: Collate data for a mesh element.
    """

    def _compute_step(self) -> Tuple[int, int]:
        """
        Compute step in X and Y direction.

        Returns:
            Tuple[int, int]: Step size in X and Y directions.
        """

        raise NotImplementedError

    @staticmethod
    def _step_in_x(
        bound: Tuple[float, float, float, float], normalizer: int = 1
    ) -> int:
        """
        Compute step size to take in X.

        Args:
            bound (Tuple[float, float, float, float]): Boundary coordinates.
            normalizer (int, optional): Step size normalizer. Larger values lead to smaller steps. Defaults to 1.

        Returns:
            int: Step size in X direction.
        """

        return int(((bound[2] - bound[0]) / normalizer))

    @staticmethod
    def _step_in_y(
        bound: Tuple[float, float, float, float], normalizer: int = 1
    ) -> int:
        """
        Compute step size to take in Y.

        Args:
            bound (Tuple[float, float, float, float]): Boundary coordinates.
            normalizer (int, optional): Step size normalizer. Larger values lead to smaller steps. Defaults to 1.

        Returns:
            int: Step size in Y direction.
        """
        return int(((bound[-1] - bound[1]) / normalizer))

    def mesh(self) -> Generator[dict, None, None]:
        """
        Compute mesh elements.

        Yields:
            dict: Data describing each mesh element.
        """

        raise NotImplementedError

    def collate_data(self, extent: Tuple[float, float, float, float]) -> dict:
        """
        Collate data for a mesh element.

        Args:
            extent (Tuple[float, float, float, float]): Extent coordinates.

        Returns:
            dict: Collated data.
        """

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

    Attributes:
        grid_size (tuple): Size of the grid in rows and columns.
        mesh_size (tuple): Size of the mesh.
        sections (tuple): Number of sections in rows and columns.
        mesh_transform (affine.Affine): Mesh transformation.
        mesh_bound (tuple): Mesh boundary coordinates.

    Methods:
        _compute_step(): Compute step in X and Y direction.
        mesh(): Generate non-overlapping grid within specified bounds.
        collate_data(extent: Tuple[float, float, float, float]) -> dict: Collate data for a mesh element.

    """

    grid_size: tuple
    mesh_size: tuple
    sections: tuple
    mesh_transform: affine.Affine
    mesh_bound: tuple

    def _compute_step(self) -> Tuple[int, int]:
        """
        Compute step in X and Y direction.

        Returns:
            Tuple[int, int]: Step size in X and Y directions.
        """

        step_in_x = self._step_in_x(self.mesh_bound, self.sections[0])
        step_in_y = self._step_in_y(self.mesh_bound, self.sections[1])

        return step_in_x, step_in_y

    def mesh(self) -> Generator[dict, None, None]:
        """
        Generate non-overlapping grid bounded within specified bounds.

        Yields:
            dict: Data describing each mesh element.
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
        """
        Collate data for a mesh element.

        Args:
            extent (Tuple[float, float, float, float]): Extent coordinates.

        Returns:
            dict: Collated data.
        """
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

    Attributes:
        grid_size (tuple): Size of the grid in rows and columns.
        mesh_size (tuple): Size of the mesh.
        sections (tuple): Number of sections in rows and columns.
        mesh_transform (affine.Affine): Mesh transformation.
        mesh_bound (tuple): Mesh boundary coordinates.
        overlap_mesh_bound (Tuple[float, float, float, float]): Overlap mesh boundary coordinates.
        buffer_mesh_bound (tuple): Buffer mesh boundary coordinates.

    Methods:
        _is_overlap_in_col_direction(): Check if there is overlap in X direction.
        _is_overlap_in_row_direction(): Check if there is overlap in Y direction.
        _compute_buffer_step(): Compute buffer step in X and Y direction.
        _compute_overlap_step(): Compute overlap step in X and Y direction.
        _compute_step(): Compute buffer and overlap steps.
        mesh(): Generate overlapping grid within specified bounds.
        collate_data(extent: Tuple[float, float, float, float]) -> dict: Collate data for a mesh element.

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
        Check if there is overlap in X direction.

        Returns:
            bool: True if overlap exists, False otherwise.
        """
        return True if self.mesh_size[0] % self.grid_size[0] else False

    def _is_overlap_in_row_direction(self) -> bool:
        """
        Check if there is overlap in Y direction.

        Returns:
            bool: True if overlap exists, False otherwise.
        """
        return True if self.mesh_size[1] % self.grid_size[1] else False

    def _compute_buffer_step(self) -> Tuple[int, int]:
        """
        Compute buffer step in X and Y direction.

        Returns:
            Tuple[int, int]: Buffer step size in X and Y directions.
        """

        buffered_step_in_x = self._step_in_x(self.buffer_mesh_bound, self.sections[0])
        buffered_step_in_y = self._step_in_y(self.buffer_mesh_bound, self.sections[1])

        return buffered_step_in_x, buffered_step_in_y

    def _compute_overlap_step(self) -> Tuple[Union[int, None], Union[int, None]]:
        """
        Compute overlap step in X and Y direction.

        Returns:
            Tuple[Union[int, None], Union[int, None]]: Overlap step size in X and Y directions.
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
        """
        Compute buffer and overlap steps.

        Returns:
            Tuple[Tuple[int, int], Tuple[Union[int, None], Union[int, None]]]:
                Tuple containing buffer step sizes in X and Y directions,
                and overlap step sizes in X and Y directions.
        """
        return self._compute_buffer_step(), self._compute_overlap_step()

    def mesh(self) -> Generator[dict, None, None]:
        """
        Generate overlapping grid within specified bounds.

        Yields:
            dict: Data describing each mesh element.
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
        """
        Collate data for a mesh element.

        Args:
            extent (Tuple[float, float, float, float]): Extent coordinates.

        Returns:
            dict: Collated data.
        """

        data = {
            "extent": extent,
            "window": get_window(extent, self.mesh_transform),
            "mesh_size": self.mesh_size,
        }
        return data


@dataclass
class ShpMesh(Mesh):
    """
    Class for generating mesh data based on a LineString geometry.

    Attributes:
        geom (LineString): LineString geometry.
        grid_width (float): Width of the grid cells.
        mesh_width (float): Width of the mesh.

    Methods:
        _get_horizontal_lines(grid_lines: np.ndarray) -> List[LineString]:
            Get horizontal lines from grid lines.
        _get_vertical_lines(grid_lines: np.ndarray) -> List[LineString]:
            Get vertical lines from grid lines.
        _generate_grid_line(pts, distance: float, side: str) -> np.ndarray:
            Generate a grid line.
        mesh(): Generate mesh based on the LineString geometry.
        _get_grid_lines() -> np.ndarray:
            Get grid lines from LineString geometry.
        collate_data(geom: Polygon) -> dict:
            Collate data for a mesh element.
    """

    geom: LineString
    grid_width: float
    mesh_width: float

    @property
    def total_grid(self):
        return math.ceil(self.mesh_width / self.grid_width)

    @staticmethod
    def _get_horizontal_lines(grid_lines: np.ndarray) -> List[LineString]:
        """
        Get horizontal lines from grid lines.

        Args:
            grid_lines (np.ndarray): Grid lines.

        Returns:
            List[LineString]: List of horizontal LineString geometries.
        """

        horizontal_line_geom = list()
        _, pts_count, _ = grid_lines.shape
        for j in range(pts_count):
            horizontal_line_geom.append(LineString(list(grid_lines[:, j, :])))
        return horizontal_line_geom

    @staticmethod
    def _get_vertical_lines(grid_lines: np.ndarray) -> List[LineString]:
        """
        Get vertical lines from grid lines.

        Args:
            grid_lines (np.ndarray): Grid lines.

        Returns:
            List[LineString]: List of vertical LineString geometries.
        """

        vertical_line_geom = list()
        pts_count, _, _ = grid_lines.shape
        for j in range(pts_count):
            vertical_line_geom.append(LineString(list(grid_lines[j, :, :])))
        return vertical_line_geom

    def _generate_grid_line(self, pts, distance: float, side: str) -> np.ndarray:
        """
        Generate a grid line.

        Args:
            pts (List[Point]): List of Points.
            distance (float): Distance value.
            side (str): Side to generate grid line.

        Returns:
            np.ndarray: Grid line coordinates.
        """

        line = LineString(
            get_reference_shift(
                center_line_points=pts,
                translated_line=self.geom.parallel_offset(distance, side),
            )
        )
        return np.concatenate([line])

    def mesh(self):
        """
        Generate mesh based on the LineString geometry.

        Yields:
            dict: Data describing each mesh element.
        """
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
                yield self.collate_data(
                    geom=d, vertical_lines=v_line, horizontal_line=h_line
                )

    def _get_grid_lines(self) -> np.ndarray:
        """
        Get grid lines from LineString geometry.

        Returns:
            np.ndarray: Grid lines.
        """

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

    def collate_data(self, geom: Polygon, **kwargs) -> dict:
        """
        Collate data for a mesh element.

        Args:
            geom (Polygon): Polygon geometry.

        Returns:
            dict: Collated data.
        """

        return {**{"geom": geom}, **kwargs}


def mesh_from_img_param(
    grid_size: Tuple[int, int] = None,
    mesh_size: Tuple[int, int] = None,
    transform: affine.Affine = None,
    mesh_bounds: Tuple[float, float, float, float] = None,
    overlap: bool = True,
) -> Union[ImageNonOverLapMesh, ImageOverLapMesh]:
    """
    Create a mesh from image parameters.

    Args:
        grid_size (Tuple[int, int], optional): Size of the grid in rows and columns. Defaults to None.
        mesh_size (Tuple[int, int], optional): Size of the mesh. Defaults to None.
        transform (affine.Affine, optional): Affine transformation. Defaults to None.
        mesh_bounds (Tuple[float, float, float, float], optional): Bounds of the mesh. Defaults to None.
        overlap (bool, optional): Whether to use overlapping mesh. Defaults to True.

    Returns:
        Union[ImageNonOverLapMesh, ImageOverLapMesh]: Either ImageNonOverLapMesh or ImageOverLapMesh instance.
    """
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
    """
    Create a mesh using image parameters.

    Args:
        mesh_bounds (Tuple[float, float, float, float]): Tuple of minimum and maximum X and Y coordinates of the mesh.
        grid_size (Tuple[int, int]): Tuple representing the grid size in rows and columns.
        pixel_resolution (Tuple[float, float]): Tuple representing the pixel resolution in X and Y directions.
        is_overlap (bool, optional): Boolean indicating whether to use overlapping mesh. Defaults to False.

    Returns:
        Union[ImageNonOverLapMesh, ImageOverLapMesh]: Either ImageNonOverLapMesh or ImageOverLapMesh instance.
    """

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


def mesh_from_line(line: LineString, grid_width: float, mesh_width: float) -> ShpMesh:
    """
    Create a mesh from a LineString.

    Args:
        line (LineString): LineString geometry representing the line.
        grid_width (float): Width of the grid cells.
        mesh_width (float): Width of the mesh.

    Returns:
        ShpMesh: ShpMesh instance.
    """
    return ShpMesh(line, grid_width, mesh_width)
