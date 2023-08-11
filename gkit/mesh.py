from dataclasses import dataclass
from typing import Union, Tuple, Generator

import affine
from gkit.utils import (
    get_window,
    get_pixel_resolution,
    compute_dimension,
    compute_bounds,
    get_mesh_transform,
    get_affine_transform,
    compute_num_of_col_and_rows,
)


class Mesh:
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

    def extent(self) -> Generator[dict, None, None]:
        """
        Compute Mesh

        :return:
        """

        raise NotImplementedError

    def collate_data(self, extent: Tuple[float, float, float, float]) -> dict:
        raise NotImplementedError


@dataclass
class ImageNonOverLapMesh(Mesh):
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

    def extent(self) -> Generator[dict, None, None]:
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
class ImageOverLapMesh(Mesh):
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

    def extent(self) -> Generator[dict, None, None]:
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


def mesh_from_geo_transform(
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


def create_mesh(
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

    mesh = mesh_from_geo_transform(
        grid_size=grid_size,
        transform=get_affine_transform(
            mesh_bounds[0], mesh_bounds[-1], *pixel_resolution
        ),
        mesh_bounds=mesh_bounds,
        overlap=is_overlap,
    )
    return mesh
