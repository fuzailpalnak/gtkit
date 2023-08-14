import numpy as np
from scipy.spatial.distance import pdist, squareform

from py_gis_utility.helper import (
    is_line_segment_3d,
    is_value_3d,
    is_point_3d,
)


def angle_between_vector(v1: tuple, v2: tuple):
    """
    two vectors have either the same direction -  https://stackoverflow.com/a/13849249/71522
    :param v1:
    :param v2:
    :return:
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def vector(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """

    :param p1:
    :param p2:
    :return:
    """
    return np.subtract(p1, p2)


def magnitude(vec: np.ndarray, axis=-1) -> np.ndarray:
    return np.linalg.norm(vec, axis=axis)


def unit_vector(vec: np.ndarray, axis=-1) -> np.ndarray:
    """

    :param axis:
    :param vec:
    :return: unit vector
    """
    return np.divide(
        vec, np.concatenate([magnitude(vec, axis)[:, :, np.newaxis]] * 2, axis=axis)
    )


def euclidean_between_two_sets(
    coordinates_a: np.ndarray, coordinates_b: np.ndarray
) -> np.ndarray:
    """
    This function will return set of distances from coordinates_b to coordinates_a
    ex -
        coordinates_b = [[0, 0], [1, 1]]
        coordinates_a = [[2, 1], [3, 1], [2, 4], [5, 2]]

        return
            sets of distances from [0, 0] to all the coordinates present in coordinates_a
            sets of distances from [1, 1] to all the coordinates present in coordinates_a

    visually -
        | distance from [0, 0] to [2, 1]   | distance from [1, 1] to [2, 1] |
        | distance from [0, 0] to [3, 1]   | distance from [1, 1] to [3, 1] |
        | distance from [0, 0] to [2, 4]   | distance from [1, 1] to [2, 4] |
        | distance from [0, 0] to [5, 2]   | distance from [1, 1] to [5, 2] |

    resulting in shape of [n_coordinates_a, n_coordinates_b]

    :param coordinates_a: array of shape [n_coordinates_a, 2]
    :param coordinates_b: array of shape [n_coordinates_b, 2]
    :return: array of shape [n_coordinates_a, n_coordinates_b]
    """

    assert (
        type(coordinates_a) is np.ndarray and type(coordinates_b) is np.ndarray
    ), f"Expected to have input type 'np.ndarray' got {type(coordinates_a)}, {type(coordinates_b)}"

    assert (
        coordinates_b.shape[-1] == 2 and coordinates_b.ndim == 2
    ), f"Expected input_coordinates to have shape '[number of points, 2]' got {coordinates_b.shape}"

    assert (
        coordinates_a.shape[-1] == 2 and coordinates_a.ndim == 2
    ), f"Expected input_coordinates to have shape '[number of points, 2]' got {coordinates_a.shape}"

    distances = np.linalg.norm(
        np.concatenate(
            [coordinates_a[np.newaxis, :, :]] * coordinates_b.shape[0], axis=0
        )
        - coordinates_b[:, np.newaxis, :],
        axis=-1,
    ).T
    return distances


def euclidean(coordinates_a: np.ndarray) -> np.ndarray:
    """
    This function will return set of distances from coordinates_a to coordinates_a
    ex -
        coordinates_a = [[0, 0], [1, 1]]

        return
            sets of distances from [0, 0] to all the coordinates present in coordinates_a
            sets of distances from [1, 1] to all the coordinates present in coordinates_a

    visually -
        | distance from [0, 0] to [0, 0]   | distance from [1, 1] to [0, 0] |
        | distance from [0, 0] to [1, 1]   | distance from [1, 1] to [1, 1] |

    resulting in shape of [n_coordinates_a, n_coordinates_a]

    :param coordinates_a: array of shape [n_coordinates_a, 2]
    :return: array of shape [n_coordinates_a, n_coordinates_a]
    """
    assert type(coordinates_a) is np.ndarray, (
        "Expected to have input type 'np.ndarray'" "got %s",
        (type(coordinates_a),),
    )
    assert (
        type(coordinates_a) is np.ndarray
    ), f"Expected to have input type 'np.ndarray' got {type(coordinates_a)}"

    assert (
        coordinates_a.shape[-1] == 2 and coordinates_a.ndim == 2
    ), f"Expected input_coordinates to have shape '[number of points, 2]' got {coordinates_a.shape}"

    return squareform(pdist(coordinates_a))


def perpendicular_distance_from_point_to_line_segment_in_2d(
    line_segment: np.ndarray, coordinates: np.ndarray
) -> np.ndarray:
    """
    The function will compute perpendicular distance for the coordinates value present in coordinates to the given
    input line segment

    :param line_segment: array of shape [number_of_line_segments, 2, 2]

            If the  line segment to which perpendicular distances are to be computed then pass it as
            follows :
                -- the dimension [number_of_line_segments, 2, 2] are [
                                                                        [
                                                                        [start_line_segment_1_x, start_line_segment_1_y],
                                                                        [end_line_segment_1_x,   end_line_segment_1_y]
                                                                        ],
                                                                        [
                                                                        [start_line_segment_2_x, start_line_segment_2_y],
                                                                        [end_line_segment_2_x,   end_line_segment_2_y]
                                                                        ],
                                                                        ....
                                                                        ...
                                                                        [
                                                                        [start_line_segment_n_x, start_line_segment_n_y],
                                                                        [end_line_segment_n_x,   end_line_segment_n_y]
                                                                        ],
                                                                    ]
    :param coordinates: array of shape [number of points, 2]
    :return: array of shape [number of segments, number of points]

    if single line segment is passed for computation, i.e. input with dim [1, 2, 2] then expect the output in
        format:
            | distance from line_Segment_1 to coordinate[0]  .... | .... distance from line_Segment_1 to coordinate[n] |

        if multiple line segments are passed for computation, i.e. input with dim [n_segments, 2, 2] then expect
         the output in format:
            | distance from line_Segment_1 to coordinate[0]  ....  | ... distance from line_Segment_1 to coordinate[n] |
            | distance from line_Segment_2 to coordinate[0]  ....  | ... distance from line_Segment_2 to coordinate[n] |
            ....
            ...
            | distance from line_Segment_n to coordinate[0]  ....  | ...distance from line_Segment_n to coordinate[n] |

    """

    # https://stackoverflow.com/a/53176074/7984359
    # https://math.stackexchange.com/questions/1300484/distance-between-line-and-a-point
    # https://www.geeksforgeeks.org/minimum-distance-from-a-point-to-the-line-segment-using-vectors/
    # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points

    assert (
        type(line_segment) is np.ndarray and type(coordinates) is np.ndarray
    ), f"Expected to have input type 'np.ndarray' got {type(line_segment)} and {type(coordinates)}"

    assert coordinates.ndim == 2 and coordinates.shape[-1] == 2, (
        f"Expected coordinates to be '2' dimensional and have shape '[number of point, 2]' got {coordinates.shape},"
        f" {coordinates.ndim}"
    )

    assert is_line_segment_3d(line_segment), (
        f"Expected line segments to be either '[n_line_segments, 2, 2]' for n_dim == 3"
        f"got {line_segment.shape} for n_dim == {line_segment.ndim}"
    )

    coordinates_repeat_copy = np.concatenate(
        [coordinates[np.newaxis, :, :]] * line_segment.shape[0], axis=0
    )

    dp = line_segment[:, 1:2, :] - line_segment[:, 0:1, :]
    st = dp[:, :, 0:1] ** 2 + dp[:, :, 1:2] ** 2

    u = (
        (coordinates_repeat_copy[:, :, 0:1] - line_segment[:, 0:1, 0:1]) * dp[:, :, 0:1]
        + (coordinates_repeat_copy[:, :, 1:2] - line_segment[:, 0:1, 1:2])
        * dp[:, :, 1:2]
    ) / st

    u[u > 1.0] = 1.0
    u[u < 0.0] = 0.0

    dx = (line_segment[:, 0:1, 0:1] + u * dp[:, :, 0:1]) - coordinates_repeat_copy[
        :, :, 0:1
    ]
    dy = (line_segment[:, 0:1, 1:2] + u * dp[:, :, 1:2]) - coordinates_repeat_copy[
        :, :, 1:2
    ]

    return np.squeeze(np.sqrt(dx ** 2 + dy ** 2), axis=-1)


def new_perpendicular_point_to_line_segment(
    line_segment: np.ndarray, distance_from_the_line: np.ndarray
):
    """
    Get perpendicular point with reference to start and end point of the segment

    :param line_segment: array of shape [number_of_line_segments, 2, 2]

            If the line segment to which perpendicular distances are to be computed then pass it as
            follows :
                -- the dimension [number_of_line_segments, 2, 2] are [
                                                                        [
                                                                        [start_line_segment_1_x, start_line_segment_1_y],
                                                                        [end_line_segment_1_x,   end_line_segment_1_y]
                                                                        ],
                                                                        [
                                                                        [start_line_segment_2_x, start_line_segment_2_y],
                                                                        [end_line_segment_2_x,   end_line_segment_2_y]
                                                                        ],
                                                                        ....
                                                                        ...
                                                                        [
                                                                        [start_line_segment_n_x, start_line_segment_n_y],
                                                                        [end_line_segment_n_x,   end_line_segment_n_y]
                                                                        ],
                                                                    ]
    :param distance_from_the_line: how far the new point to create from the reference
    :return:(perpendicular points with reference to start, perpendicular points with reference to end)
            -
                return is of shape [number_of_segments, 2, 2]

                    [A_n]                          [C_n]
                    |     line_segment_with       |
                    |-----------------------------|
                    |     index value 'n'         |
                    [B_n]                         [D_n]

                to get points -

                    A_n - perpendicular_with_start[segment_index_value_n, 0, :]
                    B_n - perpendicular_with_start[segment_index_value_n, 1, :]
                    C_n - perpendicular_with_end[segment_index_value_n, 0, :]
                    D_n - perpendicular_with_end[segment_index_value_n, 1, :]

    """
    assert (
        type(line_segment) is np.ndarray and type(distance_from_the_line) is np.ndarray
    ), (
        f"Expected to have input type ['np.ndarray', 'np.ndarray'] got {type(line_segment)}"
        f" and {type(distance_from_the_line)}"
    )

    assert is_line_segment_3d(line_segment), (
        f"Expected line segments to be either '[n_line_segments, 2, 2]' for n_dim == 3 "
        f"got {line_segment.shape} for n_dim == {line_segment.ndim}"
    )

    assert is_value_3d(distance_from_the_line), (
        f"Expected values to be either '[n_values, 1, 1]' for n_dim == 3 or '[1, 1]' for n_dim == 2, "
        f"got {distance_from_the_line.shape} for n_dim == {distance_from_the_line.ndim}"
    )

    assert line_segment.shape[0] == distance_from_the_line.shape[0], (
        f"Expected number of distance to "
        f"be equal to number of line segments,"
        f" got {line_segment.shape[0],distance_from_the_line.shape[0]}"
    )

    assert np.all(
        distance_from_the_line >= 0
    ), f"Expected distance to be 'non zero' got {distance_from_the_line}"

    x1, y1 = line_segment[:, 0:1, 0:1], line_segment[:, 0:1, 1:2]
    x2, y2 = line_segment[:, 1:2, 0:1], line_segment[:, 1:2, 1:2]

    dx = x1 - x2
    dy = y1 - y2

    dist = np.linalg.norm(np.array([dx, dy]), axis=0)

    x_perpendicular = distance_from_the_line * (dx / dist)
    y_perpendicular = distance_from_the_line * (dy / dist)

    point_with_start_as_reference_x = [
        np.array(x1 + y_perpendicular).squeeze(axis=-1),
        np.array(y1 - x_perpendicular).squeeze(axis=-1),
    ]

    point_with_start_as_reference_y = [
        np.array(x1 - y_perpendicular).squeeze(axis=-1),
        np.array(y1 + x_perpendicular).squeeze(axis=-1),
    ]

    point_with_start_as_reference = np.dstack(
        (
            np.array(point_with_start_as_reference_x).squeeze(axis=-1),
            np.array(point_with_start_as_reference_y).squeeze(axis=-1),
        )
    )
    point_with_end_as_reference_x = [
        np.array(x2 - y_perpendicular).squeeze(axis=-1),
        np.array(y2 + x_perpendicular).squeeze(axis=-1),
    ]

    point_with_end_as_reference_y = [
        np.array(x2 + y_perpendicular).squeeze(axis=-1),
        np.array(y2 - x_perpendicular).squeeze(axis=-1),
    ]

    point_with_end_as_reference = np.dstack(
        (
            np.array(point_with_end_as_reference_x).squeeze(axis=-1),
            np.array(point_with_end_as_reference_y).squeeze(axis=-1),
        )
    )

    perpendicular_with_start = point_with_start_as_reference.swapaxes(-1, 1).T
    perpendicular_with_end = point_with_end_as_reference.swapaxes(-1, 1).T

    return perpendicular_with_start, perpendicular_with_end


def new_coordinate_based_on_angle_and_distance(
    points: np.ndarray, angle: np.ndarray, distance: np.ndarray
) -> np.ndarray:
    """
    Given a Point find a new point at an given 'angle' with given 'distance'

          / B [New Point]
         /
        /  angle CAB and distance AB [GIVEN]
       A ------------ C

    # https://math.stackexchange.com/questions/39390/determining-end-coordinates-of-line-with-the-specified-length-and-angle

    :param points: array of shape [number_of_line_segments, 1, 2]
            If the line segment to which perpendicular distances are to be computed then pass it as
            follows :
                -- the dimension [number_of_points, 1, 2] are [
                                                                        [
                                                                        [point_1_x, point_1_y],
                                                                        ],
                                                                        [
                                                                        [point_2_x, point_2_y],
                                                                        ],
                                                                        ....
                                                                        ...
                                                                        [
                                                                        [point_n_x, point_n_y],
                                                                        ],
                                                                    ]
    :param angle: array of shape [number_of_points, 1, 1]
    :param distance: array of shape [number_of_points, 1, 1]
    :return:
    """
    assert (
        type(points) is np.ndarray
        and type(distance) is np.ndarray
        and type(angle) is np.ndarray
    ), (
        "Expected to have input type ['np.ndarray', 'np.ndarray', 'np.ndarray']"
        f"got {type(points), type(distance), type(angle)}"
    )

    assert is_point_3d(points), (
        f"Expected points to be either '[n_points, 1, 2]' for n_dim == 3 "
        f"got {points.shape} for n_dim == {points.ndim}"
    )
    assert is_value_3d(angle), (
        f"Expected values to be either '[n_values, 1, 1]' for n_dim == 3 "
        f"got {angle.shape} for n_dim == {angle.ndim}"
    )
    assert is_value_3d(distance), (
        f"Expected values to be either '[n_values, 1, 1]' "
        f"got {distance.shape} for n_dim == {distance.ndim}"
    )
    assert points.shape[0] == distance.shape[0] == angle.shape[0], (
        f"Expected number of points, number of distance, "
        f"number of angles to have the same count"
        f" got {points.shape[0],distance.shape[0], angle.shape}"
    )

    assert np.all(distance >= 0), f"Expected distance to be 'non zero' got {distance}"

    return np.concatenate(
        [
            points[:, :, 0:1] + (distance * np.cos(angle)),
            points[:, :, 1:2] + (distance * np.sin(angle)),
        ],
        axis=-1,
    )


def new_point_after_certain_distance(
    line_segments: np.ndarray, distance_from_start: np.ndarray
) -> np.ndarray:
    """
    https://math.stackexchange.com/questions/175896/finding-a-point-along-a-line-a-certain-distance-away-from-another-point
    https://math.stackexchange.com/a/426810

    :param distance_from_start: array specifying the distance to compute the point
            shape [number_of_line_segments, 1, 1] or [1, 1]
    :param line_segments: array of shape [number_of_line_segments, 2, 2]

            If the line segment to which perpendicular distances are to be computed then pass it as
            follows :
                -- the dimension [number_of_line_segments, 2, 2] are [
                                                                        [
                                                                        [start_line_segment_1_x, start_line_segment_1_y],
                                                                        [end_line_segment_1_x,   end_line_segment_1_y]
                                                                        ],
                                                                        [
                                                                        [start_line_segment_2_x, start_line_segment_2_y],
                                                                        [end_line_segment_2_x,   end_line_segment_2_y]
                                                                        ],
                                                                        ....
                                                                        ...
                                                                        [
                                                                        [start_line_segment_n_x, start_line_segment_n_y],
                                                                        [end_line_segment_n_x,   end_line_segment_n_y]
                                                                        ],
                                                                    ]
    :return: points on the line segment with distance 'd'

    """
    assert (
        type(line_segments) is np.ndarray and type(distance_from_start) is np.ndarray
    ), f"Expected to have input type ['np.ndarray', 'np.ndarray'] got {type(line_segments), type(distance_from_start)}"

    assert is_line_segment_3d(line_segments), (
        f"Expected line segments to be either '[n_values, 2, 2]' for n_dim == 3 "
        f"got {line_segments.shape} for n_dim == {line_segments.ndim}"
    )

    assert np.all(
        distance_from_start >= 0
    ), f"Expected distance_from_the_line to be  'non negative' got {distance_from_start}"

    assert is_value_3d(distance_from_start), (
        f"Expected values to be either '[n_values, 1, 1]' for n_dim == 3"
        f"got {distance_from_start.shape} for n_dim == {distance_from_start.ndim}"
    )

    assert line_segments.shape[0] == distance_from_start.shape[0], (
        f"Expected input to have same number of count,"
        f" got {line_segments.shape[0],distance_from_start.shape[0]}"
    )

    vec = vector(line_segments[:, 1:2, :], line_segments[:, 0:1, :])
    vec_mag = np.concatenate([magnitude(vec, -1)[:, :, np.newaxis]] * 2, axis=-1)
    return (distance_from_start * vec_mag) * unit_vector(vec)
