from typing import Tuple

import geopandas as gpd
from shapely.geometry import box
from sklearn.cluster import AgglomerativeClustering

from gtkit.gops import centroid_with_z, midpoint


def gpdread(inp_file: str) -> gpd.GeoDataFrame:
    """
    Read a GeoDataFrame from a file.

    Args:
        inp_file (str): The path to the input file.

    Returns:
        gpd.GeoDataFrame: The GeoDataFrame read from the file.
    """
    return gpd.read_file(inp_file)


def nearest_linestring_to_a_point_df(
    pts: gpd.GeoDataFrame, lines: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Find the nearest linestring to each point in a GeoDataFrame.

    Args:
        pts (gpd.GeoDataFrame): The GeoDataFrame containing points.
        lines (gpd.GeoDataFrame): The GeoDataFrame containing linestrings.

    Returns:
        gpd.GeoDataFrame: The resulting GeoDataFrame with nearest line information.
    """
    # https://stackoverflow.com/questions/70626218/how-to-find-the-nearest-linestring-to-a-point
    gdf_p = pts.to_crs(pts.estimate_utm_crs())
    gdf_l = lines.to_crs(gdf_p.crs)

    df_n = gpd.sjoin_nearest(gdf_p, gdf_l).merge(
        gdf_l, left_on="index_right", right_index=True
    )

    df_n["distance"] = df_n.apply(
        lambda r: r["geometry_x"].distance(r["geometry_y"]), axis=1
    )
    return df_n


def points_in_a_radius_around_centroid(
    centroid: gpd.GeoDataFrame, lines: gpd.GeoDataFrame, raduis: int = 5
) -> gpd.GeoDataFrame:
    """
    Find points within a specified radius around centroids.

    Args:
        centroid (gpd.GeoDataFrame): The GeoDataFrame containing centroid geometries.
        lines (gpd.GeoDataFrame): The GeoDataFrame containing linestrings.
        raduis (int, optional): The radius within which to search for points. Defaults to 5.

    Returns:
        gpd.GeoDataFrame: The resulting GeoDataFrame with points within the radius.
    """
    # https://gis.stackexchange.com/questions/246782/geopandas-line-polygon-intersection
    return gpd.sjoin(centroid.buffer(raduis), lines, op="intersects")


def get_centroid3d(geom_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Calculate the 3D centroid for each geometry in the GeoDataFrame.

    Args:
        geom_df (gpd.GeoDataFrame): The GeoDataFrame containing geometries.

    Returns:
        gpd.GeoDataFrame: The GeoDataFrame with 3D centroids added.
    """
    geom_df["geometry"] = geom_df.apply(
        lambda row: centroid_with_z(
            row.geom if "geom" in row else row.geometry,
        ),
        axis=1,
    )
    return geom_df


def get_centroid(geom_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Calculate the centroid for each geometry in the GeoDataFrame.

    Args:
        geom_df (gpd.GeoDataFrame): The GeoDataFrame containing geometries.

    Returns:
        gpd.GeoDataFrame: The GeoDataFrame with centroid points added.
    """
    geom_df["midpoint"] = geom_df.apply(
        lambda row: row["geometry"].centroid,
        axis=1,
    )
    return geom_df


def get_common_columns(
    df_a: gpd.GeoDataFrame, df_b: gpd.GeoDataFrame
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Get GeoDataFrames with common columns between two GeoDataFrames.

    Args:
        df_a (gpd.GeoDataFrame): The first GeoDataFrame.
        df_b (gpd.GeoDataFrame): The second GeoDataFrame.

    Returns:
        Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]: GeoDataFrames with common columns.
    """
    df_b_drop = non_overlapping_columns(df_b, df_a)
    if len(df_b_drop) != 0:
        df_b = df_b.drop(columns=df_b_drop)

    df_a_drop = non_overlapping_columns(df_a, df_b)
    if len(df_a_drop) != 0:
        df_a = df_a.drop(columns=df_a_drop)
    return df_a, df_b


def non_overlapping_columns(a: gpd.GeoDataFrame, b: gpd.GeoDataFrame) -> list:
    """
    Get a list of columns that are in GeoDataFrame `a` but not in GeoDataFrame `b`.

    Args:
        a (gpd.GeoDataFrame): The first GeoDataFrame.
        b (gpd.GeoDataFrame): The second GeoDataFrame.

    Returns:
        list: A list of non-overlapping column names.
    """
    return list(set(a.columns.tolist()).difference(set(b.columns.tolist())))


def project_points_to_line(
    points: gpd.GeoDataFrame, lines: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Project points onto the nearest positions on lines.

    Args:
        points (gpd.GeoDataFrame): The GeoDataFrame containing points to project.
        lines (gpd.GeoDataFrame): The GeoDataFrame containing lines to project onto.

    Returns:
        gpd.GeoDataFrame: The GeoDataFrame with projected points.
    """
    points["PROJECTED"] = points.apply(
        lambda row: lines.interpolate(
            lines.project(row.geom if "geom" in row else row.geometry)
        ),
        axis=1,
    )
    return points


def merge_geometries(in_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Merge geometries within a GeoDataFrame into a single geometry.

    Args:
        in_df (gpd.GeoDataFrame): The GeoDataFrame containing geometries to merge.

    Returns:
        gpd.GeoDataFrame: The GeoDataFrame with merged geometries.
    """
    if not in_df.empty:
        in_df = in_df.dropna(axis=1)
        unary_geom = (
            [in_df.geometry.unary_union]
            if len(in_df) == 1
            else in_df.geometry.unary_union.geoms
        )

        geoms = list()
        for g in unary_geom:
            geoms.append(g)

        df = gpd.GeoDataFrame(
            columns=["ids", "geometry"], crs=in_df.crs, geometry="geometry"
        )
        df["geometry"] = list(geoms)
        df["ids"] = list(range(len(geoms)))
        return df


def merge_geoms_df_from_file(inp_file: str) -> gpd.GeoDataFrame:
    """
    Read a GeoDataFrame from a file and merge its geometries.

    Args:
        inp_file (str): The path to the input file.

    Returns:
        gpd.GeoDataFrame: The GeoDataFrame with merged geometries.
    """
    return merge_geometries(gpdread(inp_file=inp_file))


def cluster_shapes_by_distance(
    df: gpd.GeoDataFrame, distance: float, check_crs: bool = False
) -> gpd.GeoDataFrame:
    """
    https://gis.stackexchange.com/a/437352

    Make groups for all shapes within a defined distance. For a shape to be
    excluded from a group, it must be greater than the defined distance
    from *all* shapes in the group.
    Distances are calculated using shape centroids.

    Cluster shapes within a defined distance based on centroids.

    Args:
        df (gpd.GeoDataFrame): The GeoDataFrame containing shapes.
        distance (float): The distance threshold for clustering.
        check_crs (bool, optional): Check if the GeoDataFrame has a projected CRS. Defaults to False.

    Returns:
        gpd.GeoDataFrame: The GeoDataFrame with cluster labels.
    """
    if check_crs:
        assert (
            df.crs.is_projected
        ), "geodf should be a projected crs with meters as the unit"

    centers = [p.centroid for p in df.geometry]
    centers_xy = [[c.x, c.y] for c in centers]

    cluster = AgglomerativeClustering(
        n_clusters=None,
        linkage="single",
        affinity="euclidean",
        distance_threshold=distance,
    )
    cluster.fit(centers_xy)
    df["group"] = cluster.labels_
    return df


def bbox_df(inp_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Create bounding boxes around geometries in a GeoDataFrame.

    Args:
        inp_df (gpd.GeoDataFrame): The GeoDataFrame containing geometries.

    Returns:
        gpd.GeoDataFrame: The GeoDataFrame with bounding boxes.
    """
    _bounds = inp_df.bounds

    geometry = [
        box(*i, ccw=True)
        for i in zip(_bounds.minx, _bounds.miny, _bounds.maxx, _bounds.maxy)
    ]
    return gpd.GeoDataFrame(_bounds, crs=inp_df.crs, geometry=geometry)


def get_points_within_bbox(
    bbox: gpd.GeoDataFrame, points: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Get points within a bounding box.

    Args:
        bbox (gpd.GeoDataFrame): The GeoDataFrame representing the bounding box.
        points (gpd.GeoDataFrame): The GeoDataFrame containing points.

    Returns:
        gpd.GeoDataFrame: The GeoDataFrame with points within the bounding box.
    """
    intersection = gpd.sjoin(
        points.to_crs(bbox.crs),
        bbox_df,
        op="intersects",
    )
    common_cols = intersection.columns.intersection(points.columns)
    return intersection[common_cols]


def line_midpoint_df(geom_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Calculate midpoints for line geometries in a GeoDataFrame.

    Args:
        geom_df (gpd.GeoDataFrame): The GeoDataFrame containing line geometries.

    Returns:
        gpd.GeoDataFrame: The GeoDataFrame with midpoint geometries added.
    """
    geom_df["midpoint"] = geom_df.apply(
        lambda row: midpoint(row["geometry"]),
        axis=1,
    )
    return geom_df


def create_temp_df(from_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Create an empty GeoDataFrame with the same columns and CRS as the input GeoDataFrame.

    Args:
        from_df (gpd.GeoDataFrame): The input GeoDataFrame.

    Returns:
        gpd.GeoDataFrame: An empty GeoDataFrame with matching columns and CRS.
    """
    _temp_df = from_df[0:0].copy(deep=True)
    _temp_df = _temp_df.dropna(axis=1, how="all")
    _temp_df = _temp_df.reset_index(drop=True)

    return _temp_df
