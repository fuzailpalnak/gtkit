

# GTKit - GIS Toolkit
[![Downloads](https://static.pepy.tech/badge/gtkit)](https://pepy.tech/project/gtkit)

GTKit (GIS Toolkit) is a library designed to streamline GIS 
(Geographic Information System) releated tasks. 
Whether it is geometry, GeoDataFrames, images, or mathematical operations,
GTKit provides a set of commonly used methods and operations to simplify your workflow.
This utility has been curated to include some of the regularly used methods and operations that I, frequently employ.

## Features

GTKit offers a range of functionalities to assist you in working with GIS data effectively:

- **Geometry Operations**: Perform various geometry-based operations such as splitting a LineString, line referencing,
interpolation, extrapolation

- **GeoDataFrame Manipulation**
- **Image Processing**: Convert images to geometries, Generate Geometries from binary geo referenced mask
- **Mesh Creation**: Create Mesh either using images or using LineString.

## Installation with pip

External dependencies

- *_Geopandas - [installation](https://anaconda.org/conda-forge/geopandas)_*
- *_Rasterio - [installation](https://anaconda.org/conda-forge/rasterio)_*
- *_GDAL - [installation](https://anaconda.org/conda-forge/gdal)_*
- *_Fiona -  [installation](https://anaconda.org/conda-forge/fiona)_*
- *_Shapely -  [installation](https://anaconda.org/conda-forge/shapely)_*

1. Create conda env

```bash
conda env create --name gtkit --file=env.yml
conda activate gtkit
```

2. Run:

```bash
pip install gtkit
```


## Usage

Import GTKit in your Python script or notebook:

```python
import gtkit
```

GTKit's modular structure allows you to import specific functionalities as needed for your project.

### Examples

<table>
    <tr>
        <th>A.</th>
        <td>
            <a href="tutorials/shpToBitmap.ipynb">Generate Bitmap From Shp</a>.</br>
            <a href="tutorials/bitmapToShp.ipynb">Generate Shp From Bitmap</a>.
        </td>
        <td>
            <img src="https://github.com/fuzailpalnak/gtkit/assets/24665570/880d07c9-3d77-448a-99a7-9fd1b6d873fb" alt="alt text" width="256" height="256">
        </td>
    </tr>
    <tr>
        <th>B.</th>
        <td>
            <a href="tutorials/shp2Mesh.ipynb">Generate mesh around a line string</a>.
        </td>
        <td>
            <img src="https://github.com/fuzailpalnak/gtkit/assets/24665570/050e2df5-79c3-4d65-ad8a-d19d0ee9feb8" alt="alt text" width="256" height="256">
        </td>
    </tr>
    <tr>
        <th>C.</th>
        <td>
            <a href="tutorials/lineOps.ipynb">Some geometry Operations</a>.
        </td>
        <td>
            <img src="https://github.com/fuzailpalnak/gtkit/assets/24665570/4b1ff8ae-6e62-4d92-b7fa-694591ea05f9" alt="alt text" width="256" height="256">
        </td>
    </tr>
</table>

### Stitch And Split Geo Reference Image
```python
import numpy as np

from gtkit.imgops import georead, geowrite
from gtkit.imgops import StitchNSplitGeo

sns = StitchNSplitGeo(split_size=(256, 256, 3), img_size=(1500, 1500, 3))
image = georead(r"22978945_15.tiff")

stitched_image = np.zeros((1500, 1500, 3))
for win_number, window in sns:
    split_image, meta = sns.split(image, window)
    # ....Processing on image
    stitched_image = sns.stitch(split_image, stitched_image, window)

geowrite(
    save_path=r"new.tiff",
    image=stitched_image,
    transform=image.transform,
    crs=image.crs,
)
```
## Documentation

For detailed information on available methods, classes, and their usage, refer to the [GTKit Documentation](https://gtkit.readthedocs.io).

[comment]: <> (## Contributing)

[comment]: <> (We welcome contributions from the GIS community! If you'd like to contribute to GTKit, please refer to our [Contribution Guidelines]&#40;https://gtkit-docs.example.com/contributing&#41; for more information.)

[comment]: <> (## License)

[comment]: <> (GTKit is released under the [MIT License]&#40;https://opensource.org/licenses/MIT&#41;.)

[comment]: <> (## Contact)

[comment]: <> (Have questions or suggestions? Feel free to contact us at `contact@example.com`.)



