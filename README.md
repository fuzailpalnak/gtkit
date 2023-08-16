

# GTKit - GIS Toolkit

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

[comment]: <> (## Installation)

[comment]: <> (You can install GTKit using `pip`:)

[comment]: <> (```bash)

[comment]: <> (pip install gtkit)

[comment]: <> (```)

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
            <img src="https://github.com/fuzailpalnak/gtkit/assets/24665570/c6927658-2911-49fd-ab57-e387a6554513" alt="alt text" width="256" height="256">
        </td>
    </tr>
    <tr>
        <th>B.</th>
        <td>
            <a href="tutorials/shp2Mesh.ipynb">Generate mesh around a line string</a>.
        </td>
        <td>
            <img src="https://github.com/fuzailpalnak/gtkit/assets/24665570/ea348f2b-89a0-41aa-af42-0c21e7ab6c64" alt="alt text" width="256" height="256">
        </td>
    </tr>
    <tr>
        <th>C.</th>
        <td>
            <a href="tutorials/lineOps.ipynb">Some geometry Operations</a>.
        </td>
        <td>
            <img src="https://github.com/fuzailpalnak/gtkit/assets/24665570/917955bd-36d9-4ac0-9f98-92d36924b8ba" alt="alt text" width="256" height="256">
        </td>
    </tr>
</table>

[comment]: <> (## Documentation)

[comment]: <> (For detailed information on available methods, classes, and their usage, refer to the [GTKit Documentation]&#40;https://gtkit-docs.example.com&#41;.)

[comment]: <> (## Contributing)

[comment]: <> (We welcome contributions from the GIS community! If you'd like to contribute to GTKit, please refer to our [Contribution Guidelines]&#40;https://gtkit-docs.example.com/contributing&#41; for more information.)

[comment]: <> (## License)

[comment]: <> (GTKit is released under the [MIT License]&#40;https://opensource.org/licenses/MIT&#41;.)

[comment]: <> (## Contact)

[comment]: <> (Have questions or suggestions? Feel free to contact us at `contact@example.com`.)



