import os
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

current_dir = os.path.abspath(os.path.dirname(__file__))
try:
    with open(os.path.join(current_dir, "requirements.txt"), encoding="utf-8") as f:
        install_requires = f.read().split("\n")
except FileNotFoundError:
    install_requires = []


setup(
    name="gtkit",
    version="0.1.1",
    author="Fuzail Palnak",
    author_email="fuzailpalnak@gmail.com",
    url="https://github.com/fuzailpalnak/gtkit",
    description="A library designed to streamline GIS (Geographic Information System) related tasks."
    " Whether it is geometry, GeoDataFrames, images, or mathematical operations, "
    "GTKit provides a set of commonly used methods and operations to simplify your workflow.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=install_requires,
    keywords=[
        "Gis utils",
        "Gis Operations",
        "Raster Operations",
        "Geometry Operations",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
    ],
)
