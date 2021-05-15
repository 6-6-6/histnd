from setuptools import setup
from setuptools_rust import RustExtension

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    # TODO: read metadata from Cargo.toml
    name="histnd",
    version="0.1.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Rust",
        "Operating System :: POSIX",
        "Operating System :: MacOS :: MacOS X",
    ],
    packages=["histnd"],
    rust_extensions=[RustExtension("histnd.histnd")],
    include_package_data=True,
    zip_safe=False,
)