# histnd
A python module that computes multi-dimensional histogram.

The serial version should be faster that [numpy.histogramdd](https://numpy.org/doc/stable/reference/generated/numpy.histogram.html) when sample scale is small.
The parallel version should be faster that [numpy.histogramdd](https://numpy.org/doc/stable/reference/generated/numpy.histogram.html) when sample scale is large.

## Installation
```
cd path/to/this/repo
pip install .
```

## TODO list
- [ ] update documentation
- [ ] implementation of histnd_parallel_py() with generic data types of Rust
- [ ] benchmarks

## Examples
Python example scripts are located [here](examples/).
