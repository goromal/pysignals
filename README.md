# pysignals

![example workflow](https://github.com/goromal/pysignals/actions/workflows/test.yml/badge.svg)

Python bindings for the [signals-cpp](https://github.com/goromal/signals-cpp) library.

**Under construction**

## Building / Installing

This library is built with CMake. Most recently tested with the following dependencies:

- Pybind11
- Eigen 3.4.0
- [signals-cpp](https://github.com/goromal/signals-cpp)
- [manif-geom-cpp](https://github.com/goromal/manif-geom-cpp)

```bash
mkdir build
cd build
cmake ..
make # or make install
```

Pytest unit tests are present in the `tests/` folder.
