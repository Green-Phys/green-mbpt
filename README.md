[![GitHub license](https://img.shields.io/github/license/Green-Phys/green-mbpt?cacheSeconds=3600&color=informational&label=License)](./LICENSE)
[![GitHub license](https://img.shields.io/badge/C%2B%2B-17-blue)](https://en.cppreference.com/w/cpp/compiler_support/17)
[![DOI](https://zenodo.org/badge/699493450.svg)](https://zenodo.org/doi/10.5281/zenodo.10071545)

![grids](https://github.com/Green-Phys/green-mbpt/actions/workflows/test.yaml/badge.svg)
[![codecov](https://codecov.io/gh/Green-Phys/green-mbpt/graph/badge.svg?token=ZHN38G4O5U)](https://codecov.io/gh/Green-Phys/green-mbpt)

```
 █▀▀█ █▀▀█ █▀▀ █▀▀ █▀▀▄
 █ ▄▄ █▄▄▀ █▀▀ █▀▀ █  █
 █▄▄█ ▀ ▀▀ ▀▀▀ ▀▀▀ ▀  ▀

 █   █ █▀▀ █▀▀█ █ █     █▀▀█ █▀▀█ █  █ █▀▀█ █    ▀  █▀▀▄ █▀▀▀
 █ █ █ █▀▀ █▄▄█ █▀▄ ▀▀  █    █  █ █  █ █  █ █   ▀█▀ █  █ █ ▀█
 █▄▀▄█ ▀▀▀ ▀  ▀ ▀ ▀     █▄▄█ ▀▀▀▀  ▀▀▀ █▀▀▀ ▀▀▀ ▀▀▀ ▀  ▀ ▀▀▀▀
```
***

`Green/WeakCoupling` is a weak-coupling perturbation expansion solver for real materials expressed in Gaussian Bloch orbitals

## Installation

### Dependencies

`Green/WeakCoupling` has the following required external dependencies
  - HDF5 library version >= 1.10.2
  - Message Passing Interface >= 3.1
  - Eigen3 library >= 3.4.0
  - BLAS

To build `Green/WeakCoupling` CMake version 3.18 or above is required

`PySCF` interface requires
  - `PySCF` version >= 2.0
  - `numba` version >= 0.57

### Build and Install

The following example will build, test and install `Green/WeakCoupling` to `/path/to/weakcoupling/install/dir` directory.

```ShellSession
$ git clone https://github.com/Green-Phys/green-mbpt
$ cd green-mbpt
$ mkdir build && cd build
$ cmake -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/path/to/weakcoupling/install/dir ..
$ make
$ make test
$ make install
```

## Basic usage

### Generate input data
`Green/WeakCoupling` provides `PySCF` interface to generate input data and initial starting point through the `green-mbtools` Python package.

To generate initial mean-field solution and one- and two-body integrals run:
```ShellSession
python python/init_data_df.py --a <a.dat> --atom <atom.dat> --nk <nk> --basis <basis specification>
```

To perform weak-coupling simulations, one have to call `mbpt.exe` executable located at the installation path in the `bin` subdirectory.
Minimal parameters that are needed to run weak-coupling simulations are following:

- `--BETA`  inverse temperature
- `--scf_type` type of self-consistent approximation, should be either `GW`, `GF2` or `HF`
- `--grid_file`  path to a file containing non-uniform grids, program will check three possible locations:
    - current directory or absolute path
    - `<installation directory>/share`
    - build directory of weak-coupling code

Currently, we provide IR (`ir` subdirectory) and Chebyshev grids (`cheb` subdirectory) for nonuniform imaginary time representation.

After succesful completetion results will be written to a file located at `--results_file` (by default set to `sim.h5`)
To get information about other parameters and their default values call `mbpt.exe --help`.

## Acknowledgements

This work is supported by the National Science Foundation under the award OAC-2310582
