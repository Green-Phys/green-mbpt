# Build and release selection

Prefer compatible released components. Do not combine a released `green-mbpt`
with arbitrary component development branches unless the user is intentionally
testing that combination.

The canonical source-build guide, including current prerequisites, CPU/GPU
configuration, release examples, and supported HPC notes, is:

<https://green-phys.org/docs/installation/from_sources/>

A normal CPU build uses an out-of-source CMake tree:

```bash
cmake -S green-mbpt -B green-mbpt-build \
  -DCMAKE_INSTALL_PREFIX=/path/to/install \
  -DCMAKE_BUILD_TYPE=Release
cmake --build green-mbpt-build -j 4
(cd green-mbpt-build && ctest --output-on-failure)
cmake --install green-mbpt-build
```

The first configure may require network access because CMake fetches component
libraries. Do not claim GPU support merely because configuration succeeds; use
the current guide and the selected release's tests to establish support.

