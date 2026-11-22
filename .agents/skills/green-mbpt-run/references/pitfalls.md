# Pitfalls and decision boundaries

- Do not mix component versions based only on branch names. Select a documented
  released combination or state that development code is being tested.
- Do not use a legacy repository script merely because an old README or example
  names it. Start from the current `green-mbtools` and Green website workflow.
- `HF`, `GF2`, `GW`, embedding choices, inverse temperature, basis and
  pseudopotential selection, grids, convergence thresholds, and continuation
  methods are scientific inputs. Ask the user or cite an accepted workflow.
- A process exit without an error is not proof of physical convergence or a
  valid scientific result. Use the documented convergence and output criteria.
- MPI rank count, threading, math libraries, and CPU/GPU kernels can affect
  reproducibility and performance. Record them when comparisons matter.
- Do not interpret HDF5 paths from names alone. Use the versioned data contract.
- Do not silently repair incompatible inputs. Use the supported migration or
  regenerate them with the matching `green-mbtools` release.
- When current code, `--help`, tests, and published documentation conflict,
  report the mismatch and prefer current implementation evidence.

