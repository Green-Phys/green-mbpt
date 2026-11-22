# AGENTS.md

Guidance for coding agents working in `green-mbpt`.

## Sources of truth

- Current code, tests, and CI define implemented behavior.
- User documentation is maintained at <https://green-phys.org/docs/>.
- The Green 1.x HDF5 contracts are documented under
  <https://green-phys.org/docs/components/data-formats/>. Do not infer dataset
  meanings, shapes, units, spin conventions, or compatibility from names alone.
- `green-mbtools` owns supported input generation and post-processing workflows;
  `green-mbpt` owns the weak-coupling and embedding executables.
- When prose and implementation disagree, verify against tests and CLI parameter
  definitions, then update the stale documentation rather than changing code to
  restore an obsolete description.

## Build and test

Use an out-of-source build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
ctest --test-dir build --output-on-failure
```

The first configure may fetch Green component libraries and Catch2. CI is the
authority for supported Linux and macOS environments. Keep C++17 compatibility.

## Change discipline

- Preserve explicit MPI collective behavior and rank ownership.
- Treat HDF5 paths, version attributes, array dimensions, symmetry metadata,
  spin layout, and restart state as public data contracts.
- Numerical tests should check a known value, identity, symmetry, conservation
  law, or reference result; shape-only and finite-only checks are insufficient
  for a changed scientific path.
- Changes to public CLI options, component/release compatibility, input or
  output formats, or supported workflows must update the relevant page on
  `green-phys.org` and the bundled `green-mbpt-run` skill.
- Do not silently select a scientific method, convergence criterion, basis,
  grid, analytic-continuation method, or migration path on a user's behalf.
- Preserve existing copyright headers and document external scientific or code
  provenance.

## Agent-facing surfaces

- `.agents/skills/green-mbpt-run/` guides downstream use of GREEN.
- `scripts/check-agent-docs.py` checks the skill layout and its required links.
- Repository contribution guidance is at
  <https://green-phys.org/docs/contribution/>.

