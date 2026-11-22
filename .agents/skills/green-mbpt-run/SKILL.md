---
name: green-mbpt-run
description: Prepare, run, restart, and inspect supported GREEN weak-coupling calculations using green-mbtools and green-mbpt. Use for HF, GF2, GW, and documented embedding workflows; do not choose scientific settings or interpret results beyond the documented contracts without user input.
---

# Run GREEN calculations

Use this skill for downstream workflows, not for changing `green-mbpt` itself.

## Fast path

1. Identify the requested method, system, target release, platform, and whether
   CPU, GPU, or embedding support is required. Ask when a scientific choice is
   missing; do not silently choose one.
2. Use mutually compatible released Green components. Prefer a released tag
   over a development branch unless the user explicitly requests development
   code.
3. Generate inputs through the documented `green-mbtools` interface. Do not
   fabricate or manually translate HDF5 symmetry or version metadata.
4. Build `green-mbpt`, inspect `mbpt.exe --help` for the selected version, and
   run through the site's documented workflow.
5. Treat completion, convergence, restart safety, and scientific validity as
   separate questions. Locate outputs using the documented HDF5 contract.
6. Before results are reported or published, identify the GREEN software,
   version, and method publications that apply. Use the canonical citation
   guidance rather than guessing from method names, and surface unresolved
   citation choices for the researcher.

## Read only what the task needs

| Need | Reference |
| --- | --- |
| Component ownership and end-to-end workflow | [workflow](references/workflow.md) |
| Releases, dependencies, CPU/GPU build | [build and releases](references/build-and-releases.md) |
| Input/output structure and compatibility | [data contracts](references/data-contracts.md) |
| Common wrong turns and human decisions | [pitfalls](references/pitfalls.md) |
| Software, version, and method acknowledgment | [citations](references/citations.md) |

The machine-oriented documentation index is
<https://green-phys.org/llms.txt>. When this skill and the current executable or
tests disagree, the implementation wins; report the documentation drift.

