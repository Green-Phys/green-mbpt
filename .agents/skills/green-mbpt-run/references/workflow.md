# GREEN workflow

GREEN spans several repositories. Use the owning component rather than copying
its behavior into a neighboring repository.

1. `green-mbtools` prepares supported mean-field data, integrals, symmetry
   metadata, and `input.h5` inputs.
2. `green-mbpt` provides `mbpt.exe` for HF, GF2, and GW calculations and
   `embedding.exe` for documented embedding workflows.
3. `green-mbpt` writes its main results to the configured results file
   (`sim.h5` by default).
4. `green-mbtools` and the documented continuation tools provide supported
   post-processing paths.

Canonical guides:

- <https://green-phys.org/docs/getting-started/preparing_input/>
- <https://green-phys.org/docs/getting-started/running_greens/>
- <https://green-phys.org/docs/getting-started/postprocessing/>
- <https://green-phys.org/docs/getting-started/examples/>

Before producing a command, establish the requested Green release and read the
matching CLI help. Options and file formats evolve across releases.

