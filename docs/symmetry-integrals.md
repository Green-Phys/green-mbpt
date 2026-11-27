# Opt-in space-group DF integrals

This branch reads `green.df.space_group.v1` archives through the shared
`green-symmetry` host implementation. Requested factors are reconstructed in
the producer's captured auxiliary frame. The three tensor legs, affine Bloch
phases, TR/exchange chain and optional square orbital basis maps are applied
before contractions. The one-body and q-star weights retain their existing
meaning. Legacy files and default dispatch continue to work without new flags.

The consumer requires the `green-symmetry` integral archive API. GPU use also
requires the SG-aware `green-gpu` reader. Dependency repositories default to
the official upstream repositories and revisions follow `GREEN_RELEASE`.
CMake cache variables
`GREEN_SYMMETRY_GIT_REPOSITORY`, `GREEN_SYMMETRY_GIT_TAG`,
`GREEN_GPU_GIT_REPOSITORY` and `GREEN_GPU_GIT_TAG` allow explicit overrides.
To test coordinated source checkouts:

```bash
cmake -S . -B build \
  -DFETCHCONTENT_SOURCE_DIR_GREEN-SYMMETRY=/path/to/green-symmetry \
  -DFETCHCONTENT_SOURCE_DIR_GREEN-GPU=/path/to/green-gpu \
  -DCUSTOM_KERNELS=https://github.com/Green-Phys/green-gpu.git
```

For a CPU-only build, omit the last two options. Both producer and consumers
must understand the archive schema. Archives intentionally omit legacy chunk
aliases. The one-body input follows the upstream native-complex format 1.1.0
or newer; migrate older inputs with `python/migrate.py` into a new file.

From a fresh result directory, use the same one-body input and numerical
settings as the matching legacy calculation, adding:

```bash
mbpt.exe --input_file /path/to/input.h5 \
  --dfintegral_hf_file /path/to/hf_sg \
  --dfintegral_file /path/to/correlation_sg \
  --integral_symmetry space_group \
  --scf_type GW --kernel CPU --BETA 10 \
  --grid_file /path/to/ir/1e4.h5 --itermax 2 --restart false \
  --results_file result.h5
```

CPU HF/GW/GF2 and GPU HF/GW are covered. `GF2 --kernel GPU` uses GPU HF and
CPU GF2 correlation. No GPU GF2 correlation implementation is added.
For GPU use, select `--kernel GPU`, set `--cuda_low_cpu_memory` and
`--cuda_low_gpu_memory` explicitly, and choose the precision flags below.
The existing GPU precision behavior uses single precision when both `P_sp`
and `Sigma_sp` are true. Mixed flag combinations retain existing behavior.

Each reader has a byte-bounded transform/metric cache (64 MiB by default;
GF2 owns six readers). Set `--integral_symmetry_cache_bytes N` to change it.
`--integral_symmetry_preload_bytes N` bounds GPU representative host preload
per node (1 GiB by default), before allocation. Only representatives are
preloaded. Shared buffers remain read-only during owned reconstruction;
target-Q slices sum all source-Q rows. Individual factor/chunk buffers have
a 128 MiB bound. Descriptor memory and active output tensors are additional.

The initial profile requires scalar spherical 3D cells, positive full-rank
Cholesky metrics, and square orbital maps. Truncated/negative/ED gauges,
X2C/spinors, range-separated physical interactions, `df_ewald.h5`/`AqQ.h5`
special sidecars, special finite-size flags, coarse graining and GW q=0
extrapolation are rejected explicitly for SG. Reduced-q GW input must identify
its actual correlation gauge. Ordinary HF and default GREEN legacy Ewald
q=0 correlation sets have separate captured frames.

With `BUILD_TESTING=ON`, `test/frozen_integral_probe` runs production kernels
once on an externally supplied arbitrary complex `G_tau` tensor in
interleaved float64 `(tau,spin,k,i,j,2)` layout. It requires unreduced one-body
k points and checks that G is unchanged. Use the solver flags above plus
`--frozen_G_file /path/to/frozen_G.h5 --probe_output fresh-result.h5`.
Compare `Sigma1` and `Selfenergy` with a separately generated complete reference.
This probe does not iterate self-consistency. Shared library CTests register
format, reconstruction, ownership and preload checks; optional physical
fixtures use `GREEN_SG_VALIDATION_ROOT`. Explicit `GREEN_SG_HF_FIXTURE_DIR`
and `GREEN_SG_CORRELATION_FIXTURE_DIR` select fresh ordinary and corrected
fixtures, each containing `sg`, `native-reference.h5` and `native-input.h5`.
The producer's opt-in physical regression generates these using complete
references; its corrected reference calls the original GREEN routine directly.
