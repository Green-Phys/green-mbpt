#!/usr/bin/env python3
"""Generate H2 / cc-pvdz test fixtures for the orth basis-invariance
test cases in test/solvers_test.cpp (`SECTION("HF orth invariance ...")`
and the GW / GF2 siblings).

Layout produced under this directory:

    none/    input.h5  df_hf_int/  data.h5
    lowdin/  input.h5  df_hf_int/  data.h5
    mo/      input.h5  df_hf_int/  data.h5
    natural/ input.h5  df_hf_int/  data.h5

The same physical H2 / cc-pvdz system, written into four on-disk
bases (``--orth = none | lowdin | mo | natural``). ``input.h5`` + ``df_hf_int/``
come from green-mbtools' ``init_data_mol_df.py``. ``data.h5`` carries
the post-Dyson ``iter1/G_tau/data`` from one HF iteration of
``mbpt.exe`` and supplies the basis-correct density that
``mbpt::compute_energy`` needs when the C++ test recomputes the
energies. (A naive ``G_tau[β] = -dm[input.h5]`` is NOT enough — see
the feat(integrals) commit message in green-mbtools.)

For each scf type (HF / GW / GF2) the C++ test runs one solver
iteration in each mode, calls ``mbpt::compute_energy``, and asserts
the returned ``(e_1b, e_HF, e_corr)`` triple agrees across modes to
``1e-10``. Before the ``feat(integrals)`` fix in green-mbtools, lowdin
and mo drifted from ``none`` by ~0.1-0.2 Ha on ``e_HF``.

Run from this directory with the green-mbtools dev tree on PYTHONPATH
and ``mbpt.exe`` built. CI does NOT run this script — it consumes the
committed fixtures. Re-run only when the integral pipeline, the orth
convention, or the data layout intentionally changes.

    cd green-mbpt/test/data/HF_orth_invariance
    PYTHONPATH=/path/to/green-mbtools python3 generate.py \\
        /path/to/green-mbpt/build/mbpt.exe \\
        /path/to/green-grids/data/ir/1e4.h5
"""

import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
INIT_SCRIPT = os.path.abspath(
    os.path.join(ROOT, "..", "..", "..", "python", "init_data_mol_df.py")
)

MODES = ["none", "lowdin", "mo", "natural"]
ATOM = "H 0 0 0; H 0 0 0.74"
BASIS = ["H", "cc-pvdz"]
AUXBASIS = ["H", "cc-pvdz-jkfit"]


def init_mbtools(mode):
    d = os.path.join(ROOT, mode)
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)
    os.chdir(d)
    cmd = [
        sys.executable, INIT_SCRIPT,
        "--atom", ATOM,
        "--basis", *BASIS,
        "--auxbasis", *AUXBASIS,
        "--orth", mode,
        "--output_path", "input.h5",
        "--df_int", "1",
        "--hf_int_path", "df_hf_int",
        "--int_path", "df_int",
    ]
    subprocess.run(cmd, check=True)
    # Drop transient products. dm.h5 is unused (iter1/G_tau in data.h5
    # supplies the density via -G[β]); cderi_mol.h5 / tmp.chk are pyscf
    # scratch; df_int is the empty correlated-integrals dir (the
    # molecular path only fills df_hf_int).
    for f in ["cderi_mol.h5", "tmp.chk", "dm.h5"]:
        if os.path.exists(f):
            os.remove(f)
    if os.path.isdir("df_int"):
        shutil.rmtree("df_int")
    os.chdir(ROOT)


def init_data_h5(mode, mbpt_exe, grid_file):
    """Run one HF iteration to produce data.h5 with the post-Dyson G_tau."""
    d = os.path.join(ROOT, mode)
    os.chdir(d)
    for f in ["data.h5", "sim.h5"]:
        if os.path.exists(f):
            os.remove(f)
    cmd = [
        mbpt_exe,
        "--BETA", "100",
        "--grid_file", grid_file,
        "--itermax", "1",
        "--scf_type", "HF",
        "--input_file", "input.h5",
        "--dfintegral_hf_file", "df_hf_int",
        "--dfintegral_file", "df_hf_int",   # molecular: same integrals serve HF + correlation
        "--results_file", "data.h5",
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    # mbpt.exe also drops sim.h5 alongside the explicitly-named
    # results_file; remove that duplicate.
    if os.path.exists("sim.h5"):
        os.remove("sim.h5")
    os.chdir(ROOT)


def main():
    if len(sys.argv) != 3:
        sys.stderr.write(
            "usage: generate.py <path/to/mbpt.exe> <path/to/ir/1e4.h5>\n"
        )
        sys.exit(2)
    mbpt_exe = os.path.abspath(sys.argv[1])
    grid_file = os.path.abspath(sys.argv[2])
    if not os.path.isfile(mbpt_exe):
        sys.stderr.write(f"mbpt.exe not found: {mbpt_exe}\n")
        sys.exit(2)
    if not os.path.isfile(grid_file):
        sys.stderr.write(f"grid file not found: {grid_file}\n")
        sys.exit(2)

    for mode in MODES:
        print(f"=== {mode}: mbtools init ===")
        init_mbtools(mode)
    for mode in MODES:
        print(f"=== {mode}: mbpt iter1 ===")
        init_data_h5(mode, mbpt_exe, grid_file)
    print("Test data ready under:", ROOT)


if __name__ == "__main__":
    main()
