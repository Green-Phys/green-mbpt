import argparse
import sys

import green_mbtools.mint as pymb
import h5py
import numpy as np
from green_mbtools.mint import ortho_utils as ou
from green_mbtools.pesto import orth


ORTH_MODE_MAP = {
    "symmetrical_orbitals": "symmetric_lowdin",
    "canonical_orbitals":   "lowdin",
    "molecular_orbitals":   "mo",
    "natural_orbitals":     "natural",
}


def kspace_to_transform_h5_storage(X_k_new, X_inv_k_new):
    """Convert build_X_kspace_from_ao_reps output to the convention
    transform.h5 consumers (itransform.cpp + the rest of this script) use.

    build_X_kspace_from_ao_reps returns ``(X, X_inv)`` in the
    ``X S X† = I`` convention: ``X`` is ``(n_ortho, n)``, ``X_inv`` is
    ``(n, n_ortho)``. transform.h5 wants the conjugate-transposed pair
    ``X = X.conj().T`` (shape ``(n, n_ortho)``) and
    ``X_inv = X_inv.conj().T`` (shape ``(n_ortho, n)``), tuple-ordered
    ``(X_inv, X)``.
    """
    X_out = X_k_new.conj().swapaxes(1, 2)
    X_inv_out = X_inv_k_new.conj().swapaxes(1, 2)
    return X_inv_out, X_out


def load_kspace_symmetry(input_file):
    """Read the symmetry/k arrays needed by build_X_kspace_from_ao_reps.

    Raises if the input.h5 lacks the symmetry/k group entirely —
    regenerate with the current green-mbtools mean-field driver. The
    group's individual datasets are read without further guards on the
    assumption that current mbtools writes them as a complete set; a
    finer-grained schema check via __green_version__ is a TODO.
    """
    with h5py.File(input_file, "r") as f:
        symm = f.get("symmetry/k")
        if symm is None:
            raise RuntimeError(
                f"{input_file}: symmetry/k group not found. "
                "Regenerate input.h5 with the current green-mbtools "
                "mean-field driver."
            )
        ibz2bz = symm["ibz2bz"][()]
        bz2ibz = symm["bz2ibz"][()]
        k_sym_ao = symm["k_sym_transform_ao"][()]
        tr_conj = symm["tr_conj"][()] if "tr_conj" in symm else None
    if not np.iscomplexobj(k_sym_ao):
        k_sym_ao = k_sym_ao.astype(np.complex128)
    return {
        "ibz2bz": np.asarray(ibz2bz),
        "bz2ibz": np.asarray(bz2ibz),
        "k_sym_transform_ao": k_sym_ao,
        "tr_conj": np.asarray(tr_conj, dtype=bool) if tr_conj is not None else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SEET pre-processing")
    parser.add_argument("--orth", type=lambda x: (str(x).lower() in ['true','1', 'yes']), default='true', help="Apply basis orthogonalization.")
    parser.add_argument("--input_file", type=str, default="input.h5", help="Input file name.")
    parser.add_argument("--gf2_input_file", type=str, default="sim.h5", help="Converged GF2 simpulation result file.")
    parser.add_argument("--transform_file", type=str, default="transform.h5", help="Input file name.")
    parser.add_argument("--active_space", type=int, nargs='+', action='append', help="Input file name.", required=True)
    parser.add_argument("--orth_method",
                        choices=["symmetrical_orbitals",
                                 "canonical_orbitals",
                                 "molecular_orbitals",
                                 "natural_orbitals"],
                        default="natural_orbitals",
                        help="Type of the orthogonalization.")
    parser.add_argument("--tau_grid", choices=["even", "ir"], default="ir", help="type of tau grid for impurity solver.")
    parser.add_argument("--n_tau", type=int, default=1001, help="number of tau points for even grid.")
    parser.add_argument("--ir_file", type=str, default="1e4.h5", help="IR lambda parameter for even tau grid transformation.")

    args = parser.parse_args()
    with h5py.File(args.input_file, "r") as fff:
        nao = fff["params/nao"][()]
        nso = fff["params/nso"][()]
        ns = fff["params/ns"][()]
        x2c = (nso == nao * 2)
        if x2c and args.orth and args.orth_method not in (
                "symmetrical_orbitals", "canonical_orbitals"):
            raise RuntimeError(
                "ortho not supported for 2-component / x2c1e calculations "
                "with --orth_method={!r}; allowed methods are "
                "symmetrical_orbitals and canonical_orbitals (Löwdin "
                "variants; MO / natural rotations would have non-block-"
                "diagonal X in the spinor basis).".format(args.orth_method)
            )

    sys.stdout.write("Reading the input data...")
    sys.stdout.flush()

    seet = pymb.seet_init(args)

    F, S, T, dm, dm_s, kmesh, kmesh_sc = seet.get_input_data()
    print("done!")

    if args.orth:
        sys.stdout.write("Constructing orthogonalization basis...")
        sys.stdout.flush()

        symm = load_kspace_symmetry(args.input_file)
        mode = ORTH_MODE_MAP[args.orth_method]
        ibz2bz = symm["ibz2bz"]
        bz2ibz = symm["bz2ibz"]
        k_sym_ao = symm["k_sym_transform_ao"]
        tr_conj = symm["tr_conj"]

        # Slice IBZ quantities from the full-BZ arrays.
        S_ibz = S[0][ibz2bz]
        if mode == "natural":
            F_avg_full = 0.5 * (F[0] + F[1]) if ns == 2 else F[0]
            F_ibz = F_avg_full[ibz2bz]
            dm_ibz = dm[ibz2bz]
        elif mode == "mo":
            F_avg_full = 0.5 * (F[0] + F[1]) if ns == 2 else F[0]
            F_ibz = F_avg_full[ibz2bz]
            dm_ibz = None
        else:  # lowdin or symmetric_lowdin
            F_ibz = None
            dm_ibz = None

        X_k_new, X_inv_k_new = ou.build_X_kspace_from_ao_reps(
            mode,
            S_ibz, ibz2bz, bz2ibz, k_sym_ao,
            tr_conj=tr_conj,
            F_ibz=F_ibz, dm_ibz=dm_ibz,
        )

        X_inv_k, X_k = kspace_to_transform_h5_storage(X_k_new, X_inv_k_new)
        X_k = list(X_k)
        X_inv_k = list(X_inv_k)

        if nso == nao * 2:
            X_ERI_k = np.array(X_k)[:, :nao, :nao].copy()
        else:
            X_ERI_k = np.array(X_k)[:, :, :].copy()
        print("done!")

    if args.orth:
        sys.stdout.write("Transforming into the orthogonal basis...")
        sys.stdout.flush()
        for s in range(F.shape[0]):
            F[s] = orth.transform(F[s], X_k,X_inv_k)
            S[s] = orth.transform(S[s], X_k,X_inv_k)
            T[s] = orth.transform(T[s], X_k,X_inv_k)
            dm_s[s] = orth.transform(dm_s[s], [x_inv_k.conj().T for x_inv_k in X_inv_k], [x_k.conj().T for x_k in X_k])
        dm = orth.transform(dm, [x_inv_k.conj().T for x_inv_k in X_inv_k],[x_k.conj().T for x_k in X_k])
        if ns == 2:
            dmmmm = np.einsum("kij->ij", dm) / dm.shape[0]
            # check that local DM is pure real
            assert(np.allclose(dmmmm.imag, np.zeros(dm[0].shape)))
        for ik in range(dm.shape[0]):
            for s in range(F.shape[0]):
                assert (np.allclose(S[s, ik], np.eye(S.shape[2]),atol=1e-7))
        print("done!")

    print("Constructing the correlated subspace...")

    print(args.active_space)

    UUs = []
    UU_ERIs = []

    for i in args.active_space:
        imp = np.array(i)
        UU = np.zeros((imp.shape[0], nao))
        print("Active space orbitals", imp)
        for ii, orb in enumerate(imp):
            UU[ii, orb] = 1
        UU_ERIs.append(UU)
        if x2c:
            UU = np.zeros((imp.shape[0]*2, nso))
            for ii, orb in enumerate(imp):
                UU[ii, orb] = 1
                UU[ii+imp.shape[0], orb + nao] = 1
            UUs.append(UU)
        else:
            UUs.append(UU)

    print("Done!")

    if args.orth:
        with h5py.File(args.transform_file, "w") as tfile:
            tfile["X_k"]     = np.array(X_k)
            tfile["X_ERI_k"] = np.array(X_ERI_k)
            tfile["X_inv_k"] = np.array(X_inv_k)
            tfile["nimp"] = len(args.active_space)
            for i in range(len(UUs)):
                tfile["{}/UU_ERI".format(i)] = UU_ERIs[i].view(np.float64)
                tfile["{}/UU".format(i)] = UUs[i].view(np.float64)
    else:
        with h5py.File(args.transform_file, "w") as tfile:
            X_k = np.array([np.eye(F.shape[2], dtype=np.complex128)]*kmesh.shape[0])
            UU  = np.eye(F.shape[2], dtype=np.float64)
            tfile["X_k"] = np.array(X_k)
            tfile["X_ERI_k"] = np.array(X_ERI_k)
            tfile["X_inv_k"] = np.array(X_k)
            tfile["nimp"] = len(args.active_space)
            for i in range(len(UUs)):
                tfile["{}/UU_ERI".format(i)] = UU.view(np.float64)
                tfile["{}/UU".format(i)] = UU.view(np.float64)
    print("Done")

    if args.tau_grid == "even":
        from green_grids.repn import ir as green_grids_ir
        print("Generating even tau grid transformer with {} points for impurity solver.".format(args.n_tau))
        # Load IR file data
        with h5py.File(args.ir_file, "r") as irf:
            ir_lambda = irf['fermi/metadata/lambda'][()]
            ir_ncoeff = irf['fermi/metadata/ncoeff'][()]
        sparse_ir = green_grids_ir.Basis(ir_lambda, ir_ncoeff, 'fermi', trim=True)
        x_grid = np.linspace(-1.0, 1.0, args.n_tau)
        uxl_even = np.array([sparse_ir._uxl(None, x) for x in x_grid])
        with h5py.File(args.transform_file, "a") as tfile:
            tfile["to_even_tau"] = uxl_even
        print("Done!")
