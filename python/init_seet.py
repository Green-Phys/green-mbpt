import argparse
import sys

import green_mbtools.mint as pymb
import h5py
import numpy as np
from green_mbtools.mint import ortho_utils as ou
from green_mbtools.pesto import orth


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SEET pre-processing")
    parser.add_argument("--orth", type=lambda x: (str(x).lower() in ['true','1', 'yes']), default='true', help="Apply basis orthogonalization.")
    parser.add_argument("--input_file", type=str, default="input.h5", help="Input file name.")
    parser.add_argument("--gf2_input_file", type=str, default="sim.h5", help="Converged GF2 simpulation result file.")
    parser.add_argument("--transform_file", type=str, default="transform.h5", help="Input file name.")
    parser.add_argument("--active_space", type=int, nargs='+', action='append', help="Input file name.", required=True)
    parser.add_argument("--orth_method", choices=["natural_orbitals", "canonical_orbitals", "symmetrical_orbitals"],
                        default="natural_orbitals", help="Type of the orthogonalization.")
    parser.add_argument("--from_ibz", type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default='false',
                        help="Input data is in the reduced BZ.")
    parser.add_argument("--tau_grid", choices=["even", "ir"], default="ir", help="type of tau grid for impurity solver.")
    parser.add_argument("--n_tau", type=int, default=1001, help="number of tau points for even grid.")
    parser.add_argument("--ir_file", type=str, default="1e4.h5", help="IR lambda parameter for even tau grid transformation.")

    args = parser.parse_args()
    with h5py.File(args.input_file, "r") as fff:
        nao = fff["params/nao"][()]
        nso = fff["params/nso"][()]
        ns  = fff["params/ns"][()]
        if nso == nao * 2 and args.orth_method != "symmetrical_orbitals":
            raise RuntimeError("X2C supports only symmetrical orthogonalization")
        x2c = (nso == nao * 2)

    sys.stdout.write("Reading the input data...")
    sys.stdout.flush()

    seet = pymb.seet_init(args)

    F, S, T, dm, dm_s, e_nuc, nk, kmesh, kmesh_sc, reduced_mesh, reduced_mesh_sc, weight, conj_list, ir_list, bz_index \
        = seet.get_input_data()
    print("done!")

    # Set the orthoganlization method
    if args.orth_method == "natural_orbitals":
        orth_method = ou.natural_orbitals
    elif args.orth_method == "canonical_orbitals":
        orth_method = ou.canonical_orbitals
    elif args.orth_method == "symmetrical_orbitals":
        orth_method = ou.symmetrical_orbitals

    nkx, nky, nkz = [nk, nk, nk]
    if args.orth:
        sys.stdout.write("Constructing natural orbital basis...")
        sys.stdout.flush()
        if ns == 2:  # unrestricted
            X_inv_k, X_k = orth_method(S[0], dm, F[0], F[1], T[0], T[1], kmesh_sc)
        else:  # restricted or x2c
            X_inv_k, X_k = orth_method(S[0], dm, F[0], F[0], T[0], T[0], kmesh_sc)
        if nso == nao * 2:  # restricted or unrestricted
            X_ERI_k = np.array(X_k)[:,:nao,:nao].copy()
        else:  # x2c
            X_ERI_k = np.array(X_k)[:,:,:].copy()
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
