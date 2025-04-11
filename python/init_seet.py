import argparse
import sys

import green_mbtools.mint as pymb
import h5py
import numpy as np
from green_mbtools.mint import ortho_utils as ou

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SEET pre-processing")
    parser.add_argument("--orth", type=lambda x: (str(x).lower() in ['true','1', 'yes']), default='true', help="Apply basis orthogonalization.")
    parser.add_argument("--input_file", type=str, default="input.h5", help="Input file name.")
    parser.add_argument("--gf2_input_file", type=str, default="sim.h5", help="Converged GF2 simpulation result file.")
    parser.add_argument("--transform_file", type=str, default="transform.h5", help="Input file name.")
    parser.add_argument("--active_space", type=int, nargs='+', action='append', help="Input file name.", required=True)
    parser.add_argument("--orth_method", choices=["natural_orbitals", "canonical_orbitals", "symmetrical_orbitals"], default="natural_orbitals", help="Type of the orthogonalization.")
    parser.add_argument("--from_ibz", type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default='false',
                        help="Input data is in the reduced BZ.")

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

    nkx, nky, nkz = [nk, nk, nk]
    if args.orth:
        sys.stdout.write("Constructing natural orbital basis...")
        sys.stdout.flush()
        if ns == 2:
            X_inv_k, X_k = eval("ou." + args.orth_method)(S[0], dm, F[0], F[1], T[0], T[1], kmesh_sc)
        else :
            X_inv_k, X_k = eval("ou." + args.orth_method)(S[0], dm, F[0], F[0], T[0], T[0], kmesh_sc)
        if nso == nao * 2:
            X_ERI_k = np.array(X_k)[:,:nao,:nao].copy()
        else:
            X_ERI_k = np.array(X_k)[:,:,:].copy()
        print("done!")

    if args.orth:
        sys.stdout.write("Transforming into the orthogonal basis...")
        sys.stdout.flush()
        for s in range(F.shape[0]):
            F[s] = ou.transform(F[s], X_k,X_inv_k)
            S[s] = ou.transform(S[s], X_k,X_inv_k)
            T[s] = ou.transform(T[s], X_k,X_inv_k)
            dm_s[s] = ou.transform(dm_s[s], [x_inv_k.conj().T for x_inv_k in X_inv_k], [x_k.conj().T for x_k in X_k])
        dm = ou.transform(dm, [x_inv_k.conj().T for x_inv_k in X_inv_k],[x_k.conj().T for x_k in X_k])
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
