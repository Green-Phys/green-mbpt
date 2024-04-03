import h5py
import numpy as np
import os
from pyscf.df import addons

import common_utils as comm

args = comm.init_pbc_params()

# number of k-points in each direction for Coulomb integrals
nk       = args.nk ** 3
# number of k-points in each direction to evaluate Coulomb kernel
Nk       = args.Nk

mycell = comm.cell(args)

if args.print_high_symmetry_points:
    comm.print_high_symmetry_points(mycell, args)
    exit(0)

if args.high_symmetry_path is not None:
    try:
        comm.check_high_symmetry_path(mycell, args)
    except RuntimeError as e:
        print("\n\n\n")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!! Cannot compute high-symmetry path !!!!!!!!!")
        print("!! Correct or Disable high-symmetry path evaluation !")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(e)
        exit(-1)

# number of orbitals per cell
nao = mycell.nao_nr()
Zs = np.asarray(mycell.atom_charges())
print("Number of atoms: ", Zs.shape[0])
print("Effective nuclear charge of each atom: ", Zs)
atoms_info = np.asarray(mycell.aoslice_by_atom())
last_ao = atoms_info[:,3]
print("aoslice_by_atom = ", atoms_info)
print("Last AO index for each atom = ", last_ao)

kmesh, k_ibz, ir_list, conj_list, weight, ind, num_ik = comm.init_k_mesh(args, mycell)

if args.grid_only:
    comm.store_k_grid(args, mycell, kmesh, k_ibz, ir_list, conj_list, weight, ind, num_ik)
    exit(0)


'''
Generate integrals for mean-field calculations
'''
mydf   = comm.df.GDF(mycell)
if args.auxbasis is not None:
    mydf.auxbasis = args.auxbasis
elif args.beta is not None:
    mydf.auxbasis = comm.df.aug_etb(mycell, beta=args.beta)
# Coulomb kernel mesh
if Nk > 0:
    mydf.mesh = [Nk, Nk, Nk]
mydf.kpts = kmesh
if os.path.exists("cderi.h5"):
    mydf._cderi = "cderi.h5"
else:
    mydf._cderi_to_save = "cderi.h5"
    mydf.build()
auxcell = addons.make_auxmol(mycell, mydf.auxbasis)
NQ = auxcell.nao_nr()

mf = comm.solve_mean_field(args, mydf, mycell)

# Get Overlap and Fock matrices
hf_dm = mf.make_rdm1().astype(dtype=np.complex128)
S     = mf.get_ovlp().astype(dtype=np.complex128)
T     = mf.get_hcore().astype(dtype=np.complex128)
if args.xc is not None:
    vhf = mf.get_veff().astype(dtype=np.complex128)
else:
    vhf = mf.get_veff(hf_dm).astype(dtype=np.complex128)
F     = mf.get_fock(T,S,vhf,hf_dm).astype(dtype=np.complex128)

if len(F.shape) == 3:
    F     = F.reshape((1,) + F.shape)
    hf_dm = hf_dm.reshape((1,) + hf_dm.shape)
S = np.array((S, ) * args.ns)
T = np.array((T, ) * args.ns)

X_k = []
X_inv_k = []

# Orthogonalization matrix
X_k, X_inv_k, S, F, T, hf_dm = comm.orthogonalize(mydf, args.orth, X_k, X_inv_k, F, T, hf_dm, S)
comm.save_data(args, mycell, mf, kmesh, ind, weight, num_ik, ir_list, conj_list, Nk, nk, NQ, F, S, T, hf_dm, Zs, last_ao)

comm.compute_df_int(args, mycell, kmesh, nao, X_k)

if args.high_symmetry_path is not None:
    kmesh_hs, Hk_hs, Sk_hs = comm.high_symmetry_path(mycell, args)
    inp_data = h5py.File(args.output_path, "a")
    print(kmesh_hs)
    print(mycell.get_scaled_kpts(kmesh_hs))
    inp_data["high_symm_path/k_mesh"] = mycell.get_scaled_kpts(kmesh_hs)
    inp_data["high_symm_path/r_mesh"] = comm.construct_rmesh(args.nk, args.nk, args.nk) #np.dot(comm.tools.pbc.get_lattice_Ls(mycell), np.linalg.inv(mycell.lattice_vectors()) )
    inp_data["high_symm_path/Hk"] = Hk_hs
    inp_data["high_symm_path/Sk"] = Sk_hs

print("Done")
