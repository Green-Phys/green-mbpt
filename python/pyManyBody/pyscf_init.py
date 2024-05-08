import os
import numpy as np

from . import common_utils as comm

from pyscf.df import addons


class pyscf_init:

    def __init__(self, args=None):
        if args is None :
            self.args = comm.init_pbc_params()
        else:
            self.args = args
        self.cell = comm.cell(self.args)
        self.kmesh, self.k_ibz, self.ir_list, self.conj_list, self.weight, self.ind, self.num_ik = comm.init_k_mesh(self.args, self.cell)
        

    def mean_field_input(self, mydf=None):
        if mydf is None:
            mydf = comm.construct_gdf(self.args, self.cell, self.kmesh)

        if os.path.exists("cderi.h5"):
            mydf._cderi = "cderi.h5"
        else:
            mydf._cderi_to_save = "cderi.h5"
            mydf.build()
        # number of k-points in each direction for Coulomb integrals
        nk       = self.args.nk ** 3
        # number of k-points in each direction to evaluate Coulomb kernel
        Nk       = self.args.Nk

        # number of orbitals per cell
        nao = self.cell.nao_nr()
        Zs = np.asarray(self.cell.atom_charges())
        print("Number of atoms: ", Zs.shape[0])
        print("Effective nuclear charge of each atom: ", Zs)
        atoms_info = np.asarray(self.cell.aoslice_by_atom())
        last_ao = atoms_info[:,3]
        print("aoslice_by_atom = ", atoms_info)
        print("Last AO index for each atom = ", last_ao)

        if self.args.grid_only:
            comm.store_k_grid(self.args, self.cell, self.kmesh, self.k_ibz, self.ir_list, self.conj_list, self.weight, self.ind, self.num_ik)
            return

        '''
        Generate integrals for mean-field calculations
        '''
        auxcell = addons.make_auxmol(self.cell, mydf.auxbasis)
        NQ = auxcell.nao_nr()
    
        mf = comm.solve_mean_field(self.args, mydf, self.cell)
    
        # Get Overlap and Fock matrices
        hf_dm = mf.make_rdm1().astype(dtype=np.complex128)
        S     = mf.get_ovlp().astype(dtype=np.complex128)
        T     = mf.get_hcore().astype(dtype=np.complex128)
        if self.args.xc is not None:
            vhf = mf.get_veff().astype(dtype=np.complex128)
        else:
            vhf = mf.get_veff(hf_dm).astype(dtype=np.complex128)
        F = mf.get_fock(T,S,vhf,hf_dm).astype(dtype=np.complex128)
    
        if len(F.shape) == 3:
            F     = F.reshape((1,) + F.shape)
            hf_dm = hf_dm.reshape((1,) + hf_dm.shape)
        S = np.array((S, ) * self.args.ns)
        T = np.array((T, ) * self.args.ns)
    
        X_k = []
        X_inv_k = []

        # Orthogonalization matrix
        X_k, X_inv_k, S, F, T, hf_dm = comm.orthogonalize(mydf, self.args.orth, X_k, X_inv_k, F, T, hf_dm, S)
        comm.save_data(self.args, self.cell, mf, self.kmesh, self.ind, self.weight, self.num_ik, self.ir_list, self.conj_list, Nk, nk, NQ, F, S, T, hf_dm, Zs, last_ao)

        comm.compute_df_int(self.args, self.cell, self.kmesh, nao, X_k)

    def evaluate_high_symmetry_path(self):
        if self.args.print_high_symmetry_points:
            comm.print_high_symmetry_points(self.cell, self.args)
            exit(0)
        if self.args.high_symmetry_path is None:
            raise RuntimeError("Please specify high-symmetry path")
        if args.high_symmetry_path is not None:
            try:
                comm.check_high_symmetry_path(self.cell, self.args)
            except RuntimeError as e:
                print("\n\n\n")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("!!!!!!!!! Cannot compute high-symmetry path !!!!!!!!!")
                print("!! Correct or Disable high-symmetry path evaluation !")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(e)
                exit(-1)
        kmesh_hs, Hk_hs, Sk_hs = comm.high_symmetry_path(self.cell, self.args)
        inp_data = h5py.File(self.args.output_path, "a")
        print(kmesh_hs)
        print(self.cell.get_scaled_kpts(kmesh_hs))
        inp_data["high_symm_path/k_mesh"] = self.cell.get_scaled_kpts(kmesh_hs)
        inp_data["high_symm_path/r_mesh"] = comm.construct_rmesh(self.args.nk, self.args.nk, self.args.nk)
        inp_data["high_symm_path/Hk"] = Hk_hs
        inp_data["high_symm_path/Sk"] = Sk_hs
    
    def evaluate_second_order_ewald(self):
        if not os.path.exists(args.hf_int_path):
            os.mkdir(args.hf_int_path)
        comm.compute_ewald_correction(self.args, self.cell, self.kmesh, self.args.hf_int_path + "/df_ewald.h5")

