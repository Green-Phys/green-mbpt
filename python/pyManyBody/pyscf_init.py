import os
import numpy as np
import h5py

from pyscf.df import addons
from pyscf.pbc import tools

from . import common_utils as comm
from . import integral_utils as int_utils
from . import GDF_S_metric as gdf_S





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
        if bool(self.args.df_int) :
            self.compute_df_int(nao, X_k)

    def compute_df_int(self, nao, X_k):
        '''
        Generate density-fitting integrals for correlated methods
        '''
        mydf = comm.construct_gdf(self.args, self.cell, self.kmesh)
        int_utils.compute_integrals(self.args, self.cell, mydf, self.kmesh, nao, X_k, "df_hf_int", "cderi.h5", True, self.args.keep_cderi)
        mydf = None

        if self.args.finite_size_kind in ['gf2', 'gw', 'gw_s'] :
            self.compute_twobody_finitesize_correction()
            return

        mydf = comm.construct_gdf(self.args, self.cell, self.kmesh)
        # Use Ewald for divergence treatment
        mydf.exxdiv = 'ewald'
        import importlib
        new_pyscf = importlib.find_loader('pyscf.pbc.df.gdf_builder') is not None
        if new_pyscf :
            import pyscf.pbc.df.gdf_builder as gdf
            weighted_coulG_old = gdf._CCGDFBuilder.weighted_coulG
            gdf._CCGDFBuilder.weighted_coulG = int_utils.weighted_coulG_ewald
        else:
            weighted_coulG_old = gdf.GDF.weighted_coulG
            gdf.GDF.weighted_coulG = int_utils.weighted_coulG_ewald
    
        #kij_conj, kij_trans, kpair_irre_list, kptij_idx, num_kpair_stored = 
        int_utils.compute_integrals(self.args, self.cell, mydf, self.kmesh, nao, X_k, "df_int", "cderi_ewald.h5", True, self.args.keep_cderi)
        if new_pyscf :
            gdf._CCGDFBuilder.weighted_coulG = weighted_coulG_old
        else:
            gdf.GDF.weighted_coulG = weighted_coulG_old


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

    def compute_twobody_finitesize_correction(self, mydf=None):
        if not os.path.exists(self.args.hf_int_path):
            os.mkdir(self.args.hf_int_path)
        if self.args.finite_size_kind == 'gf2' :
            comm.compute_ewald_correction(self.args, self.cell, self.kmesh, self.args.hf_int_path + "/df_ewald.h5")
        elif self.args.finite_size_kind == 'gw' :
            self.evaluate_gw_correction(mydf)
            
    
    def evaluate_gw_correction(self, mydf=None):
        if mydf is None:
            mydf = comm.construct_gdf(self.args, self.cell, self.kmesh)
        mydf.build()

        j3c, kptij_lst, j2c_sqrt, uniq_kpts = gdf_S.make_j3c(mydf, self.cell, j2c_sqrt=True, exx=False)
        
        ''' Transformation matrix from auxiliary basis to plane-wave '''
        AqQ, q_reduced, q_scaled_reduced = gdf_S.transformation_PW_to_auxbasis(mydf, self.cell, j2c_sqrt, uniq_kpts)
        
        q_abs = np.array([np.linalg.norm(qq) for qq in q_reduced])
        q_abs = np.array([round(qq, 8) for qq in q_abs])
        
        # Different prefactors for the GW finite-size correction for testing
        # In practice, the madelung constant is used, which decays as (1/nk).
        X = (6*np.pi**2)/(self.cell.vol*len(self.kmesh))
        X = (2.0/np.pi) * np.cbrt(X)
        
        X2 = 2.0 * np.cbrt(1.0/(self.cell.vol*len(self.kmesh)))
        
        f = h5py.File(self.args.hf_int_path + "/AqQ.h5", 'w')
        f["AqQ"] = AqQ
        f["qs"] = q_reduced
        f["qs_scaled"] = q_scaled_reduced
        f["q_abs"] = q_abs
        f["X"] = X
        f["X2"] = X2
        f["madelung"] = tools.pbc.madelung(self.cell, self.kmesh)
        f.close()
