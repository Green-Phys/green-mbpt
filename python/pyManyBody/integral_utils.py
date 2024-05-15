import os
import random
import string

import h5py
import numpy as np
from numba import jit
from pyscf.df import addons
from pyscf.pbc import df, tools


def compute_kG(k, Gv, wrap_around, mesh, cell):
    if abs(k).sum() > 1e-9:
        kG = k + Gv
    else:
        kG = Gv

    equal2boundary = np.zeros(Gv.shape[0], dtype=bool)
    if wrap_around and abs(k).sum() > 1e-9:
        # Here we 'wrap around' the high frequency k+G vectors into their lower
        # frequency counterparts.  Important if you want the gamma point and k-point
        # answers to agree
        b = cell.reciprocal_vectors()
        box_edge = np.einsum('i,ij->ij', np.asarray(mesh)//2+0.5, b)
        assert (all(np.linalg.solve(box_edge.T, k).round(9).astype(int)==0))
        reduced_coords = np.linalg.solve(box_edge.T, kG.T).T.round(9)
        on_edge = reduced_coords.astype(int)
        if cell.dimension >= 1:
            equal2boundary |= reduced_coords[:,0] == 1
            equal2boundary |= reduced_coords[:,0] ==-1
            kG[on_edge[:,0]== 1] -= 2 * box_edge[0]
            kG[on_edge[:,0]==-1] += 2 * box_edge[0]
        if cell.dimension >= 2:
            equal2boundary |= reduced_coords[:,1] == 1
            equal2boundary |= reduced_coords[:,1] ==-1
            kG[on_edge[:,1]== 1] -= 2 * box_edge[1]
            kG[on_edge[:,1]==-1] += 2 * box_edge[1]
        if cell.dimension == 3:
            equal2boundary |= reduced_coords[:,2] == 1
            equal2boundary |= reduced_coords[:,2] ==-1
            kG[on_edge[:,2]== 1] -= 2 * box_edge[2]
            kG[on_edge[:,2]==-1] += 2 * box_edge[2]
    return kG, equal2boundary



def get_coarsegrained_coulG(lattice_kmesh, cell, k=np.zeros(3), exx=False, mf=None, mesh=None, Gv=None,
              wrap_around=True, omega=None, **kwargs):
    '''Calculate the coarse-grained Coulomb kernel for all G-vectors, handling G=0 and exchange.
    This routine overrides get_coulG to perform interaction coarse-graining.
    '''
    exxdiv = exx
    if isinstance(exx, str):
        exxdiv = exx
    elif exx and mf is not None:
        exxdiv = mf.exxdiv

    if mesh is None:
        mesh = cell.mesh
    if 'gs' in kwargs:
        warnings.warn('cell.gs is deprecated.  It is replaced by cell.mesh,'
                      'the number of PWs (=2*gs+1) along each direction.')
        mesh = [2*n+1 for n in kwargs['gs']]
    if Gv is None:
        Gv = cell.get_Gv(mesh)
    absG2 = []
    kG_0, equal2boundary_0 = compute_kG(k, Gv, wrap_around, mesh, cell)
    absG2_0 = np.einsum('gi,gi->g', kG_0, kG_0)
    for kp in lattice_kmesh:
        kG, equal2boundary = compute_kG(k + kp, Gv, wrap_around, mesh, cell)
        absG2.append(np.einsum('gi,gi->g', kG, kG))
    absG2 = np.array(absG2)

    if getattr(mf, 'kpts', None) is not None:
        kpts = mf.kpts
    else:
        kpts = k.reshape(1,3)
    Nk = len(kpts)
    fullkpts = np.zeros([kpts.shape[0]*lattice_kmesh.shape[0],3])
    for ikk, kk in enumerate(kpts):
        for ikkp, kkp in enumerate(lattice_kmesh):
            fullkpts[ikk*lattice_kmesh.shape[0] + ikkp] = kk + kkp
    Nkk = Nk * lattice_kmesh.shape[0]

    if exxdiv == 'vcut_sph':  # PRB 77 193110
        raise NotImplementedError

    elif exxdiv == 'vcut_ws':  # PRB 87, 165122
        raise NotImplementedError

    else:
        # Ewald probe charge method to get the leading term of the finite size
        # error in exchange integrals

        G0_idx = np.where(absG2_0==0)[0]
        if cell.dimension != 2 or cell.low_dim_ft_type == 'inf_vacuum':
            coulG = np.zeros(absG2.shape)
            for ikp in range(absG2.shape[0]):
                #with np.errstate(divide='ignore'):
                coulG[ikp] = 4*np.pi/absG2[ikp]
                if np.sum(np.dot(k,k)) < 1e-9: 
                    print(ikp, 4*np.pi/absG2[ikp])
                if not np.isfinite(coulG[ikp, G0_idx]) :
                    coulG[ikp, G0_idx] = 0
            coulG = np.sum(coulG, axis=0) / absG2.shape[0]
            if np.sum(np.dot(k,k)) < 1e-9: 
               print("Sum:", coulG)
            #coulG[G0_idx] = 0
            

        elif cell.dimension == 2:

            raise NotImplementedError

        elif cell.dimension == 1:
            logger.warn(cell, 'No method for PBC dimension 1, dim-type %s.'
                        '  cell.low_dim_ft_type="inf_vacuum"  should be set.',
                        cell.low_dim_ft_type)
            raise NotImplementedError


        # The divergent part of periodic summation of (ii|ii) integrals in
        # Coulomb integrals were cancelled out by electron-nucleus
        # interaction. The periodic part of (ii|ii) in exchange cannot be
        # cancelled out by Coulomb integrals. Its leading term is calculated
        # using Ewald probe charge (the function madelung below)
        if cell.dimension > 0 and exxdiv == 'ewald' and len(G0_idx) > 0:
            coulG[G0_idx] += Nkk*cell.vol*tools.madelung(cell, fullkpts)

    coulG[equal2boundary_0] = 0

    # Scale the coulG kernel for attenuated Coulomb integrals.
    # * omega is used by RangeSeparatedJKBuilder which requires ewald probe charge
    # being evaluated with regular Coulomb interaction (1/r12).
    # * cell.omega, which affects the ewald probe charge, is often set by
    # DFT-RSH functionals to build long-range HF-exchange for erf(omega*r12)/r12
    if omega is not None:
        if omega > 0:
            # long range part
            coulG *= np.exp(-.25/omega**2 * absG2)
        elif omega < 0:
            # short range part
            coulG *= (1 - np.exp(-.25/omega**2 * absG2))
    elif cell.omega > 0:
        coulG *= np.exp(-.25/cell.omega**2 * absG2)
    elif cell.omega < 0:
        raise NotImplementedError

    return coulG

def weighted_coulG_ewald(mydf, kpt, exx, mesh, omega=None):
    return df.aft.weighted_coulG(mydf, kpt, "ewald", mesh, omega)

# a = lattice vectors / (2*pi)
@jit(nopython=True)
def kpair_reduced_lists(kptis, kptjs, kptij_idx, kmesh, a):
    nkpts = kmesh.shape[0]
    print("nkpts = ", nkpts)
    kptis = np.asarray(kptis)
    kptjs = np.asarray(kptjs)
    if kptis.shape[0] != kptjs.shape[0]:
        raise ValueError("Error: Dimension of kptis and kptjs doesn't match.")
    num_kpair = kptis.shape[0]
    conj_list = np.arange(num_kpair, dtype=np.int64)
    trans_list = np.arange(num_kpair, dtype=np.int64)
    seen = np.zeros(num_kpair, dtype=np.int64)
    for i in range(num_kpair):
        if seen[i] != 0:
            continue
        k1 = kptis[i]
        k2 = kptjs[i]
        k1_idx, k2_idx = kptij_idx[i]
        l = 0
        # conj_list: (k1, k2) = (-kk1, -kk2)
        for kk1_idx in range(k1_idx,nkpts):
            kk1 = kmesh[kk1_idx]
            kjdif = np.dot(a, k1 + kk1)
            kjdif_int = np.rint(kjdif)
            maskj = np.sum(np.abs(kjdif - kjdif_int)) < 1e-6

            if maskj == True:
                for kk2_idx in range(0,kk1_idx+1):
                    kk2 = kmesh[kk2_idx]
                    kidif = np.dot(a, k2 + kk2)
                    #kidif = np.einsum('wx,x->w', a, k2 + kk2)
                    kidif_int = np.rint(kidif)
                    maski = np.sum(np.abs(kidif - kidif_int)) < 1e-6

                    if maski == True:
                        j = int(kk1_idx * (kk1_idx + 1)/2 + kk2_idx)
                        conj_list[j] = i
                        seen[j] = 1
                        l = 1
                        break
            if l == 1:
                break
        if l == 1:
            seen[i] = 1
            continue

        # trans_list: (k1, k2) = (-kk2, -kk1)
        for kk1_idx in range(k1_idx,nkpts):
            kk1 = kmesh[kk1_idx]
            kjdif = np.dot(a, k2 + kk1)
            kjdif_int = np.rint(kjdif)
            maskj = np.sum(np.abs(kjdif - kjdif_int)) < 1e-6

            if maskj == True:
                for kk2_idx in range(0,kk1_idx+1):
                    kk2 = kmesh[kk2_idx]
                    kidif = np.dot(a, k1 + kk2)
                    kidif_int = np.rint(kidif)
                    maski = np.sum(np.abs(kidif - kidif_int)) < 1e-6

                    if maski == True:
                        j = int(kk1_idx * (kk1_idx + 1)/2 + kk2_idx)
                        trans_list[j] = i
                        seen[j] = 1
                        l = 1
                        break
            if l == 1:
                break
        seen[i] = 1

    return conj_list, trans_list


def integrals_grid(mycell, kmesh):
    a_lattice = mycell.lattice_vectors() / (2*np.pi)
    kptij_lst = [(ki, kmesh[j]) for i, ki in enumerate(kmesh) for j in range(i+1)]
    kptij_idx = [(i, j) for i in range(kmesh.shape[0]) for j in range(i+1)]
    kptij_lst = np.asarray(kptij_lst)
    kptij_idx = np.asarray(kptij_idx)
    kptis = kptij_lst[:,0]
    kptjs = kptij_lst[:,1]
    kij_conj, kij_trans = kpair_reduced_lists(kptis, kptjs, kptij_idx, kmesh, a_lattice)
    kpair_irre_list = np.argwhere(kij_conj == kij_trans)[:,0]
    num_kpair_stored = len(kpair_irre_list)
    return kptij_idx, kij_conj, kij_trans, kpair_irre_list, num_kpair_stored, kptis, kptjs

def compute_integrals(args, mycell, mydf, kmesh, nao, X_k=None, basename = "df_int", cderi_name="cderi.h5", keep=True, keep_after=False):

    kptij_idx, kij_conj, kij_trans, kpair_irre_list, num_kpair_stored, kptis, kptjs = integrals_grid(mycell, kmesh)

    mydf.kpts = kmesh
    filename = basename + "/meta.h5"
    os.system("sync") # This is needed to syncronize the NFS between nodes
    if os.path.exists(basename):
        unsafe_rm = "rm  " + basename + "/VQ*"
        os.system(unsafe_rm)
        unsafe_rm = "rm  " + basename + "/meta*"
        os.system(unsafe_rm)
        # This is needed to ensure that files are removed both on the computational and head nodes:
        os.system("sync") 
    os.system("mkdir -p " + basename) # Here "-p" is important and is needed if the if condition is trigerred
    if os.path.exists(cderi_name) and keep :
        mydf._cderi = cderi_name
    else:
        mydf._cderi_to_save = cderi_name
        mydf.build()

    auxcell = addons.make_auxmol(mycell, mydf.auxbasis)
    NQ = auxcell.nao_nr()
    print("NQ = ", NQ)

    # compute partitioning

    def compute_partitioning(tot_size, num_kpair_stored):
        # We have a guess for each fitted density upper bound of 150M
        ubound = args.memory * 1024 * 1024
        if tot_size > ubound :
            mult = tot_size // ubound
            chunks = num_kpair_stored // (mult+1)
            if chunks == 0 :
                print("\n\n\n Chunk size is bigger than upper memory bound per chunk you have \n\n\n")
                chunks = 1
            return chunks
        return num_kpair_stored

    single_rho_size = nao**2 * NQ * 16
    full_rho_size   = (num_kpair_stored * single_rho_size)
    chunk_size = compute_partitioning(full_rho_size, num_kpair_stored)
    print("The chunk size: ", chunk_size, " k-point pair")

    # open file to write integrals in
    if os.path.exists(filename) :
        os.remove(filename)
    data = h5py.File(filename, "w")

    # Loop over k-point pair
    # processed densities count
    cnt = 0
    # densities buffer
    buffer = np.zeros((chunk_size, NQ, nao, nao), dtype=complex)
    Lpq_mo = np.zeros((NQ, nao, nao), dtype=complex)
    chunk_indices = []
    for i in kpair_irre_list:
        k1 = kptis[i]
        k2 = kptjs[i]
        # auxiliary basis index
        s1 = 0
        for XXX in mydf.sr_loop((k1,k2), max_memory=4000, compact=False):
            LpqR = XXX[0]
            LpqI = XXX[1]
            Lpq = (LpqR + LpqI*1j).reshape(LpqR.shape[0], nao, nao)
            buffer[cnt% chunk_size, s1:s1+Lpq.shape[0], :, :] = Lpq[0:Lpq.shape[0],:,:]
            # s1 = NQ at maximum.
            s1 += Lpq.shape[0]
        cnt += 1

        # if reach chunk size: (cnt-chunk_size) equals to chunk id.
        if cnt % chunk_size == 0:
            chunk_name = basename + "/VQ_{}.h5".format(cnt - chunk_size)
            if os.path.exists(chunk_name) :
                os.remove(chunk_name)
            VQ = h5py.File(chunk_name, "w")
            VQ["{}".format(cnt - chunk_size)] = buffer.view(np.float64)
            VQ.close()
            chunk_indices.append(cnt - chunk_size)
            buffer[:] = 0.0
    # Deal the rest
    if cnt % chunk_size != 0:
        last_chunk = (num_kpair_stored // chunk_size) * chunk_size
        chunk_name = basename + "/VQ_{}.h5".format(last_chunk)
        if os.path.exists(chunk_name) :
            os.remove(chunk_name)
        VQ = h5py.File(chunk_name, "w")
        VQ["{}".format(last_chunk)] = buffer.view(np.float64)
        chunk_indices.append(last_chunk)
        VQ.close()
        buffer[:] = 0.0

    data["chunk_size"] = chunk_size
    data["chunk_indices"] = np.array(chunk_indices)
    data.close()
    if not keep_after:
        os.remove(cderi_name)
        os.system("sync")
    print("Integrals have been computed and stored into {}".format(filename))
    return kij_conj, kij_trans, kpair_irre_list, kptij_idx, num_kpair_stored

def weighted_coulG_ewald_2nd(mydf, kpt, exx, mesh):
    # this is a dirty hack
    # PySCF needs to have a full k-grid to properly compute the madelung constant
    # but we want to compute only the contribution for ki == kj to speedup this calculations
    # TODO Double check where we define full_k_mesh?
    if not hasattr(mydf.cell, 'full_k_mesh'):
         raise RuntimeError("Using wrong DF object")
    oldkpts = mydf.kpts
    mydf.kpts = mydf.cell.full_k_mesh
    coulG = df.aft.weighted_coulG(mydf, kpt, 'ewald', mesh)
    mydf.kpts = oldkpts
    return coulG

def compute_ewald_correction(args, maindf, kmesh, nao, filename = "df_ewald.h5"):
    # global full_k_mesh
    data = h5py.File(filename, "w")
    EW     = data.create_group("EW")
    EW_bar = data.create_group("EW_bar")
    # keep original method for computing Coulomb kernel
    import pyscf.pbc.df.gdf_builder as gdf
    weighted_coulG_old = gdf._CCGDFBuilder.weighted_coulG

    # density-fitting w/o ewald correction for fine grid
    df2 = df.GDF(maindf.cell)
    if hasattr(df2, "_prefer_ccdf"):
        df2._prefer_ccdf = True  # Disable RS-GDF switch for new pyscf versions
    if maindf.auxbasis is not None:
        df2.auxbasis = maindf.auxbasis
    # Coulomb kernel mesh
    df2.mesh = maindf.mesh
    cderi_file_2 = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10)) + ".h5"
    df2._cderi_to_save = cderi_file_2
    df2._cderi = cderi_file_2
    df2.kpts = kmesh
    df2.build()

    # densities buffer
    auxcell = addons.make_auxmol(maindf.cell, maindf.auxbasis)
    NQ = auxcell.nao_nr()
    buffer1 = np.zeros((NQ, nao, nao), dtype=np.complex128)
    buffer2 = np.zeros((NQ, nao, nao), dtype=np.complex128)
    Lpq_mo = np.zeros((NQ, nao, nao), dtype=np.complex128)

    gdf._CCGDFBuilder.weighted_coulG = weighted_coulG_ewald_2nd
    # df.aft.weighted_coulG = weighted_coulG_ewald_2nd
    df1 = df.GDF(maindf.cell)
    df1.cell.full_k_mesh = maindf.kpts
    if hasattr(df1, "_prefer_ccdf"):
        df1._prefer_ccdf = True  # Disable RS-GDF switch for new pyscf versions
    if maindf.auxbasis is not None:
        df1.auxbasis = maindf.auxbasis
    # Use Ewald for divergence treatment
    df1.exxdiv = 'ewald'
    # Coulomb kernel mesh
    df1.mesh = maindf.mesh
    cderi_file_1 = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10)) + ".h5"
    df1._cderi_to_save = cderi_file_1
    df1._cderi = cderi_file_1
    df1.kpts = kmesh
    df1.build()
    nk = args.nk

    # We know that G=0 contribution diverge only when q = 0
    # so we loop over (k1,k1) pairs
    for i, ki in enumerate(kmesh):
        # Change the way to compute Coulomb kernel to include G=0 correction
        # df.GDF.weighted_coulG = weighted_coulG_ewald_2nd
        gdf._CCGDFBuilder.weighted_coulG = weighted_coulG_ewald_2nd
        s1 = 0
        # Compute three-point integrals with G=0 contribution included with Ewald correction
        for XXX in df1.sr_loop((ki,ki), max_memory=4000, compact=False):
            LpqR = XXX[0]
            LpqI = XXX[1]
            Lpq = (LpqR + LpqI*1.j).reshape(LpqR.shape[0], nao, nao)
            for G in range(Lpq.shape[0]):
                Lpq_mo[G] = Lpq[G]
            buffer1[s1:s1+Lpq.shape[0], :, :] = Lpq_mo[0:Lpq.shape[0],:,:]
            s1 += Lpq.shape[0]

        # Restore the way to compute Coulomb kernel
        # df.aft.weighted_coulG = weighted_coulG_old
        gdf._CCGDFBuilder.weighted_coulG = weighted_coulG_old
        s1 = 0
        # Compute three-point integral without G=0 contribution included with Ewald correction
        # and subtract it from the computed buffer to keep pure Ewald correction only
        for XXX in df2.sr_loop((ki,ki), max_memory=4000, compact=False):
            LpqR = XXX[0]
            LpqI = XXX[1]
            Lpq = (LpqR + LpqI*1.j).reshape(LpqR.shape[0], nao, nao)
            for G in range(Lpq.shape[0]):
                Lpq_mo[G] = Lpq[G]
            buffer2[s1:s1+Lpq.shape[0], :, :] = Lpq_mo[0:Lpq.shape[0],:,:]
            s1 += Lpq.shape[0]

        # i = k1 * nk * nk + k2 * nk + k3
        k3 = int(i /(nk*nk))
        k2 = int((i % (nk*nk))/nk)
        k1 = int(i % nk)
        k_fine = (k3) * (nk)*(nk) + (k2) * (nk) + (k1)
        EW["{}".format(k_fine)] = (buffer1 - buffer2).view(np.float64)
        EW_bar["{}".format(k_fine)] = buffer2.view(np.float64)
        buffer1[:] = 0.0
        buffer2[:] = 0.0

    data.close()
    # cleanup
    # df.aft.weighted_coulG = weighted_coulG_old
    gdf._CCGDFBuilder.weighted_coulG = weighted_coulG_old
    os.remove(cderi_file_1)
    os.remove(cderi_file_2)
    print("Ewald correction has been computed and stored into {}".format(filename))
