import os
import numpy as np
import h5py
import scipy.linalg
import tempfile

from pyscf.pbc import df
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df import rsdf
from pyscf.pbc.df import rsdf_helper
from pyscf.df.outcore import _guess_shell_ranges
from pyscf.pbc.lib.kpts_helper import (is_zero, gamma_point, member, unique,
                                       KPT_DIFF_TOL)
from pyscf import lib
from pyscf.lib import logger


# kpti == kptj: s2 symmetry
# kpti == kptj == 0 (gamma point): real
def _make_j3c_S_metric(mydf, cell, auxcell, kptij_lst, cderi_file):
    '''
    Incore version of make_j3c using the S metric with the same input/output format as in pbc.df.RSGDF._make_j3c from PySCF.

    This function should only be used to overwrite pbc.df.RSGDF._make_j3c
    '''
    print("******** GDF using the overlap metric ********")
    t1 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(mydf.stdout, mydf.verbose)
    max_memory = max(2000, mydf.max_memory-lib.current_memory()[0])

    omega = abs(mydf.omega)

    if mydf.use_bvk and mydf.kpts_band is None:
        bvk_kmesh = rsdf.kpts_to_kmesh(cell, mydf.kpts)
        if bvk_kmesh is None:
            log.debug("Single-kpt or non-Gamma-inclusive kmesh is found. "
                      "bvk kmesh is not used.")
        else:
            log.debug("Using bvk kmesh= [%d %d %d]", *bvk_kmesh)
    else:
        bvk_kmesh = None

    # The ideal way to hold the temporary integrals is to store them in the
    # cderi_file and overwrite them inplace in the second pass.  The current
    # HDF5 library does not have an efficient way to manage free space in
    # overwriting.  It often leads to the cderi_file ~2 times larger than the
    # necessary size.  For now, dumping the DF integral intermediates to a
    # separated temporary file can avoid this issue.  The DF intermediates may
    # be terribly huge. The temporary file should be placed in the same disk
    # as cderi_file.
    swapfile = tempfile.NamedTemporaryFile(dir=os.path.dirname(cderi_file))
    fswap = lib.H5TmpFile(swapfile.name)
    # Unlink swapfile to avoid trash
    swapfile = None

    # get charge of auxbasis
    if cell.dimension == 3:
        qaux = rsdf.get_aux_chg(auxcell)
    else:
        qaux = np.zeros(auxcell.nao_nr())

    nao = cell.nao_nr()
    naux = auxcell.nao_nr()

    kptis = kptij_lst[:,0]
    kptjs = kptij_lst[:,1]
    kpt_ji = kptjs - kptis
    uniq_kpts, uniq_index, uniq_inverse = unique(kpt_ji)

    log.debug('Num uniq kpts %d', len(uniq_kpts))
    log.debug2('uniq_kpts %s', uniq_kpts)

    # compute j2c first as it informs the integral screening in computing j3c
    # short-range part of j2c ~ (-kpt_ji | kpt_ji)
    omega_j2c = abs(mydf.omega_j2c)
    j2c = rsdf_helper.intor_j2c(auxcell, omega_j2c, kpts=uniq_kpts)

    # Add (1) short-range G=0 (i.e., charge) part and (2) long-range part
    qaux2 = None
    g0_j2c = np.pi/omega_j2c**2./cell.vol
    mesh_j2c = mydf.mesh_j2c
    Gv, Gvbase, kws = cell.get_Gv_weights(mesh_j2c)
    b = cell.reciprocal_vectors()
    gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
    ngrids = gxyz.shape[0]

    max_memory = max(2000, mydf.max_memory - lib.current_memory()[0])
    blksize = max(2048, int(max_memory*.5e6/16/auxcell.nao_nr()))
    log.debug2('max_memory %s (MB)  blocksize %s', max_memory, blksize)

    for k, kpt in enumerate(uniq_kpts):
        # short-range charge part
        if is_zero(kpt) and cell.dimension == 3:
            if qaux2 is None:
                qaux2 = np.outer(qaux,qaux)
            j2c[k] -= qaux2 * g0_j2c
        # long-range part via aft
        coulG_lr = mydf.weighted_coulG(omega_j2c, kpt, False, mesh_j2c)
        for p0, p1 in lib.prange(0, ngrids, blksize):
            aoaux = ft_ao.ft_ao(auxcell, Gv[p0:p1], None, b, gxyz[p0:p1],
                                Gvbase, kpt).T
            LkR = np.asarray(aoaux.real, order='C')
            LkI = np.asarray(aoaux.imag, order='C')
            aoaux = None

            if is_zero(kpt):  # kpti == kptj
                j2c[k] += lib.ddot(LkR*coulG_lr[p0:p1], LkR.T)
                j2c[k] += lib.ddot(LkI*coulG_lr[p0:p1], LkI.T)
            else:
                j2cR, j2cI = df.df_jk.zdotCN(LkR*coulG_lr[p0:p1],
                                             LkI*coulG_lr[p0:p1], LkR.T, LkI.T)
                j2c[k] += j2cR + j2cI * 1j

            LkR = LkI = None

        fswap['j2c/%d'%k] = j2c[k]
    j2c = coulG_lr = None

    t1 = log.timer_debug1('2c2e', *t1)

    def cholesky_decomposed_metric(uniq_kptji_id):
        j2c = np.asarray(fswap['j2c/%d'%uniq_kptji_id])
        j2c_negative = None
        try:
            if mydf.j2c_eig_always:
                raise scipy.linalg.LinAlgError
            j2c = scipy.linalg.cholesky(j2c, lower=False)
            j2ctag = 'CD'
        except scipy.linalg.LinAlgError:
            #msg =('===================================\n'
            #      'J-metric not positive definite.\n'
            #      'It is likely that mesh is not enough.\n'
            #      '===================================')
            #log.error(msg)
            #raise scipy.linalg.LinAlgError('\n'.join([str(e), msg]))
            w, v = scipy.linalg.eigh(j2c)
            ndrop = np.count_nonzero(w<mydf.linear_dep_threshold)
            if ndrop > 0:
                log.debug('DF metric linear dependency for kpt %s',
                          uniq_kptji_id)
                log.debug('cond = %.4g, drop %d bfns', w[-1]/w[0], ndrop)
            v1 = v[:,w>mydf.linear_dep_threshold].conj().T
            v1 *= np.sqrt(w[w>mydf.linear_dep_threshold]).reshape(-1,1)
            j2c = v1
            if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
                idx = np.where(w < -mydf.linear_dep_threshold)[0]
                if len(idx) > 0:
                    j2c_negative = (v[:,idx]*np.sqrt(-w[idx])).conj().T
            w = v = None
            j2ctag = 'eig'
        return j2c, j2c_negative, j2ctag

    # compute j3c
    # inverting j2c, and use it's column max to determine an extra precision for 3c2e prescreening

    # short-range part
    df.outcore._aux_e2(cell, auxcell, fswap, intor='int3c1e', aosym='s2',
                       kptij_lst=kptij_lst, dataname='j3c-junk', max_memory=max_memory)
    t1 = log.timer_debug1('3c2e', *t1)

    tspans = np.zeros((3,2))    # lr, j2c_inv, j2c_cntr
    tspannames = ["ftaop+pw", "j2c_inv", "j2c_cntr"]
    feri = h5py.File(cderi_file, 'w')
    feri['j3c-kptij'] = kptij_lst
    nsegs = len(fswap['j3c-junk/0'])
    def make_kpt(uniq_kptji_id, cholesky_j2c):
        kpt = uniq_kpts[uniq_kptji_id]  # kpt = kptj - kpti
        log.debug1('kpt = %s', kpt)
        adapted_ji_idx = np.where(uniq_inverse == uniq_kptji_id)[0]
        adapted_kptjs = kptjs[adapted_ji_idx]
        nkptj = len(adapted_kptjs)
        log.debug1('adapted_ji_idx = %s', adapted_ji_idx)

        j2c, j2c_negative, j2ctag = cholesky_j2c

        if is_zero(kpt):  # kpti == kptj
            aosym = 's2'
            nao_pair = nao*(nao+1)//2
        else:
            aosym = 's1'
            nao_pair = nao**2

        mem_now = lib.current_memory()[0]
        log.debug2('memory = %s', mem_now)
        max_memory = max(2000, mydf.max_memory-mem_now)
        # nkptj for 3c-coulomb arrays plus 1 Lpq array
        buflen = min(max(int(max_memory*.38e6/16/naux/(nkptj+1)), 1), nao_pair)
        shranges = _guess_shell_ranges(cell, buflen, aosym)
        buflen = max([x[2] for x in shranges])

        # Load SR-part of j3c for aux_slice for all k-pair in adapated_ji_idx
        def load(aux_slice):
            col0, col1 = aux_slice
            j3cR = []
            j3cI = []
            for k, idx in enumerate(adapted_ji_idx):
                v = np.vstack([fswap['j3c-junk/%d/%d'%(idx,i)][0,col0:col1].T
                               for i in range(nsegs)])
                j3cR.append(np.asarray(v.real, order='C'))
                if is_zero(kpt) and gamma_point(adapted_kptjs[k]):
                    j3cI.append(None)
                else:
                    j3cI.append(np.asarray(v.imag, order='C'))
                v = None
            return j3cR, j3cI

        # buf for ft_aopair
        cols = [sh_range[2] for sh_range in shranges]
        locs = np.append(0, np.cumsum(cols))
        tasks = zip(locs[:-1], locs[1:])
        for istep, (j3cR, j3cI) in enumerate(lib.map_with_prefetch(load, tasks)):
            bstart, bend, ncol = shranges[istep]
            log.debug1('int3c2e [%d/%d], AO [%d:%d], ncol = %d',
                       istep+1, len(shranges), bstart, bend, ncol)

            tock_ = np.asarray((logger.process_clock(), logger.perf_counter()))
            for k, ji in enumerate(adapted_ji_idx):
                if is_zero(kpt) and gamma_point(adapted_kptjs[k]):
                    v = j3cR[k]
                else:
                    v = j3cR[k] + j3cI[k] * 1j
                v = scipy.linalg.solve(ovlp_2c1e_aux, v, overwrite_b=True)
                feri['j3c/%d/%d' % (ji, istep)] = lib.dot(j2c, v)

                # low-dimension systems
                if j2c_negative is not None:
                    feri['j3c-/%d/%d'%(ji,istep)] = lib.dot(j2c_negative, v)
            j3cR = j3cI = None
            tick_ = np.asarray((logger.process_clock(), logger.perf_counter()))
            tspans[2] += tick_ - tock_

        for ji in adapted_ji_idx:
            del(fswap['j3c-junk/%d'%ji])

    # Wrapped around boundary and symmetry between k and -k can be used
    # explicitly for the metric integrals.  We consider this symmetry
    # because it is used in the df_ao2mo module when contracting two 3-index
    # integral tensors to the 4-index 2e integral tensor. If the symmetry
    # related k-points are treated separately, the resultant 3-index tensors
    # may have inconsistent dimension due to the numerial noise when handling
    # linear dependency of j2c.
    def conj_j2c(cholesky_j2c):
        j2c, j2c_negative, j2ctag = cholesky_j2c
        if j2c_negative is None:
            return j2c.conj(), None, j2ctag
        else:
            return j2c.conj(), j2c_negative.conj(), j2ctag

    a = cell.lattice_vectors() / (2*np.pi)
    def kconserve_indices(kpt):
        '''search which (kpts+kpt) satisfies momentum conservation'''
        kdif = np.einsum('wx,ix->wi', a, uniq_kpts + kpt)
        kdif_int = np.rint(kdif)
        mask = np.einsum('wi->i', abs(kdif - kdif_int)) < KPT_DIFF_TOL
        uniq_kptji_ids = np.where(mask)[0]
        return uniq_kptji_ids

    # Loop over q-points, i.e. uniq_kpts
    done = np.zeros(len(uniq_kpts), dtype=bool)
    for k, kpt in enumerate(uniq_kpts):
        if done[k]:
            continue

        log.debug1('Cholesky decomposition for j2c at kpt %s', k)
        tick_ = np.asarray((logger.process_clock(), logger.perf_counter()))
        # j2c_sqrt, j2c_sqrt_negative, j2c_tag
        cholesky_j2c = cholesky_decomposed_metric(k)
        ovlp_2c1e_aux = mydf.auxcell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpt)
        tock_ = np.asarray((logger.process_clock(), logger.perf_counter()))
        tspans[1] += tock_ - tick_

        # The k-point k' which has (k - k') * a = 2n pi. Metric integrals have the
        # symmetry S = S
        uniq_kptji_ids = kconserve_indices(-kpt)  # A subset of symmetry related q-points in uniq_kpts
        log.debug1("Symmetry pattern (k - %s)*a= 2n pi", kpt)
        log.debug1("    make_kpt for uniq_kptji_ids %s", uniq_kptji_ids)
        for uniq_kptji_id in uniq_kptji_ids:
            if not done[uniq_kptji_id]:
                make_kpt(uniq_kptji_id, cholesky_j2c)
        done[uniq_kptji_ids] = True

        # The k-point k' which has (k + k') * a = 2n pi. Metric integrals have the
        # symmetry S = S*
        uniq_kptji_ids = kconserve_indices(kpt)
        log.debug1("Symmetry pattern (k + %s)*a= 2n pi", kpt)
        log.debug1("    make_kpt for %s", uniq_kptji_ids)
        cholesky_j2c = conj_j2c(cholesky_j2c)
        ovlp_2c1e_aux = ovlp_2c1e_aux.conj()
        for uniq_kptji_id in uniq_kptji_ids:
            if not done[uniq_kptji_id]:
                make_kpt(uniq_kptji_id, cholesky_j2c)
        done[uniq_kptji_ids] = True

    feri.close()

    # report time for aft part
    for tspan, tspanname in zip(tspans, tspannames):
        log.debug1("    CPU time for %s %9.2f sec, wall time %9.2f sec",
                   "%10s"%tspanname, *tspan)
    log.debug1("%s", "")
