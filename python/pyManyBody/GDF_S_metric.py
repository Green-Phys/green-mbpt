import numpy
import h5py
import scipy.linalg as LA

from . import integral_utils as int_utils
from . import common_utils as comm

from pyscf import lib
from pyscf.pbc.lib.kpts_helper import is_zero, member, unique
from pyscf.pbc.df import df, ft_ao, incore, aft
from pyscf.pbc.df.df_jk import zdotCN
import importlib
if importlib.find_loader('pyscf.pbc.df.gdf_builder') is not None :
    import pyscf.pbc.df.gdf_builder as gdf_builder

'''
Gaussian density fitting with the overlap metric
'''

def make_kptij_lst(kpts, kpts_band = None):
  if kpts_band is None:
      kband_uniq = numpy.zeros((0,3))
  else:
      kband_uniq = [k for k in kpts_band if len(member(k, kpts))==0]
  kptij_lst = [(ki, kpts[j]) for i, ki in enumerate(kpts) for j in range(i+1)]
  kptij_lst.extend([(ki, kj) for ki in kband_uniq for kj in kpts])
  kptij_lst.extend([(ki, ki) for ki in kband_uniq])
  kptij_lst = numpy.asarray(kptij_lst)
  
  return kptij_lst

def compute_j2c_sqrt(uniq_kptji_id, j2c, linear_dep_threshold=1e-9):
  try:
    # Cholesky decompose j2c
    j2c_sqrt = LA.cholesky(j2c, lower=False)
    j2ctag = 'CD'
  except LA.LinAlgError:
    w, v = LA.eigh(j2c)
    print("DF metric linear dependency for kpt ", uniq_kptji_id)
    print("cond = {}, drop {} bfns for lin_dep_threshold = {}".format(w[-1]/w[0], numpy.count_nonzero(w<linear_dep_threshold), linear_dep_threshold))
    v1 = v[:,w>linear_dep_threshold].conj().T
    v1 *= numpy.sqrt(w[w>linear_dep_threshold]).reshape(-1,1)
    j2c_sqrt = v1
    w = v = None
    j2ctag = 'eig'

  return j2c_sqrt, j2ctag

def _make_j2c_rsgdf(mydf, cell, auxcell, uniq_kpts, exx=False):
  from pyscf.pbc.df import rsdf_helper
  from pyscf.pbc.df.rsdf import get_aux_chg
  nao, naux = cell.nao_nr(), auxcell.nao_nr()
  # get charge of auxbasis
  if cell.dimension == 3:
    qaux = get_aux_chg(auxcell)
  else:
    qaux = numpy.zeros(auxcell.nao_nr())

  omega_j2c = abs(mydf.omega_j2c)
  j2c = rsdf_helper.intor_j2c(auxcell, omega_j2c, kpts=uniq_kpts)

  # Add (1) short-range G=0 (i.e., charge) part and (2) long-range part
  qaux2 = None
  g0_j2c = numpy.pi/omega_j2c**2./cell.vol
  mesh_j2c = mydf.mesh_j2c
  Gv, Gvbase, kws = cell.get_Gv_weights(mesh_j2c)
  b = cell.reciprocal_vectors()
  gxyz = lib.cartesian_prod([numpy.arange(len(x)) for x in Gvbase])
  ngrids = gxyz.shape[0]

  max_memory = max(2000, mydf.max_memory - lib.current_memory()[0])
  blksize = max(2048, int(max_memory*.5e6/16/auxcell.nao_nr()))

  for k, kpt in enumerate(uniq_kpts):
      # short-range charge part
      if is_zero(kpt) and cell.dimension == 3:
          if qaux2 is None:
              qaux2 = numpy.outer(qaux,qaux)
          j2c[k] -= qaux2 * g0_j2c
      # long-range part via aft
      coulG_lr = mydf.weighted_coulG(kpt, False, mesh_j2c, omega_j2c)
      for p0, p1 in lib.prange(0, ngrids, blksize):
          aoaux = ft_ao.ft_ao(auxcell, Gv[p0:p1], None, b, gxyz[p0:p1],
                              Gvbase, kpt).T
          LkR = numpy.asarray(aoaux.real, order='C')
          LkI = numpy.asarray(aoaux.imag, order='C')
          aoaux = None

          if is_zero(kpt):  # kpti == kptj
              j2c[k] += lib.ddot(LkR*coulG_lr[p0:p1], LkR.T)
              j2c[k] += lib.ddot(LkI*coulG_lr[p0:p1], LkI.T)
          else:
              j2cR, j2cI = df.df_jk.zdotCN(LkR*coulG_lr[p0:p1],
                                           LkI*coulG_lr[p0:p1], LkR.T, LkI.T)
              j2c[k] += j2cR + j2cI * 1j

          LkR = LkI = None

  coulG_lr = None

  return j2c

# make_j2c is updated to be consistent with PySCF 2.0.1 
# where the range of lattice Ls vectors are improved. 
# This version should be compatible with both PySCF 1.7 and 2.0
def _make_j2c_gdf(mydf, cell, auxcell, uniq_kpts, exx=False):
  import importlib
  if importlib.find_loader('pyscf.pbc.df.gdf_builder') is not None :
      fused_cell, fuse = gdf_builder.fuse_auxcell(auxcell, aft.estimate_eta(cell, cell.precision))
  else : 
      fused_cell, fuse = df.fuse_auxcell(mydf, auxcell)

  nao = cell.nao_nr()
  naux = auxcell.nao_nr()
  mesh = mydf.mesh
  Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
  b = cell.reciprocal_vectors()
  gxyz = lib.cartesian_prod([numpy.arange(len(x)) for x in Gvbase])
  ngrids = gxyz.shape[0]
  
  # Real space integrals within an given amount of repeated unit cell images 
  j2c = fused_cell.pbc_intor('int2c2e', hermi=0, kpts=uniq_kpts) 

  max_memory = max(2000, mydf.max_memory - lib.current_memory()[0])
  blksize = max(2048, int(max_memory*.5e6/16/fused_cell.nao_nr()))
  # Long-range contributions of the j2c integral are evaluated analytically in the reciprocal basis, 
  # i.e. the plane-wave basis. The G=0 contribution is excluded unless specified.
  for k, kpt in enumerate(uniq_kpts):
    if exx == 'ewald': 
      print("Enable Ewald correction!")
      coulG = mydf.weighted_coulG(kpt, exx, mesh)
    else:
      coulG = mydf.weighted_coulG(kpt, False, mesh)
    for p0, p1 in lib.prange(0, ngrids, blksize):
      aoaux = ft_ao.ft_ao(fused_cell, Gv[p0:p1], None, b, gxyz[p0:p1], Gvbase, kpt).T
      LkR = numpy.asarray(aoaux.real, order='C')
      LkI = numpy.asarray(aoaux.imag, order='C')
      aoaux = None

      if is_zero(kpt):  # kpti == kptj
        j2c_p  = lib.ddot(LkR[naux:]*coulG[p0:p1], LkR.T)
        j2c_p += lib.ddot(LkI[naux:]*coulG[p0:p1], LkI.T)
      else:
        j2cR, j2cI = zdotCN(LkR[naux:]*coulG[p0:p1],
                            LkI[naux:]*coulG[p0:p1], LkR.T, LkI.T)
        j2c_p = j2cR + j2cI * 1j
      j2c[k][naux:] -= j2c_p
      j2c[k][:naux,naux:] -= j2c_p[:,:naux].conj().T
      j2c_p = LkR = LkI = None
    # Symmetrizing the matrix is not must if the integrals converged.
    # Since symmetry cannot be enforced in the pbc_intor('int2c2e'),
    # the aggregated j2c here may have error in hermitian if the range of
    # lattice sum is not big enough.
    j2c[k] = (j2c[k] + j2c[k].conj().T) * .5
    j2c[k] = fuse(fuse(j2c[k]).T).T

  return j2c

def sqrt_j2c(mydf, j2c):
  j2ctags = []
  for iq in range(len(j2c)):
    # j2c_sqrt: (naux_effective, naux). naux_effective <= naux due to linear dependency
    j2c[iq], tag = compute_j2c_sqrt(iq, j2c[iq], mydf.linear_dep_threshold)
    j2ctags.append(tag)

  return j2c, j2ctags

def make_j2c_sqrt(mydf, cell):
  make_j2c = _make_j2c_rsgdf

  kmesh_scaled = cell.get_scaled_kpts(mydf.kpts)
  kptij_lst = make_kptij_lst(kmesh_scaled)
  nkptij = len(kptij_lst)
  kptis = kptij_lst[:,0]
  kptjs = kptij_lst[:,1]
  kpt_ji = kptjs - kptis
  # Fold kpt_ji back to [-0.5, 0.5] BZ notation
  kpt_ji = comm.fold_back_to_1stBZ(kpt_ji)
  # uniq_qs: unique q-points, # uniq_inverse: kptij -> q
  uniq_qs, uniq_index, uniq_inverse = unique(kpt_ji)
  q_ir_list, q_index, _, q_conj_list = comm.inversion_sym(uniq_qs)
  print("number of k-pairs: {}".format(len(kptij_lst)))
  print("number of uniq q-pionts: ", len(uniq_qs))
  print("number of uniq q-points w/ the inversion symmetry: ", len(q_ir_list))

  kptij_lst = cell.get_abs_kpts(kptij_lst)
  uniq_qs = cell.get_abs_kpts(uniq_qs)

  # j2c: [q_ir_list][naux, naux]
  j2c = make_j2c(mydf, cell, mydf.auxcell, uniq_qs[q_ir_list])
  j2c, j2ctags = sqrt_j2c(mydf, j2c)

  return j2c, uniq_qs


def make_j3c_outcore(mydf, cell, basename = 'df_int', rsgdf=False, j2c_sqrt=True, exx=False):
  '''
  The outcore version of make_j3c
  '''
  import os
  if not rsgdf:
    make_j2c = _make_j2c_gdf
  else:
    make_j2c = _make_j2c_rsgdf
  nao, naux = cell.nao_nr(), mydf.auxcell.nao_nr()

  a_lattice = cell.lattice_vectors() / (2*numpy.pi)
  kmesh_scaled = cell.get_scaled_kpts(mydf.kpts)
  kptij_lst = make_kptij_lst(kmesh_scaled)
  kptij_idx = [(i, j) for i in range(mydf.kpts.shape[0]) for j in range(i+1)]
  kptij_idx = numpy.asarray(kptij_idx)
  nkptij = len(kptij_lst)
  kptis = kptij_lst[:,0]
  kptjs = kptij_lst[:,1]
  kpt_ji = kptjs - kptis
  kpt_ji = comm.fold_back_to_1stBZ(kpt_ji)
  # uniq_qs: unique q-points, # uniq_inverse: kptij -> q
  uniq_qs, uniq_index, uniq_inverse = unique(kpt_ji)
  q_ir_list, q_index, _, q_conj_list = comm.inversion_sym(uniq_qs)
  # Reduced kpair list
  kptis, kptjs = cell.get_abs_kpts(kptis), cell.get_abs_kpts(kptjs)
  kij_conj, kij_trans = int_utils.kpair_reduced_lists(kptis, kptjs, kptij_idx, mydf.kpts, a_lattice)
  kpair_irre_list = numpy.argwhere(kij_conj == kij_trans)[:,0]
  num_kpair_stored = len(kpair_irre_list)
  print("number of k-pairs: {}".format(len(kptij_lst)))
  print("number of reduced k-pairs: ", num_kpair_stored)
  print("number of uniq q-pionts: ", len(uniq_qs))
  print("number of uniq q-points w/ the inversion symmetry: ", len(q_ir_list))

  kptij_lst = cell.get_abs_kpts(kptij_lst)
  uniq_qs = cell.get_abs_kpts(uniq_qs)

  single_rho_size = nao**2 * naux * 16
  full_rho_size   = (num_kpair_stored * single_rho_size)
  chunk_size = int_utils.compute_partitioning(full_rho_size, num_kpair_stored)
  num_chunks = int(numpy.ceil(num_kpair_stored / chunk_size))
  print("Chunk size: {}, number of chunks: {}".format(chunk_size, num_chunks))

  # j2c: [q_ir_list][naux, naux]
  print("Computing j2c...")
  j2c = make_j2c(mydf, cell, mydf.auxcell, uniq_qs[q_ir_list], exx)
  j2c, j2ctags = sqrt_j2c(mydf, j2c)
  print("Done!")

  # ovlp_2c1e: [q_ir_list][naux, naux]
  # Since the memory requirement is not high, this will be stored on the fly 
  print("Computing ovlp_2c1e...")
  ovlp_2c1e_aux = mydf.auxcell.pbc_intor('int1e_ovlp', hermi=1, kpts=uniq_qs[q_ir_list])
  print("Done!")

  # loop over chunks
  filename = basename + "/meta.h5"
  os.system("sync")
  if os.path.exists(basename):
    os.system("rm -r " + basename)  
    os.system("sync")
  os.system("mkdir -p " + basename)

  ovlp_3c1e = numpy.zeros((chunk_size, nao*nao, naux), dtype=complex)
  c0 = 0
  for i in range(num_chunks):
    c1 = c0 + chunk_size
    if c1 >= num_kpair_stored:
      c1 = num_kpair_stored
    local_kpairs_num = c1 - c0
    kij_idx_local = kpair_irre_list[c0:c1]
    ovlp_3c1e[:local_kpairs_num] = incore.aux_e2(cell, mydf.auxcell, intor='int3c1e', kptij_lst=kptij_lst[kij_idx_local])
    ovlp_3c1e = ovlp_3c1e.reshape(-1, nao*nao, naux)
    
    # Combine j2c, ovlp_2c1e, and ovlp_3c1e
    j3c = numpy.zeros((chunk_size, naux, nao, nao), dtype=complex)
    for ik in range(local_kpairs_num):
      # find the corresponding q-point
      kij_idx = kij_idx_local[ik]
      q_idx = uniq_inverse[kij_idx]
      iq_red = numpy.argwhere(q_ir_list == q_index[q_idx])[0][0]
      j2c_local = j2c[iq_red] if q_conj_list[q_idx] == 0 else j2c[iq_red].conj()
      ovlp_2c1e_local = ovlp_2c1e_aux[iq_red] if q_conj_list[q_idx] == 0 else ovlp_2c1e_aux[iq_red].conj()

      # df_coef = (S^-1) * ovlp_3c1e: (naux, nao*nao)
      df_coef = LA.solve(ovlp_2c1e_local, ovlp_3c1e[ik].T)
      # j3c_buffer: (naux_effective, nao*nao).
      j3c[ik] = numpy.dot(j2c_local, df_coef).reshape(-1, nao, nao)

    VQ = h5py.File(basename + "/VQ_{}.h5".format(c0), 'w')
    VQ["{}".format(c0)] = j3c.view(float)
    VQ.close()
    c0 += chunk_size
    ovlp_3c1e[:] = 0.0

  data = h5py.File(filename, "w")
  data["chunk_size"] = chunk_size
  #data["chunk_indices"] = numpy.array(chunk_indices)
  data["grid/conj_pairs_list"] = kij_conj
  data["grid/trans_pairs_list"] = kij_trans
  data["grid/kpair_irre_list"] = kpair_irre_list
  data["grid/kpair_idx"] = kptij_idx
  data["grid/num_kpair_stored"] = num_kpair_stored
  data["grid/k_mesh"] = mydf.kpts
  data["grid/k_mesh_scaled"] = cell.get_scaled_kpts(mydf.kpts)
                                                             
  data.close()
  print("Integrals have been computed and stored into {}".format(basename))

  if j2c_sqrt:
    return j2c, uniq_qs
  else:
    return 0

def make_j3c(mydf, cell, j2c_sqrt=True, exx=False):
  '''
  The inefficient incore version of make_j3c
  '''
  make_j2c = _make_j2c_gdf
  nao = cell.nao_nr()
  naux = mydf.auxcell.nao_nr()
  
  kmesh_scaled = cell.get_scaled_kpts(mydf.kpts)
  kptij_lst = make_kptij_lst(kmesh_scaled)
  nkptij = len(kptij_lst)
  kptis = kptij_lst[:,0]
  kptjs = kptij_lst[:,1]
  kpt_ji = kptjs - kptis
  # Fold kpt_ji back to [-0.5, 0.5] BZ notation
  kpt_ji = comm.fold_back_to_1stBZ(kpt_ji)
  # uniq_qs: unique q-points, # uniq_inverse: kptij -> q
  uniq_qs, uniq_index, uniq_inverse = unique(kpt_ji)
  q_ir_list, q_index, _, q_conj_list = comm.inversion_sym(uniq_qs)
  print("number of k-pairs: {}".format(len(kptij_lst)))
  print("number of uniq q-pionts: ", len(uniq_qs))
  print("number of uniq q-points w/ the inversion symmetry: ", len(q_ir_list))
  
  kptij_lst = cell.get_abs_kpts(kptij_lst)
  uniq_qs = cell.get_abs_kpts(uniq_qs)

  # j2c: [q_ir_list][naux, naux]
  j2c = make_j2c(mydf, cell, mydf.auxcell, uniq_qs[q_ir_list], exx)

  # ovlp_2c1e: [q_ir_list][naux, naux]
  ovlp_2c1e_aux = mydf.auxcell.pbc_intor('int1e_ovlp', hermi=1, kpts=uniq_qs[q_ir_list])

  # ints_3c1e: [kptij_lst][nao*nao, naux]
  ints_3c1e = incore.aux_e2(cell, mydf.auxcell, intor='int3c1e', kptij_lst=kptij_lst)
  if nkptij == 1:
    ints_3c1e = ints_3c1e.reshape((1,)+ints_3c1e.shape)

  # Combine j2c, ints_2c1e, and ints_3c1e

  # j2c_sqrt: (naux_effective, naux). naux_effective <= naux due to linear dependency
  j2ctags = []
  for iq in range(len(q_ir_list)):
    j2c[iq], tag = compute_j2c_sqrt(iq, j2c[iq], mydf.linear_dep_threshold)
    j2ctags.append(tag)

  j3c = numpy.zeros((nkptij, naux, nao, nao), dtype=complex)
  for uniq_q_id in range(len(uniq_qs)):
    adapted_ji_idx = numpy.where(uniq_inverse == uniq_q_id)[0]
    print("uniq_kptij_id = {}".format(uniq_q_id))
    print("adapted_ji_idx = {}".format(adapted_ji_idx))

    iq_red = numpy.argwhere(q_ir_list == q_index[uniq_q_id])[0][0]
    j2c_local = j2c[iq_red] if q_conj_list[uniq_q_id] == 0 else j2c[iq_red].conj()
    ovlp_2c1e_local = ovlp_2c1e_aux[iq_red] if q_conj_list[uniq_q_id] == 0 else ovlp_2c1e_aux[iq_red].conj()

    for k, ji in enumerate(adapted_ji_idx):
      # ovlp_3c1e: [nao*nao, naux]
      ovlp_3c1e = ints_3c1e[ji]
      # df_coef = (S^-1) * ovlp_3c1e: (naux, nao*nao)
      df_coef = LA.solve(ovlp_2c1e_local, ovlp_3c1e.T)
      # j3c_buffer: (naux_effective, nao*nao).
      j3c_buffer = numpy.dot(j2c_local, df_coef)
      j3c[ji,:j3c_buffer.shape[0]] = j3c_buffer.reshape(-1, nao, nao)

  if j2c_sqrt:
    return j3c, kptij_lst, j2c, uniq_qs
  else:
    return j3c, kptij_lst


def check_eri(j3c1, j3c2, kptij_lst):
  nkptij = len(kptij_lst)                                  
  kptis = kptij_lst[:,0]
  kptjs = kptij_lst[:,1]
  kpt_ji = kptjs - kptis
  # uniq_qs: unique q-points, # uniq_inverse: kptij -> q
  uniq_qs, uniq_index, uniq_inverse = unique(kpt_ji)

  for uniq_kptji_id in range(len(uniq_qs)):
    print("qid = {}".format(uniq_kptji_id))
    adapted_ji_idx = numpy.where(uniq_inverse == uniq_kptji_id)[0]
    for k, ji in enumerate(adapted_ji_idx):
      eri1 = lib.einsum('Lpq,Lsr->pqrs', j3c1[ji], j3c1[ji].conj())
      eri2 = lib.einsum('Lpq,Lsr->pqrs', j3c2[ji], j3c2[ji].conj())
      diff = eri1 - eri2
      print(numpy.max(numpy.abs(diff)))
      if uniq_kptji_id == 2 and k == 0:
        print("eri_gdf_S:")
        print(eri1)
        print("eri_Coulomb")
        print(eri2)
        print("-----------")

def transformation_PW_to_auxbasis(mydf, cell, j2c_sqrt, qs):
  from functools import reduce
  nao, NQ = cell.nao_nr(), mydf.auxcell.nao_nr() 

  kmesh_scaled = cell.get_scaled_kpts(mydf.kpts)
  qs_scaled = cell.get_scaled_kpts(qs)
  ir_list, *_ = comm.inversion_sym(kmesh_scaled)
  q_ir_list, q_index, _, q_conj_list = comm.inversion_sym(qs_scaled)
  kmesh_reduced = mydf.kpts[ir_list]
  
  print("number of uniq q-pionts: ", len(qs))
  print("size of j2c_sqrt: {}".format(len(j2c_sqrt)))
  print("number of uniq q-points w/ the inversion symmetry: ", len(q_ir_list))
  
  # ints_2c1e_aux
  ints_2c1e_aux = mydf.auxcell.pbc_intor('int1e_ovlp', hermi=1, kpts=kmesh_reduced)
  
  AqQ = numpy.zeros((len(kmesh_reduced), NQ), dtype=complex)
  q_reduced = numpy.zeros((len(kmesh_reduced),3), dtype=float)
  q_scaled_reduced = numpy.zeros((len(kmesh_reduced),3), dtype=float)
  # q from k_mesh_reduced
  for ik, q in enumerate(kmesh_reduced):
    q_scaled = cell.get_scaled_kpts(q)
  
    # Test: Make sure q is in [-0.5, 0.5] 
    q_scaled = [comm.wrap_1stBZ(l) for l in q_scaled]
    q = cell.get_abs_kpts(q_scaled)
  
    # Chi_G0: (NQ)
    Chi_G0 = ft_ao.ft_ao(mydf.auxcell, Gv=numpy.zeros((1,3)), kpt=q)[0]
    S_aux_inv = LA.inv(ints_2c1e_aux[ik])
    uniq_q_id = -1
    # q from qs
    for iq2, q2_scaled in enumerate(qs_scaled):
      # q = [0.0, 1.0]
      q2_scaled = [comm.wrap_k(l) for l in q2_scaled]
      # q = [-0.5, 0.5]
      q2_scaled = [comm.wrap_1stBZ(l) for l in q2_scaled]
      if numpy.allclose(q_scaled, q2_scaled):
        iq2_red = numpy.argwhere(q_ir_list == q_index[iq2])[0][0]
        U = j2c_sqrt[iq2_red] if q_conj_list[iq2] == 0 else j2c_sqrt[iq2_red].conj()
        uniq_q_id = iq2
        break
    if U is None:
      raise ValueError("Error: No matching j2c_sqrt is found!")
    # A^q_Q = (U)^q * (S^{-1})^{q} * Chi_G0*
    q_reduced[ik] = q
    q_scaled_reduced[ik] = q_scaled
    # U: (naux_effective, naux), S_aux_inv: (naux, naux), Chi_G0: (naux)
    AqQ[ik, :U.shape[0]] = reduce(numpy.dot, (U, S_aux_inv, Chi_G0.conj()))
    AqQ[ik] *= (numpy.linalg.norm(q) / numpy.sqrt(4*numpy.pi*cell.vol))
    U = None
  
  return AqQ, q_reduced, q_scaled_reduced
