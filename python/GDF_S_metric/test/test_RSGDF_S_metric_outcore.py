import unittest
import sys
sys.path.append("../../")
import numpy as np
import scipy.linalg as LA
import h5py
from functools import reduce 

import integral_utils as int_utils
import common_utils as comm
import GDF_S_metric.GDF_S_metric as gdf_S

from pyscf import lib
from pyscf import gto as mol_gto
from pyscf.df import addons
from pyscf.pbc import gto, scf, cc, df

def get_j3c_coulomb(mydf, kptij_lst):
    naux = mydf.get_naoaux()
    nao = cell.nao_nr()
    j3c_coulomb = np.zeros((len(kptij_lst), naux, nao, nao), dtype=complex)
    for ij in range(len(kptij_lst)):
        ki = kptij_lst[ij,0]
        kj = kptij_lst[ij,1]
        p1 = 0
        for XXX in mydf.sr_loop((ki,kj), compact=False):
            LpqR = XXX[0]
            LpqI = XXX[1]
            Lpq = (LpqR + LpqI*1j).reshape(LpqR.shape[0], nao, nao)
            p0, p1 = p1, p1 + LpqR.shape[0]
            j3c_coulomb[ij,p0:p1] = Lpq.reshape(-1,nao,nao)

    return j3c_coulomb

cell = gto.Cell()
cell.atom='''
H -0.25 -0.25 -0.25
H  0.25  0.25  0.25
'''
cell.basis = 'sto-3g'
#cell.pseudo = 'gth-pade'
cell.a = '''
4.0655,    0.0,    0.0
0.0,    4.0655,    0.0
0.0,    0.0,    4.0655'''
cell.unit = 'A'
cell.verbose = 4
cell.build()

nk = 3
kmesh = cell.make_kpts([nk,nk,nk])
kptij_lst = gdf_S.make_kptij_lst(kmesh)

auxbasis = 'def2-svp-ri'
#auxbasis = df.aug_etb(cell, beta=2.0)

mydf = df.RSGDF(cell)
mydf.kpts = kmesh
mydf.auxbasis = auxbasis
mydf._rs_build()

j2c_sqrt, uniq_kpts = gdf_S.make_j3c_outcore(mydf, cell, rsgdf=True, basename='df_int', j2c_sqrt=True, exx=False)
f = h5py.File("df_int/VQ_0.h5",'a')
j3c = f["0"][()]
f.close()

class TestGDFinOvlpMetric(unittest.TestCase):
    def test_j3c(self):
        self.assertAlmostEqual(j3c[0].imag.sum(), 0.0, 7)
        self.assertAlmostEqual(j3c[0].real.sum(), 4.818213672757964, 7)
        self.assertAlmostEqual(j3c[4].real.sum(), 6.398303130326451, 7)
        self.assertAlmostEqual(j3c.sum(), 1126.1744871606143, 7)

    # j2c is different in RSGDF and GDF due to different renormalization 
    def test_j2c(self):
        self.assertAlmostEqual(j2c_sqrt[0].sum(), 75.21876032628603, 7)
        self.assertAlmostEqual(j2c_sqrt[1].real.sum(), 101.24240585068314, 7)

    # however j3c and eri remain the same! But why j3c?
    def test_eri(self):
        #j3c_coulomb = get_j3c_coulomb(mydf, kptij_lst)
        eri = lib.einsum('Lpq,Lsr->pqrs', j3c[0], j3c[0].conj())
        self.assertAlmostEqual(eri.real.sum(), 3.069377096091279, 7)
        self.assertAlmostEqual(eri.imag.sum(), 0.0, 7)
        eri = lib.einsum('Lpq,Lsr->pqrs', j3c[2], j3c[2].conj())
        self.assertAlmostEqual(eri.real.sum(), 3.0633298717895623, 7)

        
if __name__ == '__main__':
    unittest.main()
