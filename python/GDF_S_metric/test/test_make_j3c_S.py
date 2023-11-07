import unittest
import sys
sys.path.append("../../")
import numpy as np

import GDF_S_metric.GDF_S_metric as gdf_S
import GDF_S_metric.RSGDF_make_j3c_S as RSGDF_make_j3c_S

from pyscf import lib
from pyscf.pbc import gto, df

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

df.RSGDF._make_j3c = RSGDF_make_j3c_S._make_j3c_S_metric
mydf_S = df.RSGDF(cell)
mydf_S.kpts = kmesh
mydf_S.auxbasis = auxbasis
mydf_S._cderi_to_save = "cderi_S.h5"
mydf_S.build()
j3c_S = get_j3c_coulomb(mydf_S, kptij_lst)

print("j3c_S shape: {}".format(j3c_S.shape))

class TestGDFinOvlpMetric(unittest.TestCase):
    def test_j3c(self):
        self.assertAlmostEqual(j3c_S[0].imag.sum(), 0.0, 7)
        self.assertAlmostEqual(j3c_S[0].real.sum(), 4.818213672757964, 7)
        self.assertAlmostEqual(j3c_S[4].real.sum(), 6.107511043911971, 7)
        self.assertAlmostEqual(j3c_S[4].imag.sum(), 0.28771791198741037, 7)
        self.assertAlmostEqual(j3c_S.real.sum(), 2124.3229241488502, 7)
        self.assertAlmostEqual(j3c_S.imag.sum(), 20.145735130720155, 7)

    # however j3c and eri remain the same! But why j3c?
    def test_eri(self):
        eri = lib.einsum('Lpq,Lsr->pqrs', j3c_S[0], j3c_S[0].conj())
        self.assertAlmostEqual(eri.real.sum(), 3.069377096091279, 7)
        self.assertAlmostEqual(eri.imag.sum(), 0.0, 7)
        eri = lib.einsum('Lpq,Lsr->pqrs', j3c_S[2], j3c_S[2].conj())
        self.assertAlmostEqual(eri.real.sum(), 3.0633298717895623, 7)
        
if __name__ == '__main__':
    unittest.main()
