import unittest
import sys
sys.path.append("../../")
import numpy as np
import h5py
import scipy.linalg as LA
from functools import reduce 

import integral_utils as int_utils
import common_utils as comm
import GDF_S_metric.GDF_S_metric as gdf_S

from pyscf import lib
from pyscf import gto as mol_gto
from pyscf.df import addons
from pyscf.pbc import gto, scf, cc, df

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

mydf = df.GDF(cell)
mydf.kpts = kmesh
mydf.auxbasis = auxbasis
mydf.auxcell = df.df.make_modrho_basis(mydf.cell, mydf.auxbasis, mydf.exp_to_discard)

j3c, kptij_lst, j2c_sqrt, uniq_kpts = gdf_S.make_j3c(mydf, cell, j2c_sqrt=True, exx=False)
NQ = mydf.auxcell.nao_nr()

''' Transformation matrix from auxiliary basis to plane-wave '''
AqQ, q_reduced, q_scaled_reduced = gdf_S.transformation_PW_to_auxbasis(mydf, cell, j2c_sqrt, uniq_kpts)

class TestAq(unittest.TestCase):
    def test_Aq(self):
        self.assertAlmostEqual(AqQ[0].sum(), 0.0, 7)
        self.assertAlmostEqual(AqQ[1].real.sum(), 0.963789371034357, 7)
        self.assertAlmostEqual(AqQ[1].imag.sum(), 0.08557131232163924, 7)
        self.assertAlmostEqual(AqQ[4].real.sum(), 0.9552044033262287, 7)
        self.assertAlmostEqual(AqQ[4].imag.sum(), 0.009215981832345394, 7)
        self.assertAlmostEqual(AqQ.sum(), 12.64749309403998+2.0046225482088365j, 7)
  
    def test_Aq_identity(self):
        identity = lib.einsum('qQ,qQ->q', AqQ.conj(), AqQ)
        self.assertAlmostEqual(identity[0], 0.0, 10)
        self.assertAlmostEqual(identity[1], 0.24206605692594607, 7)
        self.assertAlmostEqual(identity[4], 0.19609446986919593, 7)

        
if __name__ == '__main__':
    unittest.main()
