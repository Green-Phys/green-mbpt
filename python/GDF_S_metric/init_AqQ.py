import numpy as np
from pyscf.pbc import gto, df, tools

import common_utils as comm
import GDF_S_metric as gdf_S

import h5py
import argparse

# Default geometry
a = '''4.0655,    0.0,    0.0
           0.0,    4.0655,    0.0
           0.0,    0.0,    4.0655'''
atoms = '''H -0.25 -0.25 -0.25
           H  0.25  0.25  0.25'''
basis = 'sto-3g'

parser = argparse.ArgumentParser(description="GF2 initialization script")
parser.add_argument("--a", type=str, default=a, help="lattice geometry")
parser.add_argument("--atom", type=str, default=atoms, help="poistions of atoms")
parser.add_argument("--nk", type=int, default=3, help="number of k-points in each direction")
parser.add_argument("--Nk", type=int, default=0, help="number of plane-waves in each direction for integral evaluation")
parser.add_argument("--basis", type=str, nargs="*", default=["sto-3g"], help="basis sets definition. First specify atom then basis for this atom")
parser.add_argument("--auxbasis", type=str, nargs="*", default=[None], help="auxiliary basis")
parser.add_argument("--beta", type=float, default=None, help="Emperical parameter for even-Gaussian auxiliary basis")
parser.add_argument("--ecp", type=str, nargs="*", default=[None], help="effective core potentials")
parser.add_argument("--pseudo", type=str, nargs="*", default=[None], help="pseudopotential")
parser.add_argument("--shift", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="mesh shift")
parser.add_argument("--center", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="mesh center")
parser.add_argument("--spin", type=int, default=0, help="Local spin")


args = parser.parse_args()

a        = comm.parse_geometry(args.a)
atoms    = comm.parse_geometry(args.atom)
# number of k-points in each direction for Coulomb integrals
nk       = args.nk
# number of k-points in each direction to evaluate Coulomb kernel
Nk       = args.Nk
basis    = comm.parse_basis(args.basis)
auxbasis = comm.parse_basis(args.auxbasis)
ecp      = comm.parse_basis(args.ecp)
pseudo   = comm.parse_basis(args.pseudo)
shift    = np.array(args.shift)
center   = np.array(args.center)
spin     = args.spin

def cell():
    return gto.M(
        a = a,
        atom = atoms,
        unit = 'A',
        basis = basis,
        ecp = ecp,
        pseudo = pseudo,
        verbose = 7,
        spin = spin,
    )

mycell = cell()
nao = mycell.nao_nr()
Zs = np.asarray(mycell.atom_charges())
print("Number of atoms: ", Zs.shape[0])
print("Effective nuclear charge of each atom: ", Zs)
atoms_info = np.asarray(mycell.aoslice_by_atom())
last_ao = atoms_info[:,3]
print("aoslice_by_atom = ", atoms_info)
print("Last AO index for each atom = ", last_ao)
kmesh = mycell.make_kpts([nk, nk, nk], scaled_center=center)

for i,kk in enumerate(kmesh):
    ki = kmesh[i]
    ki = mycell.get_scaled_kpts(ki) + shift
    ki = [comm.wrap_k(l) for l in ki]
    kmesh[i] = mycell.get_abs_kpts(ki)

'''
Generate integrals for mean-field calculations
'''
mydf   = df.RSGDF(mycell)
if auxbasis is not None:
    mydf.auxbasis = auxbasis
elif args.beta is not None:
    mydf.auxbasis = df.aug_etb(mycell, beta=args.beta)
# Coulomb kernel mesh
if Nk > 0:
    mydf.mesh = [Nk, Nk, Nk]
mydf.kpts = kmesh
mydf._rs_build()
NQ = mydf.auxcell.nao_nr()

''' compute j2c_sqrt and corresponding qs '''
print("*** Comuting the transformation matrices from auxiliary to plane-wave at G=G'=0 to AqQ.h5...")
j2c_sqrt, qs = gdf_S.make_j2c_sqrt(mydf, mycell)

''' Transformation matrix from auxiliary basis to plane-wave '''
AqQ, q_reduced, q_scaled_reduced = gdf_S.transformation_PW_to_auxbasis(mydf, mycell, j2c_sqrt, qs)
q_abs = np.array([np.linalg.norm(qq) for qq in q_reduced])
q_abs = np.array([round(qq, 8) for qq in q_abs])

# Different prefactors for the GW finite-size correction for testing 
# In practice, the madelung constant is used, which decays as (1/nk). 
X = (6*np.pi**2)/(mycell.vol*len(kmesh))
X = (2.0/np.pi) * np.cbrt(X)

X2 = 2.0 * np.cbrt(1.0/(mycell.vol*len(kmesh)))

f = h5py.File("AqQ.h5", 'w')
f["AqQ"] = AqQ.view(np.float64).reshape(AqQ.shape[0], AqQ.shape[1], 2)
f["AqQ"].attrs["__complex__"] = np.int8(1)
f["qs"] = q_reduced
f["qs_scaled"] = q_scaled_reduced
f["q_abs"] = q_abs
f["X"] = X
f["X2"] = X2
f["madelung"] = tools.pbc.madelung(mycell, kmesh)
f.close()

print("Done")
