import numpy as np
from pyscf import gto as mgto
from pyscf.pbc import gto, df, tools
from pyscf.pbc import scf, dft
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point
import scipy.linalg as LA
from pyscf import lib
from pyscf.df import addons
import integral_utils as int_utils
import common_utils as comm
#from pyscf.pbc.tools import k2gamma
import pyscf.lib.chkfile as chk
import h5py
import os
import sys
import shutil
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
parser.add_argument("--ecp", type=str, nargs="*", default=[None], help="effective core potentials")
parser.add_argument("--pseudo", type=str, nargs="*", default=[None], help="pseudopotential")
parser.add_argument("--type", type=int, default=0, help="storage type")
parser.add_argument("--shift", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="mesh shift")
parser.add_argument("--center", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="mesh center")
parser.add_argument("--xc", type=str, nargs="*", default=[None], help="XC functional")
parser.add_argument("--dm0", type=str, nargs="*", default=[None], help="initial guess for density matrix")
parser.add_argument("--df_int", type=int, default=1, help="prepare density fitting integrals or not")
parser.add_argument("--cderi_path", type=str, default="cderi.h5", help="path for cderi file used in calculation")
parser.add_argument("--orth", type=int, default=0, help="Transform to orthogonal basis or not")
parser.add_argument("--beta", type=float, default=None, help="Emperical parameter for even-Gaussian auxiliary basis")
parser.add_argument("--active_space", type=int, nargs='+', default=None, help="active space orbitals")
parser.add_argument("--spin", type=int, default=0, help="Local spin")
parser.add_argument("--newton", type=int, default=0, help="Toggle newton solver")

args = parser.parse_args()

a = comm.parse_geometry(args.a)
atoms = comm.parse_geometry(args.atom)
# number of k-points in each direction for Coulomb integrals
nk = args.nk
# number of k-points in each direction to evaluate Coulomb kernel
Nk = args.Nk
basis = comm.parse_basis(args.basis)
auxbasis = comm.parse_basis(args.auxbasis)
ecp = comm.parse_basis(args.ecp)
pseudo = comm.parse_basis(args.pseudo)
xc = comm.parse_basis(args.xc)
dm_file = comm.parse_basis(args.dm0)
df_int = args.df_int
cderi_path = args.cderi_path
orth = args.orth
itype = args.type
shift = np.array(args.shift)
center = np.array(args.center)
spin = args.spin

# Parse additional arguments
newton = args.newton


def cell():
    return gto.M(a=a, atom=atoms, unit='A', basis=basis, ecp=ecp, pseudo=pseudo, verbose=7, spin=spin, nucmod='G')


mycell = cell()
# number of orbitals per cell
nao = mycell.nao_nr()
Zs = np.asarray(mycell.atom_charges())
print("Number of atoms: ", Zs.shape[0])
print("Effective nuclear charge of each atom: ", Zs)
atoms_info = np.asarray(mycell.aoslice_by_atom())
last_ao = atoms_info[:, 3]
print("aoslice_by_atom = ", atoms_info)
print("Last AO index for each atom = ", last_ao)
# init k-points mesh for GDF
kmesh = mycell.make_kpts([nk, nk, nk], scaled_center=center)

for i, kk in enumerate(kmesh):
    ki = kmesh[i]
    ki = mycell.get_scaled_kpts(ki) + shift
    ki = [comm.wrap_k(l) for l in ki]
    kmesh[i] = mycell.get_abs_kpts(ki)

for i, ki in enumerate(kmesh):
    ki = mycell.get_scaled_kpts(ki)
    ki = [comm.wrap_k(l) for l in ki]
    ki = mycell.get_abs_kpts(ki)
    kmesh[i] = ki


def find_pos(k):
    for l, kl in enumerate(kmesh):
        if np.allclose(k, kl):
            return l
    err = "can't find k-point index for {}".format(k)
    raise ValueError(err)


print(kmesh)
print(mycell.get_scaled_kpts(kmesh))

print("Compute irreducible k-points")


def wrap_1stBZ(k):
    while k < -0.5:
        k = k + 1
    while (k - 4.9999999999e-1) > 0.0:
        k = k - 1
    return k


k_ibz = mycell.make_kpts([nk, nk, nk], scaled_center=center)
ind = np.arange(np.shape(k_ibz)[0])
weight = np.zeros(np.shape(k_ibz)[0])
for i, ki in enumerate(k_ibz):
    ki = mycell.get_scaled_kpts(ki)
    ki = [wrap_1stBZ(l) for l in ki]
    k_ibz[i] = ki

# Time-reversal symmetry
Inv = (-1) * np.identity(3)
for i, ki in enumerate(k_ibz):
    ki = np.dot(Inv, ki)
    ki = [wrap_1stBZ(l) for l in ki]
    for l, kl in enumerate(k_ibz[:i]):
        if np.allclose(ki, kl):
            k_ibz[i] = kl
            ind[i] = l
            break

uniq = np.unique(ind, return_counts=True)
for i, k in enumerate(uniq[0]):
    weight[k] = uniq[1][i]
ir_list = uniq[0]

# Mark down time-reversal-reduced k-points
conj_list = np.zeros(nk**3)
for i, k in enumerate(ind):
    if i != k:
        conj_list[i] = 1
num_ik = np.shape(uniq[0])[0]
'''
Generate integrals for mean-field calculations
'''
mydf = df.GDF(mycell)
if auxbasis is not None:
    mydf.auxbasis = auxbasis
elif args.beta is not None:
    mydf.auxbasis = df.aug_etb(mycell, beta=args.beta)
# Coulomb kernel mesh
if Nk > 0:
    mydf.mesh = [Nk, Nk, Nk]
mydf.kpts = kmesh
if os.path.exists(cderi_path):
    mydf._cderi = cderi_path
else:
    mydf._cderi_to_save = cderi_path
    mydf.build()
auxcell = addons.make_auxmol(mycell, mydf.auxbasis)
NQ = auxcell.nao_nr()
'''
Read density matrix from smaller kmesh
'''


def read_dm(dm0, dm_file):
    nao = dm0.shape[-1]
    nkpts = dm0.shape[1]
    dm = np.zeros((nao, nao), dtype=np.complex128)
    f = h5py.File(dm_file, 'r')
    dm[:, :] = f['/dm_gamma'][:]
    f.close()
    dm_kpts = np.repeat(dm[None, :, :], nkpts, axis=0)
    return dm_kpts


#
# The X2C1e/sfX2C1e features are currently only available in
# “x2c1e_kpoints” branch in Chia-Nan's forker pyscf repository, https://github.com/cnyeh/pyscf.
# The branch is maintained such that it is consistent to the master branch in https://github.com/pyscf/pyscf.
#
if xc is not None:
    print("Solve LDA")
    mf = dft.KGKS(mycell, mydf.kpts).density_fit().x2c1e()
    mf.xc = xc
    #mf.max_memory = 10000
    mydf._cderi = cderi_path
    mf.kpts = mydf.kpts
    mf.with_df = mydf
    mf.with_x2c.approx = 'None'
    #mf.with_x2c.basis = mycell.basis
    mf.diis_space = 16
    mf.max_cycle = 50
    mf.chkfile = 'tmp.chk'
    if newton:
        mf = mf.newton()
    if os.path.exists("tmp.chk"):
        init_dm = mf.from_chk('tmp.chk')
        mf.kernel(init_dm)
    else:
        mf.kernel()
else:
    print("Solve HF")
    mf = dft.KGHF(mycell, mydf.kpts).density_fit().x2c1e()
    mf.xc = xc
    #mf.max_memory = 10000
    mydf._cderi = cderi_path
    mf.kpts = mydf.kpts
    mf.with_df = mydf
    mf.with_x2c.approx = 'None'
    #mf.with_x2c.basis = mycell.basis
    mf.diis_space = 16
    mf.max_cycle = 50
    mf.chkfile = 'tmp.chk'
    if newton:
        mf = mf.newton()
    if os.path.exists("tmp.chk"):
        init_dm = mf.from_chk('tmp.chk')
        mf.kernel(init_dm)
    else:
        mf.kernel()

# Get Overlap and Fock matrices
hf_dm = mf.make_rdm1() * np.complex128(1.0)
S = mf.get_ovlp() * np.complex128(1.0)
T = mf.get_hcore() * np.complex128(1.0)
if xc is not None:
    vhf = mf.get_veff() * np.complex128(1.0)
else:
    vhf = mf.get_veff(hf_dm) * np.complex128(1.0)
F = mf.get_fock(T, S, vhf, hf_dm) * np.complex128(1.0)

F = F.reshape((1, ) + F.shape)
S = S.reshape((1, ) + S.shape)
T = T.reshape((1, ) + T.shape)
dm_2use = hf_dm.copy()
hf_dm = hf_dm.reshape((1, ) + hf_dm.shape)

# Vxc and Vhf
vxc = np.array(mf.get_veff())
vj = np.array(mf.get_j(dm_kpts=dm_2use))
vxc -= vj
vxc = vxc.reshape((1, ) + vxc.shape)

kghf = scf.KGHF(mycell, mydf.kpts).density_fit().x2c1e()
kghf.with_df = mf.with_df
kghf._cderi = mf.with_df._cderi
kghf.with_df._cderi = mf.with_df._cderi
vk = kghf.get_veff(dm_kpts=dm_2use)
vj = kghf.get_j(dm_kpts=dm_2use)
vk -= vj
vk = vk.reshape((1, ) + vk.shape)

inp_data = h5py.File("input.h5", "w")
inp_data["grid/k_mesh"] = kmesh
inp_data["grid/k_mesh_scaled"] = mycell.get_scaled_kpts(kmesh)
inp_data["grid/index"] = ind
inp_data["grid/weight"] = weight
inp_data["grid/ink"] = num_ik
inp_data["grid/ir_list"] = ir_list
inp_data["grid/conj_list"] = conj_list
inp_data["HF/Nk"] = Nk
inp_data["HF/nk"] = nk
inp_data["HF/Energy"] = mf.e_tot
inp_data["HF/Energy_nuc"] = mf.cell.energy_nuc()
inp_data["HF/Fock-k"] = F.view(np.float64).reshape(F.shape[0], F.shape[1], F.shape[2], F.shape[3], 2)
inp_data["HF/Fock-k"].attrs["__complex__"] = np.int8(1)
inp_data["HF/S-k"] = S.view(np.float64).reshape(S.shape[0], S.shape[1], S.shape[2], S.shape[3], 2)
inp_data["HF/S-k"].attrs["__complex__"] = np.int8(1)
inp_data["HF/H-k"] = T.view(np.float64).reshape(T.shape[0], T.shape[1], T.shape[2], T.shape[3], 2)
inp_data["HF/H-k"].attrs["__complex__"] = np.int8(1)
inp_data["HF/madelung"] = tools.pbc.madelung(mycell, kmesh)
inp_data["HF/mo_energy"] = mf.mo_energy
inp_data["HF/mo_coeff"] = mf.mo_coeff
inp_data["HF/vxc"] = vxc
inp_data["HF/vk"] = vk

inp_data["mulliken/Zs"] = Zs
inp_data["mulliken/last_ao"] = last_ao

inp_data["params/nao"] = mycell.nao_nr()
inp_data["params/nel_cell"] = mycell.nelectron
inp_data["params/nk"] = kmesh.shape[0]
inp_data["params/NQ"] = NQ
if args.active_space is not None:
    inp_data["as/indices"] = np.array(args.active_space)
inp_data.close()

chk.save("input.h5", "Cell", mycell.dumps())
inp_data = h5py.File("dm.h5", "w")
inp_data["HF/dm-k"] = hf_dm.view(np.float64).reshape(hf_dm.shape[0], hf_dm.shape[1], hf_dm.shape[2], hf_dm.shape[3], 2)
inp_data["HF/dm-k"].attrs["__complex__"] = np.int8(1)
inp_data["dm_gamma"] = hf_dm[:, 0, :, :]
inp_data.close()

X_k = []
X_inv_k = []
for ik, k in enumerate(mydf.kpts):
    X_inv_k.append(np.eye(F.shape[2], dtype=complex))
    X_k.append(np.eye(F.shape[2], dtype=complex))

X_inv_k = np.asarray(X_inv_k).reshape(F.shape[1:])
X_k = np.asarray(X_k).reshape(F.shape[1:])
'''
Generate density-fitting integrals for correlated methods
'''
if bool(df_int):
    # Use gaussian density fitting to get fitted densities
    mydf = df.GDF(mycell)
    if auxbasis is not None:
        mydf.auxbasis = auxbasis
    elif args.beta is not None:
        mydf.auxbasis = df.aug_etb(mycell, beta=args.beta)
    # Coulomb kernel mesh
    if Nk > 0:
        mydf.mesh = [Nk, Nk, Nk]
    # Use Ewald for divergence treatment
    mydf.exxdiv = 'ewald'
    weighted_coulG_old = df.GDF.weighted_coulG
    df.GDF.weighted_coulG = int_utils.weighted_coulG_ewald
    int_utils.compute_integrals(mycell, mydf, kmesh, nao, X_k, "df_int", "cderi_ewald.h5", True)

    mydf = None
    # Use gaussian density fitting to get fitted densities
    mydf = df.GDF(mycell)
    if auxbasis is not None:
        mydf.auxbasis = auxbasis
    elif args.beta is not None:
        mydf.auxbasis = df.aug_etb(mycell, beta=args.beta)
    # Coulomb kernel mesh
    if Nk > 0:
        mydf.mesh = [Nk, Nk, Nk]
    df.GDF.weighted_coulG = weighted_coulG_old
    int_utils.compute_integrals(mycell, mydf, kmesh, nao, X_k, "df_hf_int", cderi_path, True)

print("Done")
