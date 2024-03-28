import numpy as np
from pyscf import gto as mgto
from pyscf import scf as mscf
from pyscf import dft as mdft
from pyscf import df as mdf
from pyscf.pbc import gto, df, tools
from pyscf.pbc import scf, dft
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point
import scipy.linalg as LA
from pyscf import lib
from pyscf.df import addons
import pyscf.lib.chkfile as chk
import h5py
import os
import sys
import argparse
import integral_utils as int_utils

np.set_printoptions(suppress=True, precision=5, linewidth=1500)


def wrap_k(k):
    while k < 0:
        k = 1 + k
    while (k - 9.9999999999e-1) > 0.0:
        k = k - 1
    return k


def parse_basis(basis_list):
    if len(basis_list) % 2 == 0:
        b = {}
        for atom_i in range(0, len(basis_list), 2):
            bas_i = basis_list[atom_i + 1]
            if os.path.exists(bas_i):
                with open(bas_i) as bfile:
                    bas = mgto.parse(bfile.read())
            # if basis specified as a standard basis
            else:
                bas = bas_i
            b[basis_list[atom_i]] = bas
        return b
    else:
        return basis_list[0]


def parse_geometry(g):
    res = ""
    if os.path.exists(g):
        with open(g) as gf:
            res = gf.read()
    else:
        res = g
    return res


a = '''4.0655,    0.0,    0.0
           0.0,    4.0655,    0.0
           0.0,    0.0,    4.0655'''
atoms = '''H -0.25 -0.25 -0.25
           H  0.25  0.25  0.25'''
basis = 'sto-3g'

parser = argparse.ArgumentParser(description="GF2 initialization script")
parser.add_argument("--a", type=str, default=a, help="lattice geometry")
parser.add_argument("--atom", type=str, default=atoms, help="poistions of atoms")
parser.add_argument("--spin", type=int, default=0, help="spin: Alpha - Beta")
parser.add_argument("--charge", type=int, default=0, help="total charge of the system")
parser.add_argument("--basis", type=str, nargs="*", default=["sto-3g"], help="basis sets definition. First specify atom then basis for this atom")
parser.add_argument("--auxbasis", type=str, nargs="*", default=[None], help="auxiliary basis")
parser.add_argument("--beta", type=float, default=None, help="Emperical parameter for even-Gaussian auxiliary basis")
parser.add_argument("--ecp", type=str, nargs="*", default=[None], help="effective core potentials")
parser.add_argument("--type", type=int, default=0, help="storage type")
#parser.add_argument("--scf", type=int, default=0, help="0: hf, 1: dft")
parser.add_argument("--xc", type=str, nargs="*", default=[None], help="XC functional")  #this takes a list as input.
parser.add_argument("--dm0", type=str, nargs="*", default=[None], help="initial guess for density matrix")
parser.add_argument("--tmp", type=str, nargs="*", default=["tmp.chk"], help="Initial guess for scf calculation")
parser.add_argument("--df_int", type=int, default=1, help="prepare density fitting integrals or not")

# Additional info for G0W0.
parser.add_argument("--store_vxc", type=int, default=0, help="Store vxc and vk for G0W0")
parser.add_argument("--gw", type=int, default=1, help="Perform G0W0 after input data generation")

args = parser.parse_args()

a = parse_geometry(args.a)
atoms = parse_geometry(args.atom)
spin = args.spin
ns = 1
charge = args.charge
basis = parse_basis(args.basis)
auxbasis = parse_basis(args.auxbasis)

ecp = parse_basis(args.ecp)
xc = parse_basis(args.xc)
dm_file = parse_basis(args.dm0)
tmp = parse_basis(args.tmp)
df_int = args.df_int
itype = args.type
store_vxc = args.store_vxc
dogw = bool(args.gw)


def cell():
    return mgto.M(
        atom=atoms,
        spin=spin,
        charge=charge,
        unit='A',
        basis=basis,
        verbose=6,
    )


def pcell():
    return gto.M(a=a, spin=spin, atom=atoms, unit='A', basis=basis, verbose=6, nucmod='G')


mycell = mgto.Mole(verbose=6)
mycell.atom = atoms
mycell.spin = spin
mycell.charge = charge
mycell.unit = 'A'
mycell.basis = basis
mycell.ecp = ecp

mycell.build()

kcell = gto.Cell(verbose=0)
kcell.a = a
kcell.atom = atoms
kcell.spin = spin
kcell.charge = charge
kcell.unit = 'A'
kcell.basis = basis
kcell.kpts = kcell.make_kpts([1, 1, 1])
kcell.ecp = mycell.ecp
kcell.build()

nk = 1
kmesh = kcell.kpts

# number of orbitals
nao = mycell.nao_nr()
Zs = np.asarray(mycell.atom_charges())
print("Number of atoms: ", Zs.shape[0])
print("Effective nuclear charge of each atom: ", Zs)
atoms_info = np.asarray(mycell.aoslice_by_atom())
last_ao = atoms_info[:, 3]
print("aoslice_by_atom = ", atoms_info)
print("Last AO index for each atom = ", last_ao)

# GHF does not support density fittiong for now so run UHF to store integrals
print("Solve UHF")
mf = mscf.UHF(mycell).density_fit()
mf.max_cycle = 2
#mf.with_x2c.approx = 'None'
mf.diis_space = 16
mf.direct_scf_tol = 1e-10
#mf.with_df.auxbasis = auxbasis
mf.with_df._cderi_to_save = "cderi_mol.h5"
#mf.chkfile = 'tmp.chk'
mf.kernel()
auxcell = addons.make_auxmol(mycell, mf.with_df.auxbasis)

NQ = auxcell.nao_nr()
print("NQ", NQ)

if xc is not None:
    print("Solve GDFT")
    #mf = mdft.GKS(mycell).x2c().density_fit()
    mf = mdft.GKS(mycell).x2c1e()
    mf.max_cycle = 300
    #mf.with_x2c.approx = 'None'
    mf.diis_space = 16
    mf.xc = xc
    #mf.with_df.auxbasis = auxbasis
    #mf.with_df._cderi_to_save = "cderi_mol.h5"
    mf.chkfile = tmp
    if os.path.exists(tmp):
        dm = mf.from_chk(tmp)
        mf.kernel(dm)
    else:
        mf.kernel()
else:
    print("Solve GHF")
    #mf = mscf.GHF(mycell).x2c1e().density_fit()
    mf = mscf.GHF(mycell).x2c1e()
    mf.max_cycle = 300
    #mf.with_x2c.approx = 'None'
    mf.diis_space = 16
    mf.direct_scf_tol = 1e-10
    #mf.with_df.auxbasis = auxbasis
    #mf.with_df._cderi_to_save = "cderi_mol.h5"
    mf.chkfile = 'tmp.chk'
    if os.path.exists(tmp):
        dm = mf.from_chk(tmp)
        #mf=mscf.newton(mf)
        mf.kernel(dm)
    else:
        #mf=mscf.newton(mf)
        mf.kernel()

# Get Overlap and Fock matrices
hf_dm_mol = mf.make_rdm1() * complex(1, 0)
S_mol = mf.get_ovlp() * complex(1, 0)
T_mol = mf.get_hcore() * complex(1, 0)
if xc is None:
    vhf_mol = mf.get_veff(dm=hf_dm_mol) * complex(1, 0)
else:
    vhf_mol = mf.get_veff(dm=hf_dm_mol) * complex(1, 0)
F_mol = mf.get_fock(T_mol, S_mol, vhf_mol, hf_dm_mol) * complex(1, 0)

# Convert results into Gamma-point formulation
hf_dm = hf_dm_mol.reshape((
    1,
    1,
) + hf_dm_mol.shape)
S = S_mol.reshape((
    1,
    1,
) + S_mol.shape)
T = T_mol.reshape((
    1,
    1,
) + T_mol.shape)
F = F_mol.reshape((
    1,
    1,
) + F_mol.shape)
vhf = vhf_mol.reshape((
    1,
    1,
) + vhf_mol.shape)

ind = np.arange(1)
weight = np.array([1 for i in range(1)])
num_ik = 1
ir_list = ind
conj_list = np.zeros((1))

# Integrals into Gamma-point formulation
if bool(df_int):
    h_in = h5py.File("cderi_mol.h5", 'a')
    h_out = h5py.File("cderi.h5", 'w')

    j3c_obj = h_in["/j3c"]
    if not isinstance(j3c_obj, h5py.Dataset):  # not a dataset
        if isinstance(j3c_obj, h5py.Group):  # pyscf >= 2.1
            h_in.copy(h_in["/j3c"], h_out, "j3c/0")
        else:
            raise ValueError("Unknown structure of cderi_mol.h5. Perhaps, PySCF upgrade went badly...")
    else:  # pyscf < 2.1
        h_in.copy(h_in["/j3c"], h_out, "j3c/0/0")

    kptij = np.zeros((1, 2, 3))
    h_out["j3c-kptij"] = kptij

    h_in.close()
    h_out.close()

    #nk = 1
    #kmesh = kcell.make_kpts([nk, nk, nk])
    # Use gaussian density fitting to get fitted densities
    mydf = df.GDF(kcell)
    #mydf.auxbasis = auxbasis
    int_utils.compute_integrals(gto.M(a="1 0 0\n 0 1 0\n 0 0 1", atom=mycell.atom, basis=mycell.basis), mydf, kmesh, nao, None, "df_hf_int",
                                "cderi.h5", True)

if store_vxc == 1:

    #dm for couloumb vk
    dm = np.array(hf_dm_mol)

    #vxc Veff -vj
    vxc = mf.get_veff() - mf.get_j()

    #temp_hf for Hartree forck exchange vk extraction
    # define the hf object but not analyzed. only used for vk extraction.
    temp_kghf = mscf.GHF(mycell).x2c1e()

    vk = temp_kghf.get_veff(mycell, dm)
    vj = temp_kghf.get_j(mycell, dm)
    vk = vk - vj

# Dump results
inp_data = h5py.File("input.h5", "w")
inp_data["grid/k_mesh"] = kmesh
inp_data["grid/k_mesh_scaled"] = kmesh
inp_data["grid/index"] = ind
inp_data["grid/weight"] = weight
inp_data["grid/ink"] = num_ik
inp_data["grid/ir_list"] = ir_list
inp_data["grid/conj_list"] = conj_list
inp_data["HF/Nk"] = 1
inp_data["HF/nk"] = 1
inp_data["HF/Energy"] = mf.e_tot
inp_data["HF/Energy_nuc"] = mf.energy_nuc()
inp_data["HF/Fock-k"] = F.view(float).reshape(F.shape[0], F.shape[1], F.shape[2], F.shape[3], 2)
inp_data["HF/Fock-k"].attrs["__complex__"] = np.int8(1)
inp_data["HF/S-k"] = S.view(float).reshape(S.shape[0], S.shape[1], S.shape[2], S.shape[3], 2)
inp_data["HF/S-k"].attrs["__complex__"] = np.int8(1)
inp_data["HF/H-k"] = T.view(float).reshape(T.shape[0], T.shape[1], T.shape[2], T.shape[3], 2)
inp_data["HF/H-k"].attrs["__complex__"] = np.int8(1)
inp_data["HF/madelung"] = 0.0
inp_data["HF/mo_energy"] = mf.mo_energy
inp_data["HF/mo_coeff"] = mf.mo_coeff

# additional data for G0W0
if store_vxc == 1:
    inp_data["HF/vxc"] = vxc
    inp_data["HF/vk"] = vk

inp_data["params/nao"] = nao
inp_data["params/nel_cell"] = mycell.nelectron
inp_data["params/nk"] = kmesh.shape[0]
inp_data["params/NQ"] = NQ

inp_data["mulliken/Zs"] = Zs
inp_data["mulliken/last_ao"] = last_ao
inp_data.close()

chk.save("input.h5", "Cell", mycell.dumps())
inp_data = h5py.File("dm.h5", "w")
inp_data["HF/dm-k"] = hf_dm.view(float).reshape(hf_dm.shape[0], hf_dm.shape[1], hf_dm.shape[2], hf_dm.shape[3], 2)
inp_data["HF/dm-k"].attrs["__complex__"] = np.int8(1)

inp_data.close()

print("Done")
