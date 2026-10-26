"""
Build the mean-field G_tau for the H-chain fixture, reproducing the collaborator's
PR #77 example: beta = 25 Ha^-1, mu = 0 (fixed), restricted. Stores G_tau in data.h5.
G(iw)_k = [ (iw + mu) S_k - F_k ]^{-1} on the irreducible wedge, then IR -> tau.
"""
import h5py
import numpy as np
from pathlib import Path
from green_mbtools.pesto.ir import IR_factory

beta = 25.0
mu = 0.0
ir_file = Path.home() / "Documents/Codes/green-grids/data/ir/1e4.h5"
myir = IR_factory(beta, str(ir_file))
wsample = myir.wsample


def as_complex(a):
    a = np.asarray(a)
    if a.dtype.names is not None:
        return a['r'] + 1j * a['i']
    if np.iscomplexobj(a):
        return a
    if a.shape[-1] == 2:
        return a[..., 0] + 1j * a[..., 1]
    return a.astype(complex)


with h5py.File("input.h5", "r") as f:
    fock = as_complex(f["HF/Fock-k"][()])
    ovlp = as_complex(f["HF/S-k"][()])
    ir_list = f["symmetry/k/ibz2bz"][()]
    ink = int(f["symmetry/k/ink"][()])

ns, nk, nao = fock.shape[:3]
F = np.array([fock[0, k] for k in ir_list])
S = np.array([ovlp[0, k] for k in ir_list])

nw = wsample.shape[0]
g_iw = np.zeros((nw, ns, ink, nao, nao), dtype=complex)
for iw, w in enumerate(wsample):
    for k in range(ink):
        g_iw[iw, 0, k] = np.linalg.inv((1j * w + mu) * S[k] - F[k])

gtau = myir.w_to_tau(g_iw)
print(f"beta={beta} mu={mu}  G_tau shape: {gtau.shape}")

with h5py.File("data.h5", "w") as f:
    f["G_tau"] = gtau
print("wrote data.h5 [G_tau]")
