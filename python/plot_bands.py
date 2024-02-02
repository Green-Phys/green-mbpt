import numpy as np
import matplotlib.pyplot as plt
import h5py
from pyscf.pbc import gto
from ase.dft.kpoints import bandpath
import argparse

parser = argparse.ArgumentParser("Band-structure plotting")

parser.add_argument("--input_file", type=str, requireed=True, description="Name of the file containing analytically continued data evaluated on high-symmetry path")
parser.add_argument("--output_dir", type=str, requireed=True, description="Name of the directory to save band-strtructure plots")

args = parser.parse_args()

with h5py.File(args.output_dir, "r") as fff:
    if not ("G_tau_hs" is in fff) :
        raise RuntimeError("Input file does not contain G_tau_hs group. Check your input data!!!")
    DOS = fff["G_tau_hs/data"][()]
    mesh = fff["G_tau_hs/mesh"][()]

freqs = mesh
print(DOS.shape)
nomega=DOS.shape[0]
nspin=DOS.shape[1]
nk_again=DOS.shape[2]
norb=DOS.shape[3]
nk = nk_again
KDOS = np.einsum("wski->skw", -DOS.imag/np.pi)

if not os.path.isdir(args.output_dir):
   os.makedirs(args.output_dir)


TXDOS =np.zeros([nomega,2])
for s in range(nspin):
  for k in range(nk):
    #writing spectral functions to text
    TXDOS[:,0]=freqs
    TXDOS[:,1]=KDOS[s,k,:]
    np.savetxt(args.output_dir+"/dos_spin"+str(s)+"_k"+str(k)+".txt",TXDOS)

HartreeToEv = 27.211396641308
freqs_limit=mesh*HartreeToEv
mask=(freqs_limit<12) & (freqs_limit>-12)
KDOS=KDOS[0,:,mask]*HartreeToEv
path=np.array(range(0,nk_again))
plt.imshow(KDOS.T, aspect='auto', origin='lower', cmap='hot', extent=[path[0], path[-1], freqs_limit[mask][0], freqs_limit[mask][-1]])

plt.colorbar()
plt.xlabel('k-path')
plt.ylabel('Frequency')
plt.savefig(args.output_dir+"bands.png")

