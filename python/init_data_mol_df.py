import logging
import green_mbtools.mint as pymb

logging.basicConfig(level=logging.INFO)

pyscf_init = pymb.pyscf_mol_init()

pyscf_init.mean_field_input()

print("Done")
