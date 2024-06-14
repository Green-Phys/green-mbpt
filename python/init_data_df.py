import logging
import green_mbtools.mint as pymb

logging.basicConfig(level=logging.INFO)

pyscf_init = pymb.pyscf_pbc_init()

if "init" in pyscf_init.args.job:
    pyscf_init.mean_field_input()
if "sym_path" in pyscf_init.args.job:
    pyscf_init.evaluate_high_symmetry_path()
if "ewald_corr" in pyscf_init.args.job:
    pyscf_init.compute_twobody_finitesize_correction()

print("Done")
