
import green_mbtools.mint as pymb

pyscf_init = pymb.pyscf_init()

print(pyscf_init.args.job)

if "init" in pyscf_init.args.job:
    pyscf_init.mean_field_input()
if "sym_path" in pyscf_init.args.job:
    pyscf_init.evaluate_high_symmetry_path()
if "ewald_corr" in pyscf_init.args.job:
    pyscf_init.compute_twobody_finitesize_correction()

print("Done")
