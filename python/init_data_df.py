
import pyManyBody as pymb

pyscf_init = pymb.pyscf_init()

if pyscf_init.args.job == "init":
    pyscf_init.mean_field_input()
elif pyscf_init.args.job == "sym_path":
    pyscf_init.evaluate_high_symmetry_path()
elif pyscf_init.args.job == "ewald_corr":
    pyscf_init.evaluate_second_order_ewald()

print("Done")
