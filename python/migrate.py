import h5py
import argparse
import os


parser = argparse.ArgumentParser(description="Migration util.")

parser.add_argument("--old_input_file", type=str, required=True, help="Name of the input file for pre-Green codes")
parser.add_argument("--old_integral_path", type=str, required=True, help="Path to a directory with density-fitted integrals for old code")
parser.add_argument("--new_input_file", type=str, required=True, help="Name of the input file for Green weak-coupling code")

args = parser.parse_args()

if not os.path.exists(args.old_input_file) or not os.path.isfile(args.old_input_file):
    print("pre-Green input file is not found")
    exit(-1)

if not os.path.exists(args.old_integral_path):
    print("pre-Green density-fitted integrals are not found")
    exit(-1)

old_input = h5py.File(args.old_input_file, "r")
old_meta = h5py.File(args.old_integral_path + "/meta.h5", "r")
new_input = h5py.File(args.new_input_file, "w")


old_input.copy("HF", new_input)
old_input.copy("params", new_input)
old_input.copy("grid", new_input)
old_input.copy("Cell", new_input)
if "mulliken" in old_input.keys():
    old_input.copy("mulliken", new_input)

X = old_input["HF/S-k"][()]
ns = X.shape[0]
nso = X.shape[2]

new_input["params/ns"] = ns
new_input["params/nso"] = nso

new_grid = new_input["/grid/"]
old_input["params"].copy("nk", new_grid)

meta_dsets = ["conj_pairs_list", "kpair_idx", "kpair_irre_list", "num_kpair_stored", "trans_pairs_list"]
for dset in meta_dsets:
    if dset in old_meta["/grid"].keys():
        old_meta["/grid"].copy(dset, new_grid)

old_input.close()
old_meta.close()
new_input.close()
