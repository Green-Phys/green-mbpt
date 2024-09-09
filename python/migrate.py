import argparse
import os

import h5py

def copy_input_data(old_input_file, new_input_file):
    '''
    copy all existing data from old file into a new file
    both files have to be open
    Parameters
    ----------
    old_input_file: old HDF5 file object
    new_input_file: new HDF5 file object
    '''
    for key in old_input_file.keys():
        old_input_file.copy(key, new_input_file)



def pregreen_version_0_2_4(args):
    '''
    Migrate data into version 0.2.4
    '''
    old_input = None
    new_input = None
    old_meta = h5py.File(args.old_integral_path[0] + "/meta.h5", "r")

    if args.old_input_file != args.new_input_file:
        old_input = h5py.File(args.old_input_file, "r")
        new_input = h5py.File(args.new_input_file, "a")
        copy_input_data(old_input, new_input)
    else:
        new_input = h5py.File(args.new_input_file, "a")
        old_input = new_input

    # get shapes for spin and spin-orbits
    X = old_input["HF/S-k"][()]
    ns = X.shape[0]
    nk = X.shape[1]
    nso = X.shape[2]

    if not "params/ns" in new_input:
        new_input["params/ns"] = ns
    if not "params/nso" in new_input:
        new_input["params/nso"] = nso
    if not "grid/nk" in new_input:
        new_input["grid/nk"] = nk

    if not "/grid" in new_input :
        new_input.create_group("/grid")
    new_grid = new_input["/grid"]

    meta_dsets = ["conj_pairs_list", "kpair_idx", "kpair_irre_list", "num_kpair_stored", "trans_pairs_list"]
    if "/grid" in old_meta:
        for dset in meta_dsets:
            if dset in old_meta["/grid"].keys() and not dset in new_grid.keys():
                old_meta["/grid"].copy(dset, new_grid)

    old_meta.close()
    for m in args.new_integral_path:
        if not os.path.exists(m):
            os.mkdir(m)
        meta = h5py.File(m + "/meta.h5", "a")
        meta.attrs[GREEN_VERSION] = "0.2.4"
        meta.close()

    new_input.attrs[GREEN_VERSION] = "0.2.4"
    new_input.close()
    old_input.close()


GREEN_VERSION = "__green_version__"

migration_map = {
    (("", ""), "0.2.4"): pregreen_version_0_2_4
}

parser = argparse.ArgumentParser(description="Migration util.")

parser.add_argument("--old_input_file", type=str, required=True, help="Name of the input file for pre-Green codes")
parser.add_argument("--old_integral_path", type=str, nargs="+", required=True,
                    help="Path to a directory with density-fitted integrals for old code")
parser.add_argument("--new_input_file", type=str, required=True,
                    help="Name of the input file for Green weak-coupling code")
parser.add_argument("--new_integral_path", type=str, nargs="+", required=True,
                    help="Path to a directory with density-fitted integrals for new code")
parser.add_argument("--version", type=str, required=True, help="version to migrate to")

args = parser.parse_args()

if not os.path.exists(args.old_input_file) or not os.path.isfile(args.old_input_file):
    print("pre-Green input file is not found")
    exit(-1)

for m in args.old_integral_path:
    if not os.path.exists(m):
        print("previous version density-fitted integrals are not found")
        exit(-1)

assert (len(args.old_integral_path) == len(args.new_integral_path))

old_input = h5py.File(args.old_input_file, "r")
inp_version = ""
if GREEN_VERSION in old_input.attrs:
    inp_version = old_input.attrs[GREEN_VERSION]
old_input.close()

old_meta_versions = []
for m in args.old_integral_path:
    old_meta = h5py.File(m + "/meta.h5", "r")
    version = ""
    if GREEN_VERSION in old_meta.attrs:
        version = old_meta.attrs[GREEN_VERSION]
    old_meta.close()
    old_meta_versions.append(version)

# Check that all integrals have the same version
assert (old_meta_versions.count(old_meta_versions[0]) == len(old_meta_versions))

v2 = inp_version if inp_version != '' else "undefined"
v2 = old_meta_versions[0] if old_meta_versions[0] != '' else "undefined"

print(f"old input file version is '{inp_version}', old integrals version "
      f"is '{v2}'")

if not ((inp_version, old_meta_versions[0]), args.version) in migration_map:
    print(f"No migration between version {inp_version} and {args.version} has been defined")
    exit(0)

migration_map[((inp_version, old_meta_versions[0]), args.version)](args)

print(f"Migration to version {args.version} successfully finished")