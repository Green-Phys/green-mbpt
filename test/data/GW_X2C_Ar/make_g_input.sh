#!/bin/bash

export GWBin=$HOME/Documents/Codes/green-mbpt/build/bin

mpirun -n 3 $GWBin/mbpt.exe --BETA=10 --grid_file ir/1e4.h5 \
  --input_file input_full_symm.h5 --results_file sim_full_symm.h5 \
  --scf_type GW --mixing_weight 0.7 --itermax 1 --verbose 4 \
  --dfintegral_file df_hf_int > out_gw_full_symm

mpirun -n 3 $GWBin/mbpt.exe --BETA=10 --grid_file ir/1e4.h5 \
  --input_file input_trs_only.h5 --results_file sim_trs_only.h5 \
  --scf_type GW --mixing_weight 0.7 --itermax 1 --verbose 4 \
  --dfintegral_file df_hf_int > out_gw_trs

mpirun -n 3 $GWBin/mbpt.exe --BETA=10 --grid_file ir/1e4.h5 \
  --input_file input_no_symm.h5 --results_file sim_no_symm.h5 \
  --scf_type GW --mixing_weight 0.7 --itermax 1 --verbose 4 \
  --dfintegral_file df_hf_int > out_gw_no_symm
