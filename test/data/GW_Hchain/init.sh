#!/bin/bash
#
# Uniform 1D hydrogen chain: one H per cell, equal spacing a = 1.4 bohr
# (0.740848 Ang), transverse cell 10 bohr. nk = 4 1 1 gives q-points
# {0, 1/4, 1/2, 3/4}; 1/4 and 3/4 = -1/4 are non-TRIM.
#
# This is the system on which the periodic-GW polarization bug (the old
# half-tau mirror P(q, beta-t) = P(q, t)) was first noticed by a collaborator.
#
set -e
export ScriptDir=$HOME/Documents/Codes/green-mbpt/python

# Matches the collaborator's PR #77 example: STO-6G, all-electron, nk=4,
# FULL q mesh (no space-group and NO time-reversal reduction). The missing
# time-reversal reduction is what exposes the P(q,beta-t) vs P(-q,beta-t) bug.
python $ScriptDir/init_data_df.py \
  --a a.dat --atom atom.dat \
  --basis sto-6g --xc HF \
  --nk 4 1 1 --restricted true \
  --df_int 1 --keep_cderi false \
  --space_symm false --tr_symm false \
  --output_path input.h5 --int_path df_int --hf_int_path df_hf_int

rm -rf df_hf_int

# Remove pyscf/mean-field scratch files; the test only needs input.h5 + df_int.
rm -f dm.h5 tmp.chk cderi* tmp[0-9]* tmp_* 2>/dev/null || true

# Build the input Green's function G_tau (beta=25, mu=0) into data.h5, then
# produce the golden result/Sigma_tau by running the corrected GW solver:
#   ./solvers_test "[gen_hchain]"
python gen_gtau.py
