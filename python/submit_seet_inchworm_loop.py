#!/usr/bin/env python3
"""
Script to submit SEET + inchworm solver jobs in sequence.
Alternates between SEET (with itermax=1) and inchworm solver jobs.

NOTE:
- This is a template script. You need to fill in the actual SEET and inchworm solver commands.
- The processing of inchworm output is a placeholder and needs to be implemented based on your requirements.
- This script will only support SIGMA_MIXING mode for now; CDIIS will require further implementation.
"""

import numpy as np
import h5py
import argparse
import subprocess
import time
from pathlib import Path


SEET_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=seet_iter_{iter}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=seet_{iter}_%j.log

# Load modules, set up environment, etc.
# module load ...

# Run SEET with inchworm solver and itermax=1
seet_command_here --input input.h5 --itermax 1 --restart true \
  --mixing_type SIGMA_MIXING --mixing_weight {mixing} \
  --results_file {results_filename} --seet_input {transform_filename} \
  --impurity_solver "INCHWORM" ...
"""

INCHWORM_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=inchworm_iter_{iter}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=inchworm_{iter}_%j.log

# Load modules, set up environment, etc.
# module load ...

# Run inchworm solver
inchworm_solver_command_here
"""


def submit_slurm_job(script_path: Path) -> str:
    """
    Submit a SLURM job and return the job ID.
    """
    result = subprocess.run(
        ["sbatch", str(script_path)],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to submit job: {result.stderr}")
    
    # Extract job ID from output (format: "Submitted batch job XXXXX")
    job_id = result.stdout.strip().split()[-1]
    print(f"Submitted job {job_id}: {script_path.name}")
    return job_id


def check_job_status(job_id: str) -> str:
    """
    Check the status of a SLURM job.
    Returns: "COMPLETED", "RUNNING", "PENDING", or "FAILED"
    """
    result = subprocess.run(["scontrol", "show", "job", job_id], capture_output=True, text=True)
    if result.returncode != 0:
        return "UNKNOWN"
    
    if "JobState=COMPLETED" in result.stdout:
        return "COMPLETED"
    elif "JobState=RUNNING" in result.stdout:
        return "RUNNING"
    elif "JobState=PENDING" in result.stdout:
        return "PENDING"
    elif "JobState=FAILED" in result.stdout or "JobState=TIMEOUT" in result.stdout:
        return "FAILED"
    else:
        return "UNKNOWN"


def wait_for_job(job_id: str, check_interval: int = 30, max_wait: int = 86400) -> bool:
    """
    Wait for a job to complete.
    Returns True if job completed successfully, False otherwise.
    """
    elapsed = 0
    while elapsed < max_wait:
        status = check_job_status(job_id)
        print(f"Job {job_id} status: {status}")
        
        if status == "COMPLETED":
            print(f"Job {job_id} completed successfully")
            return True
        elif status == "FAILED":
            print(f"Job {job_id} failed")
            return False
        elif status == "UNKNOWN":
            print(f"Job {job_id} not found (may have been purged)")
            return False
        
        time.sleep(check_interval)
        elapsed += check_interval
    
    print(f"Job {job_id} did not complete within {max_wait} seconds")
    return False


def process_inchworm_output(iteration: int, workdir: Path, args: argparse.Namespace) -> bool:
    """
    Process inchworm solver output before next SEET iteration.
    
    PLACEHOLDER: Implement your post-processing logic here.
    
    Examples of what you might do:
    - Read inchworm output files
    - Extract self-energy, Green's function, or other observables
    - Transform/convert data formats
    - Update input files for next SEET iteration
    - Check convergence criteria
    - Copy/move files to appropriate locations
    
    Parameters
    ----------
    iteration : int
        Current iteration number (starts from 1)
    workdir : Path
        Working directory containing input/output files
    args : argparse.Namespace
        Command-line arguments containing results_file and transform_file
    
    Returns
    -------
    bool
        True if processing succeeded, False if it failed
    """
    print(f"\n{'='*60}")
    print(f"Processing inchworm output from iteration {iteration}")
    print(f"{'='*60}")

    # Open and obtain transformation
    ftransform = h5py.File(args.transform_file, 'r')
    nimp = ftransform['nimp'][()]
    # AO <-> Orthogonal basis transformations
    X_k = ftransform['X_k'][()]
    # Projection from active to full orbital space
    uu_trans = []
    nao_imp = []
    for i in range(nimp):
        uu_i = ftransform[f"{i}/UU"][()] + 0j
        uu_trans.append(uu_i)
        nao_imp.append(uu_i.shape[0])
    ftransform.close()
    
    # PLACEHOLDER: 
    # 1. Read inchworm output files
    # 2. Extract new local self-energy
    #       NOTE: Separate static and dynamic contributions not needed, but highly appreciated
    # 3. Store them in the variable sigma_new of shape (ntau, nao_imp, nao_imp) where
    #       ntau is the number of imaginary time points on IR grid
    #       nao_imp is the number of impurity orbitals

    ntau = 100  # Placeholder -- remove if needed
    ns_imp = 2  # Placeholder number of spin -- remove if needed
    sigma_tau_inchworm = [np.zeros((ntau, ns_imp, nao_imp[i], nao_imp[i]), dtype=np.complex128) for i in range(nimp)]
    sigma_inf_inchworm = [np.zeros((ns_imp, nao_imp[i], nao_imp[i]), dtype=np.complex128) for i in range(nimp)]
    # NOTE: expecting sigma on imaginary-time grid.

    # Open output file
    fsimseet = h5py.File(args.results_file, 'r+')
    group = fsimseet['iter{}/Selfenergy'.format(iteration)]
    sigma_in = group['data'][()]
    sigma_inf_in = fsimseet['iter{}/Sigma1'.format(iteration)][()]
    ntau, ns, nk, nao_full, _ = sigma_in.shape
    assert ns_imp == ns, "Spin dimension mismatch"

    # Active space to full orthogonal basis and combine SIGMA from all impurity blocks
    sigma_tau_local_orth = np.zeros((ntau, ns, nao_full, nao_full), dtype=np.complex128)
    sigma_inf_local_orth = np.zeros((ns, nao_full, nao_full), dtype=np.complex128)
    for i in range(nimp):
        sigma_tau_local_orth += np.einsum('pi, tspq, qj -> tsij', uu_trans[i].conj(), sigma_tau_inchworm[i], uu_trans[i])
        sigma_inf_local_orth += np.einsum('pi, spq, qj -> sij', uu_trans[i].conj(), sigma_inf_inchworm[i], uu_trans[i])

    # Orthogonal to AO basisi for each k-point
    sigma_loc_ao_ts_ibz = np.zeros((ntau, ns, nk, nao_full, nao_full), dtype=np.complex128)
    sigma_loc_ao_inf_ibz = np.zeros((ns, nk, nao_full, nao_full), dtype=np.complex128)
    for t in range(ntau):
        for s in range(ns):
            sigma_loc_ao_ts_ibz[t, s] = np.einsum('kab, bc, sdc -> kad', X_k, sigma_tau_local_orth[t, s], X_k.conj())
            sigma_loc_ao_inf_ibz[s] = np.einsum('kab, bc, sdc -> kad', X_k, sigma_inf_local_orth[s], X_k.conj())
    
    # Update SIGMA from results file
    for t in range(ntau):
        for s in range(ns):
            sigma_in[t, s] += sigma_loc_ao_ts_ibz[t, s] * args.mixing
            sigma_inf_in[s] += sigma_loc_ao_inf_ibz[s] * args.mixing
    
    # Save the data back to results file
    group['data'][...] = sigma_in
    fsimseet['iter{}/Sigma1'.format(iteration)][...] = sigma_inf_in
    fsimseet.close()
    
    print("[PLACEHOLDER] Inchworm output processing not implemented")
    print("              Add your processing logic to process_inchworm_output()")
    return True  # Return True to continue workflow in placeholder mode


def main():
    parser = argparse.ArgumentParser(description="Submit SEET + inchworm solver jobs in sequence")
    parser.add_argument("--niters", type=int, default=3, help="Number of SEET iterations to run")
    parser.add_argument("--workdir", type=Path, default=Path.cwd(), help="Working directory for job scripts")
    parser.add_argument("--check-interval", type=int, default=30, help="Interval (seconds) to check job status")
    parser.add_argument("--dry-run", action="store_true", help="Print job scripts without submitting")
    parser.add_argument("--transform_file", type=Path, default="transform.h5", help="Path to transformation file")
    parser.add_argument("--results_file", type=Path, default="seet_results.h5", help="Path to SEET results file")
    parser.add_argument("--mixing", type=float, default=0.5, help="Mixing parameter for self-energy update")
    parser.add_argument("--ir_file", type=Path, default="1e5.h5", help="Path to IR grid file")
    parser.add_argument("--BETA", type=float, default=100.0, help="Inverse temperature for IR grid")
    
    args = parser.parse_args()
    workdir = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    
    job_ids = []
    
    for iteration in range(1, args.niters + 1):
        # ===== SEET job =====
        seet_script_path = workdir / f"seet_iter_{iteration}.sh"
        seet_script = SEET_SCRIPT_TEMPLATE.format(
            iter=iteration, transform_filename=args.transform_file,
            results_filename=args.results_file, mixing=args.mixing
        )
        seet_script_path.write_text(seet_script)
        seet_script_path.chmod(0o755)
        
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}: Submitting SEET job")
        print(f"{'='*60}")
        if args.dry_run:
            print(f"[DRY-RUN] Would submit: {seet_script_path}")
            print(seet_script)
        else:
            seet_job_id = submit_slurm_job(seet_script_path)
            job_ids.append(("SEET", iteration, seet_job_id))
            
            # Wait for SEET to complete
            if not wait_for_job(seet_job_id, check_interval=args.check_interval):
                print(f"SEET job failed at iteration {iteration}. Stopping.")
                break
        
        # ===== Inchworm job =====
        inchworm_script_path = workdir / f"inchworm_iter_{iteration}.sh"
        inchworm_script = INCHWORM_SCRIPT_TEMPLATE.format(iter=iteration)
        inchworm_script_path.write_text(inchworm_script)
        inchworm_script_path.chmod(0o755)
        
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}: Submitting inchworm solver job")
        print(f"{'='*60}")
        if args.dry_run:
            print(f"[DRY-RUN] Would submit: {inchworm_script_path}")
            print(inchworm_script)
        else:
            inchworm_job_id = submit_slurm_job(inchworm_script_path)
            job_ids.append(("Inchworm", iteration, inchworm_job_id))
            
            # Wait for inchworm to complete
            if not wait_for_job(inchworm_job_id, check_interval=args.check_interval):
                print(f"Inchworm job failed at iteration {iteration}. Stopping.")
                break
            
            # Process inchworm output before next SEET iteration
            if not process_inchworm_output(iteration, workdir, args):
                print(f"Failed to process inchworm output at iteration {iteration}. Stopping.")
                break
    
    # Summary
    print(f"\n{'='*60}")
    print("Job submission summary:")
    print(f"{'='*60}")
    for job_type, iteration, job_id in job_ids:
        print(f"{job_type:12} Iteration {iteration}: Job ID {job_id}")


if __name__ == "__main__":
    main()
