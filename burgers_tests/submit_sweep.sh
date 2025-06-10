#!/bin/bash
#SBATCH --account=diffprog
#SBATCH --time=4:00:00
#SBATCH --job-name=burger_sweep
#SBATCH --mail-user=jonathan.maack@nrel.gov
#SBATCH --mail-type=ALL
#SBATCH --output=burger_sweep.%j.log  # %j will be replaced with the job ID

source ${HOME}/.bash_profile
module load julia/1.11

cd /projects/diffprog/jmaack/AutoDiffSVDCompression/burgers_tests
julia burgers_sweep.jl
