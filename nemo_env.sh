#!/bin/bash

#SBATCH -A ICCS-SL2-CPU               # Project account for SLURM
#SBATCH -p icelake                     # Partition name in cluster (e.g., icelake)
#SBATCH -N 1                              # Number of nodes
#SBATCH -t 24:00:00                    # Time limit (HH:MM:SS)
#SBATCH -n 36                    # Number of tasks (processes) to be run in parallel

# ========================
#  Define Environment Variables
# ========================
export WORK_DIR="/home/sg2147/nemo_4.2.1/tests/DINO/EXP00"   # Both nemo executable and output directories are at the same location
export NEMO_EXEC="nemo"

# ======================
# LOAD MODULES
# ======================
module purge
module load rhel8/default-ccl
module load netcdf-fortran/4.6.1/intel/intel-oneapi-mpi/kqukipdf
module load boost/1.85.0/gcc/zouxm6hy

echo "******Modules loaded successfully!******"


# =============================
# SET DYNAMIC LIBRARY PATHS
# =============================
export BASE_DIR="/rds/project/rds-5mCMIDBOkPU/ma595/nemo/NEMO/trunk"

export CPATH="$BASE_DIR/inc:$CPATH"
export LD_LIBRARY_PATH="$BASE_DIR/lib:$LD_LIBRARY_PATH"

echo "******Environment paths set successfully!******"


# =============================
# MOVE TO WORKING DIRECTORY
# =============================
cd "$WORK_DIR" || { echo "Directory not found!"; exit 1; }

echo "******Moved to working directory!******"

# =======================================
# DETERMINE RUN TYPE (New Run or Re-run)
# =======================================
RUN_TYPE=${1:-new}  # Takes the first argument passed at runtime, and Default to "new" if no argument is given
RUN_ID=$(date +"%Y%m%d_%H%M%S")  

if [[ "$RUN_TYPE" == "acc" ]]; then      ### For Accelerated run
    OUTPUT_DIR="$WORK_DIR/acc_$RUN_ID"
else
    OUTPUT_DIR="$WORK_DIR/sim_$RUN_ID"   ### For Simulated run
fi

mkdir -p "$OUTPUT_DIR"

echo "******Created new output folder!******"

# =======================================
# DETERMINE RUN TYPE (New Run or Re-run)
# =======================================
mpirun -np 36 "$NEMO_EXEC"   ### mpi is a communication protocol to run code in parallel.
