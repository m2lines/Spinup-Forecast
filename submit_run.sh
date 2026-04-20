#!/bin/bash
export PROJECT_ACCOUNT="ICCS-SL2-CPU"
export PARTITION="icelake"
export NUM_NODES=1
export TIME_LIMIT="24:00:00"
export NUM_TASKS=36
export WORK_DIR="/home/sg2147/nemo_4.2.1/tests/DINO/EXP00"
export NEMO_EXEC="nemo"

sbatch run_nemo.sh
