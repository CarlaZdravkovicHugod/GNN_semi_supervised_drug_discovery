#!/bin/sh

# A100 GPU queue, there is also gpua40 and gpua10, a100, v100
#BSUB -q gpuv100

# job name
#BSUB -J GRAPH_SAGE_16

# 4 cpus, 1 machine, 1 gpu, 12 hours (the max is 24)
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 12:00

# at least 32 GB RAM
#BSUB -R "rusage[mem=32GB]"

# stdout/stderr files for debugging (%J is substituted for job ID)
#BSUB -o logs/my_run_%J.out
#BSUB -e logs/my_run_%J.err

# your training script here, e.g.
# activate environment ...
source .venv/bin/activate
export WAND_PROJECT_NAME="carlahugod-danmarks-tekniske-universitet-dtu/GNN_semi_supervised"
PYTHONPATH="." python src/run.py model=graphsage16