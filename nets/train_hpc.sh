#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=8:mem=48gb:ngpus=1:gpu_type=RTX6000
module load cuda

#!/usr/bin/env bash

