#!/bin/bash

#SBATCH --account=beig-delta-gpu
#SBATCH --partition=gpuA100x4
### NODE/CPU/MEM/GPU  ### # can add mem gpu bind params below optionally
#SBATCH --mem-per-cpu=20G
#SBATCH --cpus-per-gpu=1
### TODO set the above mem-per-cpu and cpus-per-gpu according to your use case