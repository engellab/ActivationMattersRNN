#!/bin/bash
#SBATCH --job-name="RNN_analysis"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=20G
#SBATCH --time=40:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=pt1290@princeton.edu
#SBATCH --output=../log/%x.%A_%a.out
#SBATCH --error=../log/%x.%A_%a.err

# Load modules for CUDA, CUDNN, and conda
module purge
module load cudatoolkit/12.0
module load cudnn/cuda-11.x/8.2.0
module load anaconda3/2022.5

# Activate conda environment
conda activate RNN

# to run a script with specific parameters run the following command from src/slurm (as an example):

python ~/AcitvationMattersRNN/activation_matters/analysis/run_entire_pipeline.py -- multirun taskname=CDDM,GoNoGo,MemoryNumber dataSegment=top,bottom paths=paths_DELLA
