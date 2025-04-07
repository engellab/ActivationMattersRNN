#!/bin/bash
#SBATCH --job-name="RNN_analysis"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem-per-cpu=1G
#SBATCH --time=05:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=pt1290@princeton.edu
#SBATCH --output=../log/%x.%A_%a.out
#SBATCH --error=../log/%x.%A_%a.err

module purge
module load anaconda3/2022.5

# Activate conda environment
conda activate rnns

# to run a script with specific parameters run the following command from src/slurm (as an example):
ray start --head --num-cpus=50
python /home/pt1290/ActivationMattersRNN/activation_matters/analysis/run_entire_pipeline.py --multirun task=CDDM,GoNoGo,MemoryNumber dataSegment=top,bottom control_type=shuffled,random paths=paths_DELLA
ray stop
