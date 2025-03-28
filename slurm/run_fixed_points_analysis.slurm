#!/bin/bash
#SBATCH --job-name="FPs"
#SBATCH --nodes=1
#SBATCH --ntasks=25
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=72:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=pt1290@princeton.edu
#SBATCH --output=../log/%x.%A_%a.out
#SBATCH --error=../log/%x.%A_%a.err

# Load modules (no GPU-related modules needed)
module purge
module load anaconda3/2022.5

# Activate conda environment
conda activate rnns

# Set the number of workers (equal to the number of tasks)
export NUM_WORKERS=25

# Run the Python script
python /home/pt1290/ActivationMattersRNN/activation_matters/analysis/computing_fp.py --multirun task=CDDM,GoNoGo,MemoryNumber control_type=random,shuffled dataSegment=top,bottom paths=paths_DELLA
python /home/pt1290/ActivationMattersRNN/activation_matters/analysis/fixed_point_analysis.py --multirun task=CDDM,GoNoGo,MemoryNumber control_type=random,shuffled dataSegment=top,bottom paths=paths_DELLA
python /home/pt1290/ActivationMattersRNN/activation_matters/plots/plotting_MDS_embedding_fixed_point_configurations.py --multirun task=CDDM,GoNoGo,MemoryNumber control_type=random,shuffled dataSegment=top,bottom paths=paths_DELLA