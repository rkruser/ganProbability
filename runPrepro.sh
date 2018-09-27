#!/bin/bash
#SBATCH --account scavenger
#SBATCH --partition scavenger
#SBATCH --mem=24gb
#SBATCH --time=12:00:00
#SBATCH --job-name=birdsnapPrepro
cd /fs/vulcan-scratch/krusinga/projects/ganProbability
python preprocessData.py --name mnist --folder /vulcan/scratch/krusinga/mnist --outSize 128
