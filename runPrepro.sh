#!/bin/bash
#SBATCH --account scavenger
#SBATCH --partition scavenger
#SBATCH --mem=24gb
#SBATCH --time=12:00:00
#SBATCH --job-name=birdsnapPrepro
cd /fs/vulcan-scratch/krusinga/projects/ganProbability
python preprocessData.py --name cub_200_2011_ --folder /vulcan/scratch/krusinga/CUB_200_2011/images --outSize 32
