#!/bin/bash
#SBATCH --account scavenger
#SBATCH --partition scavenger
#SBATCH --mem=24gb
#SBATCH --time=12:00:00
#SBATCH --job-name=birdsnapPrepro
cd /fs/vulcan-scratch/krusinga/projects/ganProbability
python preprocessData.py --name birdsnap --folder /vulcan/scratch/krusinga/birdsnap/birdsnap/download/images --outSize 32
