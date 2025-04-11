#!/bin/bash
#SBATCH --job-name=in5550
#SBATCH --account=ec403
#SBATCH --time=00:30:00
#SBATCH --partition=accel    # To use the accelerator nodes
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=4

# By default, request four CPU cores NumPy may just know how to take
# advantage of them; for larger computations, maybe use between
# six and ten; at some point, we will look at how to run on GPUs
#
#SBATCH --cpus-per-task=4

# NB: this script should be run with "sbatch train.slurm"!
# See https://www.uio.no/english/services/it/research/platforms/edu-research/help/fox/jobs/submitting.md

source ~/.bashrc

# Sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset

# The important bit: unload all current modules (just in case) and load only the necessary ones
module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/

# Load necessary modules
module load nlpl-nlptools/04-foss-2022b-Python-3.10.8
module load nlpl-pytorch/2.1.2-foss-2022b-cuda-12.0.0-Python-3.10.8
module load nlpl-gensim/4.3.2-foss-2022b-Python-3.10.8
module load nlpl-transformers/4.43.4-foss-2022b-Python-3.10.8 
module load nlpl-accelerate/0.33.0-foss-2022b-Python-3.10.8 
module load nlpl-torchmetrics/1.2.1-foss-2022b-Python-3.10.8
module load nlpl-huggingface-hub/0.23.5-foss-2022b-Python-3.10.8  # Corrected line

# Activate your Python virtual environment (if applicable)
# source /path/to/your/venv/bin/activate

# Run the training script with arguments
python3 dependencyParser.py \
    --model ltg/norbert3-xs \
    --batch_size 64 \
    --lr 3e-4 \
    --epochs 10 \
    --gradient_clipping 5 \
    --device cuda:0 \
