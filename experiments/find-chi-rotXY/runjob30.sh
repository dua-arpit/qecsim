#!/bin/bash
#SBATCH --job-name=rotXY
#SBATCH --output=job_%j.log     # Standard output and error log
#SBATCH --nodes=1                    # Run all processes on a single node
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=10:00:00

module load gcc/devtoolset/9
module load cmake/3.18.0
module load anaconda/anaconda3.7

L=17
ns=3000
python sim-rotXY.py 17 $1 30 0.19 3000 1 $2

