#!/bin/sh
#SBATCH -n 20
#SBATCH -N 3
#SBATCH -t=0-02:00
cd qecsim
python -batch defTND.py