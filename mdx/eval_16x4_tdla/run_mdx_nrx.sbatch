#!/bin/bash
#SBATCH --job-name=mdxNRX
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --environment=tf
#SBATCH --account=***
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mahdi.abdollahpour@unibo.it

# Mahdi Abdollahpour (mahdi.abdollahpour@unibo.it)
# 2025

cd $SCRATCH/dev/

set -x

cd $SCRATCH/dev/mdx/scripts/

../eval_16x4_tdla_prb273/run_mdx_nrx_mcs0.sh &

../eval_16x4_tdla_prb273/run_mdx_nrx_mcs1.sh &

../eval_16x4_tdla_prb273/run_mdx_nrx_mcs2.sh &

../eval_16x4_tdla_prb273/run_perf_gpu3_.sh


wait
echo "[sbatch] LMMSE: All GPUs Finished!"