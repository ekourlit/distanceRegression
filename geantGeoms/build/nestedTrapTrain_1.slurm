#!/bin/bash
#SBATCH -p bdwall
#SBATCH --job-name=g4NestedTwistedTrapTrain_1
#SBATCH -N 1
#SBATCH -A ATLAS-HEP-group
#SBATCH -o /home/whopkins/distanceRegression/geantNestedTwistedTrap/build//logs/g4NestedTwistedTrapTrain_1.%j.%N.out
#SBATCH -e /home/whopkins/distanceRegression/geantNestedTwistedTrap/build//logs/g4NestedTwistedTrapTrain_1.%j.%N.error
#SBATCH --time=48:00:00
#SBATCH --ntasks-per-node=36
ncores=36
srun -N1 -c $ncores -n1 --exclusive conda run -n tf_mkl_intel bash -c 'python /home/whopkins/distanceRegression//distanceRegression.py --trainValData \"/home/whopkins/distanceRegression/geantNestedTwistedTrap/build//g4NestedTwistedTrapDataGen_1_nt_distance_*.csv\" > /home/whopkins/distanceRegression/geantNestedTwistedTrap/build//logs/g4NestedTwistedTrapTrain_1.condaOut 2>&1' &
wait
