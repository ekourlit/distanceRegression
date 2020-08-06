#!/bin/bash
#SBATCH -p bdwall
#SBATCH --job-name=g4NestedTwistedTrapDataGen_1
#SBATCH -N 1
#SBATCH -A ATLAS-HEP-group
#SBATCH -o /home/whopkins/distanceRegression/geantNestedTwistedTrap/build//logs/g4NestedTwistedTrapDataGen_1.%j.%N.out
#SBATCH -e /home/whopkins/distanceRegression/geantNestedTwistedTrap/build//logs/g4NestedTwistedTrapDataGen_1.%j.%N.error
#SBATCH --time=00:28:00
#SBATCH --ntasks-per-node=36
ncores=36
for (( nodeI=0; nodeI<$SLURM_JOB_NUM_NODES; nodeI++ ))
do
    startSeed=$(( $nodeI * $ncores ))
    endSeed=$(( $startSeed + $ncores ))
    srun -N1 -c $ncores -n1 --exclusive singularity exec /home/whopkins/fullsim-sources-geant4-v10.6.2.img /home/whopkins/distanceRegression/geantNestedTwistedTrap/build//wrapper.sh /home/whopkins/distanceRegression/geantNestedTwistedTrap/build/ /home/whopkins/distanceRegression/geantNestedTwistedTrap/build/ $startSeed $endSeed run.mac 1 g4NestedTwistedTrapDataGen_1 >& /home/whopkins/distanceRegression/geantNestedTwistedTrap/build//logs/log_nNest1_startseed_${startSeed}.txt 
done;

wait
