#!/usr/bin/bash
source /opt/setup-fullsim-environment.sh
source /opt/geant4/bin/geant4.sh
baseRunDir=/home/whopkins/distanceRegression/geantNestedTwistedTrap/build
cd $baseRunDir
for (( seed=$1; seed<$2; seed++ ))
do
	$baseRunDir/geantinoMap -m $baseRunDir/geantino.mac -s $seed -n $3 -u 3.7 -r 1.01 >& $baseRunDir/logs/log_${seed} &
done
wait
