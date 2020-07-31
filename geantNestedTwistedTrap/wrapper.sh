#!/usr/bin/bash
source /opt/setup-fullsim-environment.sh
source /opt/geant4/bin/geant4.sh
baseRunDir=/home/whopkins/distanceRegression/geantNestedTwistedTrap/build
cd $baseRunDir
for (( seed=$1; seed<$2; seed++ ))
do
	$baseRunDir/geantinoMap -m $baseRunDir/$3 -s $seed --nNested $4 -o $5 >& $baseRunDir/logs/log_${seed} &
done
wait
