#!/usr/bin/bash
source /opt/setup-fullsim-environment.sh
source /opt/geant4/bin/geant4.sh
workDir=$1
baseRunDir=$2
cd $workDir
for (( seed=$3; seed<$4; seed++ ))
do
	if [ -z "$8" ]
	then
		$baseRunDir/geantinoMap -m $baseRunDir/$5 -s $seed --nNested $6 -o $7 >& $workDir/logs/log_${seed} &
	else
		valgrind --tool=callgrind $baseRunDir/geantinoMap -m $baseRunDir/$5 -s $seed --nNested $6 -o $7 >& $workDir/logs/log_${seed} &
	fi
done
wait
