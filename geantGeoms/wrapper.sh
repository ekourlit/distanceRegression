#!/usr/bin/bash
source /opt/setup-fullsim-environment.sh
source /opt/geant4/bin/geant4.sh
workDir=$1
baseRunDir=$2
TMPDIR=$workDir/tmp
cd $workDir
for (( seed=$3; seed<$4; seed++ ))
do
	if [ -z "$8" ]
	then
		$baseRunDir/geantinoMap -m $baseRunDir/$5 -s $seed --nNested $6 -o $7 >& $workDir/logs/log_nNest${6}_seed${seed} &
	else
		callgrindOut=$7"_nNested"$6"_seed"$seed".callgrind"
		valgrind --tool=callgrind --callgrind-out-file=$callgrindOut $baseRunDir/geantinoMap -m $baseRunDir/$5 -s $seed --nNested $6 -o $7 >& $workDir/logs/valgrind_log_nNest${6}_seed${seed} &
	fi
done
wait
