#!/usr/bin/bash
source /opt/setup-fullsim-environment.sh
source /opt/geant4/bin/geant4.sh
baseRunDir=/home/whopkins/geant4/examples/basic/B4/build
cd $baseRunDir
for (( seed=$1; seed<$2; seed++ ))
do
	/home/whopkins/geant4/examples/basic/B4/build/geantinoMap -m /home/whopkins/geant4/examples/basic/B4/build/geantino.mac -s $seed -n $3 -r 1.01 >& $baseRunDir/logs/log_${seed} &
done
wait