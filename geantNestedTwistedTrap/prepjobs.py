import subprocess, time, sys, math, os, stat


def runGeantinoMap(macFName, nNest, outFName):
    start_time = time.time()
    process = subprocess.Popen(f'singularity exec /home/whopkins/fullsim-sources-geant4-v10.6.2.img /home/whopkins/distanceRegression/geantNestedTwistedTrap/build/wrapper.sh 1 2 {macFName} {nNest} {outFName}', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = process.communicate()
    totalTime = time.time() - start_time
    return (out, err, totalTime)

def writeMacro(macFName, nEvents):
    testMac = open(macFName, 'w')
    testMac.write(f'/run/initialize\n/run/beamOn {nEvents}\n')
    testMac.close();

def writeSubmit(part, jobName, nNodes, account, expTime, nCores, macFName, nNests):
    srunStr = ""
    for nNest in range(1,maxNesting+1):
        outFName = f'{jobName}_{nNest}'
        srunStr += f"\tsrun -N1 -c $ncores -n1 singularity exec /home/whopkins/fullsim-sources-geant4-v10.6.2.img /home/whopkins/distanceRegression/geantNestedTwistedTrap/build/wrapper.sh $startSeed $endSeed {macFName} {nNest} {outFName}>& logs/log_nNest{nNest}_startseed_${{startSeed}}.txt &\n"
    submitStr = f"""#!/bin/bash
#SBATCH -p {part}
#SBATCH --job-name={jobName}
#SBATCH -N {nNodes}
#SBATCH -A {account}
#SBATCH -o logs/{jobName}.%j.%N.out
#SBATCH -e logs/{jobName}.%j.%N.error
#SBATCH --time={expTime}
#SBATCH --ntasks-per-node={nCores}
ncores={nCores}
for (( nodeI=0; nodeI<$SLURM_JOB_NUM_NODES; nodeI++ ))
do
    startSeed=$(( $nodeI * $ncores ))
    endSeed=$(( $startSeed + $ncores ))
    echo $startSeed $endSeed
{srunStr}
	
done;

wait
"""
    submitFName = 'job.submit'
    f = open(submitFName, 'w')
    f.write(submitStr)
    os.chmod(submitFName, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)

part = 'hepd'
account = 'condo'

macFName = 'run.mac'
maxNesting = 2;
nNodes = maxNesting
nCores = 36
desiredNEvts = 1E6
minEventsPerJob = 2000
# if nEventsPerJob < minEventsPerJob:
#     print("You should probably have more than "+str(minEventsPerJob)+" in each file. Exiting")
#     sys.exit()
lines = 0
totalTime = 0
increaseFactor = 2
# find the number of events needed 
nEvents = 10000
writeMacro(macFName, nEvents)

times = []
outs = []
errs = []
outFName = 'test'
for nNest in range(1,maxNesting+1):
    (out, err, totalTime) = runGeantinoMap(macFName, nNest, outFName)
    outs.append(out)
    errs.append(err)
    times.append(totalTime)
ntFName = f'{outFName}_nt_distance_1.csv'
lines = len(open(ntFName).readlines())
efficiency = 1.*lines/nEvents
nEventsPerJob = int(math.ceil((desiredNEvts/(nNodes*nCores))/efficiency))

expectedTimePerJob = int(math.ceil((nEventsPerJob/nEvents)*max(times)))
print(lines, nEventsPerJob, expectedTimePerJob, max(times))

writeMacro(macFName, nEventsPerJob)
safetyFact = 1.5
jobName = 'g4NestedTwistedTrap'
expTime = time.strftime('%H:%M:%S', time.gmtime(expectedTimePerJob*safetyFact))
writeSubmit(part, jobName, nNodes, account, expTime, nCores, macFName, maxNesting)

