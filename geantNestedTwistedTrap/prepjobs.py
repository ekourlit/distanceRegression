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

def writeDataGenSubmit(basedir, submitFName, part, jobName, nNodes, account, expTime, nCores, macFName, nNest, image="/home/whopkins/fullsim-sources-geant4-v10.6.2.img"):
    srunStr = ""
    srunStr += f"\tsrun -N1 -c $ncores -n1 singularity exec {image} {basedir}/wrapper.sh $startSeed $endSeed {macFName} {nNest} {jobName}>& logs/log_nNest{nNest}_startseed_${{startSeed}}.txt &\n"
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
    f = open(submitFName, 'w')
    f.write(submitStr)
    f.close()
    os.chmod(submitFName, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)

def writeTrainSubmit(submitFName, part, jobName, nNodes, account, expTime, nCores, inputPath):
    srunStr = f"srun -N1 -c $ncores -n1 conda run -n tf_mkl_intel python /home/whopkins/distanceRegression/distanceRegression.py --trainValData \\\"{inputPath}\\\" &\n"
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
{srunStr}
wait
"""
    f = open(submitFName, 'w')
    f.write(submitStr)
    f.close()
    os.chmod(submitFName, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)

part = 'hepd'
account = 'condo'
basedir = '/home/whopkins/distanceRegression/geantNestedTwistedTrap/build/'
macFName = 'run.mac'
maxNesting = 2;
nNodes = 1 #Number of nodes per data generation job. maxNesting
nCores = 36
desiredNEvts = 1E7
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
expTime = time.strftime('%H:%M:%S', time.gmtime(expectedTimePerJob*safetyFact))
outFNames = {}
dataGenSubmitFNames = {}
for nNest in range(1,maxNesting+1):
    jobName = f'g4NestedTwistedTrapDataGen_{nNest}'
    submitFName = f'nestedTrapDataGen_{nNest}.slurm'
    dataGenSubmitFNames[nNest] = submitFName
    writeDataGenSubmit(basedir, submitFName, part, jobName, nNodes, account, expTime, nCores, macFName, nNest)
    outFNames[nNest] = jobName

# Now produce the training batch jobs
trainTime = '48:00:00'
trainSubmitFNames = {}
for nNest in range(1,maxNesting+1):
    jobName = f'g4NestedTwistedTrapTrain_{nNest}'
    submitFName = f'nestedTrapTrain_{nNest}.slurm'
    trainSubmitFNames[nNest] = submitFName
    inputPath = f'{basedir}/{outFNames[nNest]}_nt_distance_*.csv'
    
    writeTrainSubmit(submitFName, part, jobName, nNodes, account, trainTime, nCores, inputPath)

submit = True
if submit:
    for nNest in range(1,maxNesting+1):
        submitFName = dataGenSubmitFNames[nNest]
        process = subprocess.Popen(f'sbatch {submitFName}', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        out, err = process.communicate()
        try:
            genJobID = out.decode().lstrip().rstrip().split("job ")[1]
        except Exception as e:
            print('Something went wrong getting job ID of data geneartion job.')
            print(e)
            sys.exit(1)
        submitFName = trainSubmitFNames[nNest]
        process = subprocess.Popen(f'sbatch --dependency=afterok:{genJobID} {submitFName}', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        

