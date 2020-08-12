import subprocess, time, sys, math, os, stat, argparse

def runGeantinoMap(macFName, nNest, outFName, workdir, basedir, valgrind=False):
    start_time = time.time()
    valgrindStr = ''
    if valgrind:
        valgrindStr = 'valgrind'
    execStr = f'singularity exec /home/whopkins/fullsim-sources-geant4-v10.6.2.img {basedir}/wrapper.sh {workdir} {basedir} 1 2 {macFName} {nNest} {outFName} {valgrindStr}'
    process = subprocess.Popen(execStr, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = process.communicate()
    totalTime = time.time() - start_time
    return (out, err, totalTime)


def writeMacro(macFName, nEvents):
    testMac = open(macFName, 'w')
    testMac.write(f'/run/initialize\n/run/beamOn {nEvents}\n')
    testMac.close();

def writeDataGenSubmit(submitFName, part, jobName, nNodes, account, expTime, nCores, macFName, nNest, workdir, basedir, image="/home/whopkins/fullsim-sources-geant4-v10.6.2.img"):
    srunStr = ""
    srunStr += f"srun -N1 -c $ncores -n1 --exclusive singularity exec {image} {basedir}/wrapper.sh {workdir} {basedir} $startSeed $endSeed {macFName} {nNest} {jobName} >& {workdir}/logs/log_nNest{nNest}_startseed_${{startSeed}}.txt "
    submitStr = f"""#!/bin/bash
#SBATCH -p {part}
#SBATCH --job-name={jobName}
#SBATCH -N {nNodes}
#SBATCH -A {account}
#SBATCH -o {workdir}/logs/{jobName}.%j.%N.out
#SBATCH -e {workdir}/logs/{jobName}.%j.%N.error
#SBATCH --time={expTime}
#SBATCH --ntasks-per-node={nCores}
ncores={nCores}
for (( nodeI=0; nodeI<$SLURM_JOB_NUM_NODES; nodeI++ ))
do
    startSeed=$(( $nodeI * $ncores ))
    endSeed=$(( $startSeed + $ncores ))
    {srunStr}
done;

wait
"""
    f = open(submitFName, 'w')
    f.write(submitStr)
    f.close()
    os.chmod(submitFName, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)

def writeTrainSubmit(submitFName, part, jobName, nNodes, account, expTime, nCores, inputPath, workdir, basedir):
    srunStr = f"srun -N1 -c $ncores -n1 --exclusive conda run -n tf_mkl_intel bash -c 'python {basedir}/distanceRegression.py --trainValData \\\"{inputPath}\\\" > {workdir}/logs/{jobName}.condaOut 2>&1' &"
    submitStr = f"""#!/bin/bash
#SBATCH -p {part}
#SBATCH --job-name={jobName}
#SBATCH -N {nNodes}
#SBATCH -A {account}
#SBATCH -o {workdir}/logs/{jobName}.%j.%N.out
#SBATCH -e {workdir}/logs/{jobName}.%j.%N.error
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

parser = argparse.ArgumentParser(description='Produce data, train regression, evaluate inference time, all in one go.')
parser.add_argument('-p', '--partition', default='hepd', choices=['hepd', 'knlall', 'bdwall'])
parser.add_argument('-a', '--account', default='condo', choices=['condo', 'ATLAS-HEP-group'])
parser.add_argument('-d', '--basedir', default='/home/whopkins/distanceRegression/geantGeoms/build/', help='Directory that contains the executable.')
parser.add_argument('-w', '--workdir', default='/home/whopkins/distanceRegression/geantGeoms/build/', help='Working directory. This is where logs and outputs will be placed.')
parser.add_argument('-n', '--nNesting', type=int, default=2, help='Degree of nesting of twisted trapezoids.')
parser.add_argument('-c', '--cores', type=int, default=36)
parser.add_argument('-e', '--events', type=int, default=1E7)
parser.add_argument('-s', '--submit', action='store_true', help='Submit jobs.')
parser.add_argument('-t', '--trainOnly', action='store_true', help='Only submit training jobs because we already have the data.')
parser.add_argument('-f', '--fudgeFactor', type=float, default=1.5, help='Fudge factor for the running time of the data generation.')
parser.add_argument('--trainBaseDir', default='/home/whopkins/distanceRegression/', help='Directory where the training scripts live')
parser.add_argument('--dataGenTimeStr', default='24:00:00', help='Time string for job submission.')
parser.add_argument('--trainTimeStr', default='48:00:00', help='Time string for job submission.')
parser.add_argument('-v', '--valgrind', action='store_true', help='Run a small valgrind job.')


args = parser.parse_args()

macFName = 'run.mac'
nNodes = 1 #Number of nodes per data generation job. 

# Check if the required directories exist.
dirs = ['logs','data', 'plots']
for directory in dirs:
    if not os.path.exists(directory):
        os.mkdir(directory)

# find the number of events needed.
if not args.trainOnly:
    lines = 0
    totalTime = 0
    nEvents = 10000
    writeMacro(macFName, nEvents)

    times = []
    outs = []
    errs = []
    outFName = 'default'
    for nNest in range(1,args.nNesting+1):
        (out, err, totalTime) = runGeantinoMap(macFName, nNest, outFName, args.workdir, args.basedir)
        outs.append(out)
        errs.append(err)
        times.append(totalTime)
    ntFName = f'{outFName}_nt_distance_1.csv'
    lines = len(open(ntFName).readlines())
    efficiency = 1.*lines/nEvents
    nEventsPerJob = int(math.ceil((args.events/(nNodes*args.cores))/efficiency))
    expectedTimePerJob = int(math.ceil((nEventsPerJob/nEvents)*max(times)))
    h, m, s = args.dataGenTimeStr.split(':')
    requestedDataGenTime = int(h) * 3600 + int(m) * 60 + int(s)

    if requestedDataGenTime < expectedTimePerJob:
        print(f'Be careful: you are requesting {args.dataGenTimeStr} for you job but extrapolating from a test job predicts that you need {expectedTimePerJob}.')
        
    # Write the data production submission file. 
    writeMacro(macFName, nEventsPerJob)
    outFNames = {}
    dataGenSubmitFNames = {}
    for nNest in range(1,args.nNesting+1):
        jobName = f'g4NestedTwistedTrapDataGen_{nNest}'
        submitFName = f'nestedTrapDataGen_{nNest}.slurm'
        dataGenSubmitFNames[nNest] = submitFName
        writeDataGenSubmit(submitFName, args.partition, jobName, nNodes, args.account, args.dataGenTimeStr, args.cores, macFName, nNest, args.basedir, args.workdir)
        outFNames[nNest] = jobName
else:
    outFNames = {}
    for nNest in range(1,args.nNesting+1):
        outFNames[nNest] = f'g4NestedTwistedTrapDataGen_{nNest}'

# Now produce the training batch jobs
trainSubmitFNames = {}
for nNest in range(1,args.nNesting+1):
    jobName = f'g4NestedTwistedTrapTrain_{nNest}'
    submitFName = f'nestedTrapTrain_{nNest}.slurm'
    trainSubmitFNames[nNest] = submitFName
    inputPath = f'{args.basedir}/{outFNames[nNest]}_nt_distance_*.csv'
    
    writeTrainSubmit(submitFName, args.partition, jobName, nNodes, args.account, args.trainTimeStr, args.cores, inputPath, args.workdir, args.trainBaseDir)

if args.submit:
    print('Submitting jobs...')
    for nNest in range(1,args.nNesting+1):
        trainSubmitFName = trainSubmitFNames[nNest]
        if not args.trainOnly:
            submitFName = dataGenSubmitFNames[nNest]            
            process = subprocess.Popen(f'sbatch {submitFName}', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            out, err = process.communicate()
            try:
                genJobID = out.decode().lstrip().rstrip().split("job ")[1]
            except Exception as e:
                print('Something went wrong getting job ID of data geneartion job.')
                print(e)
                sys.exit(1)
            process = subprocess.Popen(f'sbatch --dependency=afterok:{genJobID} {trainSubmitFName}', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        else:
            process = subprocess.Popen(f'sbatch {trainSubmitFName}', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

if args.valgrind:
    nEvents = 1
    writeMacro(macFName, nEvents)
    (out, err, totalTime) = runGeantinoMap(macFName, nNest, outFName, args.workdir, args.basedir, valgrind=True)
    print(out)
    print(err)
   
