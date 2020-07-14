import argparse, matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
font = {'size':16}
matplotlib.rc('font', **font)

def produceMacro(inFName='3Ddata.pickle', outFName='run', maxEventsPerFile=100000):
    """! 
    Produce G4 macro from input pickle data.
    """

    dataset = pd.DataFrame(pd.read_pickle(inFName))
    print(dataset.shape)

    templateText ="""
/gun/position {:.7f} {:.7f} {:.7f} mm
/gun/direction {:.7f} {:.7f} {:.7f}
/gun/particle geantino
/run/beamOn 1
"""

    outText = ""
    fCounter = 0
    for i in range(dataset.shape[0]):
        outText += templateText.format(dataset['X'][i], dataset['Y'][i], dataset['Z'][i], dataset['Xprime'][i], dataset['Yprime'][i], dataset['Zprime'][i])

        if i % maxEventsPerFile == 0 and i > 0:
            print("Processed", i, " events, now writing out a file.")

            outF = open(outFName+'_'+str(fCounter)+'.mac', 'w')
            outF.write("/run/initialize\n\n")
            outF.write(outText)
            outF.close()
            outText = ""
            fCounter+=1

    # Write the remainder out
    outF = open(outFName+'_'+str(fCounter)+'.mac', 'w')
    outF.write("/run/initialize\n\n")
    outF.write(outText)
    outF.close()
    outText = ""
    fCounter+=1
    
    fig, ax = plt.subplots()
    for label in dataset.keys():
        plt.clf()
        plt.hist(dataset[label], histtype='step', bins=100)
        ax.set_xlabel(label)
        plt.savefig(label+'.pdf', bbox_inches='tight')
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Produce a G4 macro that will be able to produce geometric info needed to train an NN.')
    parser.add_argument('inF', help="Input file name.")
    parser.add_argument('--outFName', default='run', help="Output file name.")
    parser.add_argument('--maxEventsPerFile', type=int, default=100000, help="Maximum number of events per output file")
    args = parser.parse_args()
    
    produceMacro(inFName=args.inF, outFName=args.outFName, maxEventsPerFile=args.maxEventsPerFile)
