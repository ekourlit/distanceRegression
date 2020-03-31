import argparse
import numpy as np
import pandas as pd
def produceMacro(inFName='3Ddata.pickle', outFName='run.mac'):
    """! 
    Produce G4 macro from input pickle data.
    """

    dataset = pd.DataFrame(pd.read_pickle(inFName))
    print(dataset.shape)

    templateText ="""
/gun/position {:.2f} {:.2f} {:.2f} mm
/gun/direction {:.2f} {:.2f} {:.2f}
/gun/particle geantino
/run/beamOn 1
"""

    outText = ""
    for i in range(dataset.shape[0]):
        if i % 10000 == 0:
            print("Processed", i)
        outText += templateText.format(dataset['X'][i], dataset['Y'][i], dataset['Z'][i], dataset['Xprime'][i], dataset['Yprime'][i], dataset['Zprime'][i])

    outF = open(outFName, 'w')
    outF.write("/run/initialize\n\n")
    outF.write(outText)
    outF.close()
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Produce a G4 macro that will be able to produce geometric info needed to train an NN.')
    parser.add_argument('inF', help="Perform training, analyze, or both.")
    args = parser.parse_args()
    
    produceMacro(inFName=args.inF)
