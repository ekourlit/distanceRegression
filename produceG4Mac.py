import argparse
import numpy as np

def produceMacro(minX=0, maxX=1, minY=0, maxY=1, minZ=0, maxZ=1, nPoints=10000, outFName='run.mac'):
    """! 
    Produce data in two dimension. 
    The target data will be the distance of a randomly generated point to the max. 
    """

    # If we weren't given a pdf, simply produced evenly spaced data between the min ans max
    xPoints = np.random.uniform(minX, maxX, nPoints)
    yPoints = np.random.uniform(minY, maxY, nPoints)
    zPoints = np.random.uniform(minZ, maxZ, nPoints)
    
    xDirection = np.random.uniform(minX, maxX, nPoints)
    yDirection = np.random.uniform(minY, maxY, nPoints)
    zDirection = np.random.uniform(minZ, maxZ, nPoints)
    lengths =  np.sqrt(np.power(xDirection,2)+np.power(yDirection,2)+np.power(zDirection,2))
    xDirection = xDirection/lengths
    yDirection = yDirection/lengths
    zDirection = zDirection/lengths

    templateText ="""
/gun/position {:.2f} {:.2f} {:.2f} mm
/gun/direction {:.2f} {:.2f} {:.2f}
/gun/particle geantino
/run/beamOn 1
"""

    outText = ""
    for i in range(nPoints):
        outText += templateText.format(xPoints[i], yPoints[i], zPoints[i], xDirection[i], yDirection[i], zDirection[i])

    outF = open(outFName, 'w')
    outF.write("/run/initialize\n\n")
    outF.write(outText)
    outF.close()
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Produce a G4 macro that will be able to produce geometric info needed to train an NN.')
    #parser.add_argument('trainAnalyze', help="Perform training, analyze, or both.")
    args = parser.parse_args()
    
    produceMacro()
