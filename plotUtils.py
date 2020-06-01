from matplotlib import pyplot as plt
from os import system
from datetime import date
today = str(date.today())

class Plot:
    """
    This is a class for creating plot objects. 
    The different kind of plots are created by the class methods.
    """
    def __init__(self, timestamp, truth=None, prediction=None, inputFeatures=None):
        print(".+.+. Constructing a Plot object .+.+.")
        
        self.timestamp = timestamp
        self.truth = truth
        self.prediction = prediction
        self.features = inputFeatures

        self.saveDir = 'plots/'+today+'/'+timestamp
        system('mkdir -p '+self.saveDir)

    def setInputFeatures(self, features):
        self.features = features

    def inputPlots(self, savename='inputs'):
        '''
        Produce a set of plots to inspect the inputs (array of shape (i,6))
        arguments
            savename: the plot file (.pdf) name
        '''
        print("Plot\t::\tPlotting input features")

        # create subplot env with shared y axis
        fig, axs = plt.subplots(3,2)
        fig.tight_layout(pad=2.0)

        # X
        X = self.features[:,0]
        axs[0,0].hist(X, bins=100)
        axs[0,0].set(xlabel='X')

        # Y
        Y = self.features[:,1]
        axs[1,0].hist(Y, bins=100)
        axs[1,0].set(xlabel='Y')

        # Z
        Z = self.features[:,2]
        axs[2,0].hist(Z, bins=100)
        axs[2,0].set(xlabel='Z')

        # Xprime
        Xprime = self.features[:,3]
        axs[0,1].hist(Xprime, bins=100)
        axs[0,1].set(xlabel='X\'')

        # Yprime
        Yprime = self.features[:,4]
        axs[1,1].hist(Yprime, bins=100)
        axs[1,1].set(xlabel='Y\'')

        # Zprime
        Zprime = self.features[:,5]
        axs[2,1].hist(Zprime, bins=100)
        axs[2,1].set(xlabel='Z\'')

        # save
        plt.savefig(self.saveDir+'/'+savename+'.pdf')
        print("Plot\t::\t"+savename+".pdf saved!")


    
