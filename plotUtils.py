import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from os import system
from datetime import date
today = str(date.today())
import pdb

class Plot:
    """
    This is a class for creating plot objects. 
    The different kind of plots are created by the class methods.
    """

    def __init__(self, name, timestamp, truth=None, prediction=None, inputFeatures=None, distanceNormalization=1):
        print(".+.+. Constructing a Plot object .+.+.")
        
        self.name = name
        self.timestamp = timestamp
        self.truth = truth
        self.prediction = prediction
        self.features = inputFeatures
        self.dNorm = distanceNormalization

        self.saveDir = 'plots/'+today+'/'+self.timestamp
        system('mkdir -p '+self.saveDir)
    
    # valTruth = None
    # valInput = None
    # valPrediction = None
    # def addValidationData(self, truth=None, prediction=None, inputFeatures=None):
    #     self.valTruth = truth
    #     self.valPrediction = prediction
    #     self.valInput = inputFeatures

    def setInputFeatures(self, inputFeatures):
        self.features = inputFeatures

    def plotInputs(self):
        '''
        Produce a set of plots to inspect the inputs (array of shape (i,6))
        arguments
            savename: the plot file (.pdf) name
        '''
        print("Plot\t::\tPlotting input features")

        assert self.features is not None, "Input features data have not been loaded to Plot object"

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
        savename = self.saveDir+'/'+self.name+'_inputs.pdf'
        plt.savefig(savename)
        print("Plot\t::\t"+savename+" saved!")

        if self.truth is None:
            print("Plot\t::\t Truth L is not defined. Skip plotting.")
        else:
            # plot L too
            plt.clf()
            plt.hist(self.truth, bins=100)
            axis = plt.gca()
            axis.set_xlabel(xlabel='L')
            # save
            savename = self.saveDir+'/'+self.name+'_length.pdf'
            plt.savefig(savename)
            print("Plot\t::\t"+savename+" saved!")

    def plotPerformance(self):
        '''
        Produce a set of performance plots
        arguments
            savename: the plot file (.pdf) name
        '''
        print("Plot\t::\tPlotting performance gauges")

        # safety
        assert (self.truth is not None) and (self.prediction is not None), "Either truth or prediction data have not been loaded to Plot object"

        # ###
        # basic
        # ###
        # create subplot env with shared y axis
        fig, axs = plt.subplots(1,3)
        fig.tight_layout(pad=1.3)

        # truth length
        truth_length = self.truth*self.dNorm
        axs[0].hist(truth_length, bins=100)
        axs[0].set(xlabel='Truth L', ylabel='Points')

        # predicted length
        pred_length = self.prediction*self.dNorm
        axs[1].hist(pred_length, bins=100)
        axs[1].set(xlabel='Predicted L')

        # error
        # error = np.divide(truth_length - pred_length, truth_length, out=np.zeros_like(truth_length - pred_length), where=truth_length!=0)
        error = truth_length - pred_length
        abs_error = abs(error)
        axs[2].hist(error, bins=100, log=True)
        axis = plt.gca()
        axis.minorticks_on()
        # axs[2].set(xlabel='Truth L - Predicted L / Truth L')
        axs[2].set(xlabel='Truth L - Predicted L')

        # save
        plt.tight_layout()
        savename = self.saveDir+'/'+self.name+'_basic_'+self.timestamp+'.pdf'
        plt.savefig(savename)
        print("Plot\t::\t"+savename+" saved!")

        # ###
        # 2D truth vs predicted marginal
        # ###
        # plt.clf()
        # import seaborn as sns
        # sns.set(style="ticks")
        # plot = sns.jointplot(truth_length, pred_length, kind="hex", color="#2E5D9F", joint_kws=dict(gridsize=100))
        # plot.set_axis_labels('Truth L', 'Predicted L')
        # # save
        # plt.tight_layout()
        # plt.savefig(self.saveDir+'/'+savename+'_cor_'+self.timestamp+'.pdf')
        # print(savename+'_cor_'+self.timestamp+".pdf Saved!")

        # ###
        # scatter
        # ###
        plt.clf()
        plt.scatter(truth_length, pred_length, s=0.1)
        axis = plt.gca()
        axis.set_xlabel('Truth L')
        axis.set_ylabel('Predicted L')
        # save
        plt.tight_layout()
        savename = self.saveDir+'/'+self.name+'_scatt_'+self.timestamp+'.png'
        plt.savefig(savename)
        print("Plot\t::\t"+savename+" saved!")

        # ###
        # hist2d Truth vs Predicted
        # ###
        plt.clf()
        plt.hist2d(truth_length.reshape(len(truth_length),), pred_length.reshape(len(pred_length),), bins=(200,200), norm=mpl.colors.LogNorm())
        plt.grid()
        axis = plt.gca()
        plt.plot([0,axis.get_xlim()[1]], [0,axis.get_xlim()[1]], c='r')
        axis.set_xlabel('Truth L')
        axis.set_ylabel('Predicted L')
        # save
        plt.tight_layout()
        savename = self.saveDir+'/'+self.name+'_truthVSpred_'+self.timestamp+'.pdf'
        plt.savefig(savename)
        print("Plot\t::\t"+savename+" saved!")

        # ###
        # hist2d Truth vs Error
        # ###
        plt.clf()
        h = plt.hist2d(truth_length.reshape(len(truth_length),), error.reshape(len(error),), bins=(50,50),  norm=mpl.colors.LogNorm())
        plt.colorbar(h[3])
        axis = plt.gca()
        plt.plot([axis.get_xlim()[0],axis.get_xlim()[1]], [0,0], c='r')
        axis.set_xlabel('Truth L')
        axis.set_ylabel('Truth L - Predicted L')
        # save
        plt.tight_layout()
        savename = self.saveDir+'/'+self.name+'_truthVSerror_'+self.timestamp+'.pdf'
        plt.savefig(savename)
        print("Plot\t::\t"+savename+" saved!")

        # ###
        # performance curve
        # ###
        plt.clf()
        val_truth_length = truth_length
        val_pred_length = pred_length
        val_error = np.divide(val_truth_length - val_pred_length, val_truth_length, out=np.zeros_like(val_truth_length - val_pred_length), where=val_truth_length!=0)
        val_abs_error = abs(val_error)

        # what portion of the predictions has at least x error?
        intervals = np.logspace(-4, 0, num=100)
        val_counts = [np.count_nonzero(val_abs_error>=limit) for limit in intervals]
        
        plt.plot(intervals, np.array(val_counts)/val_abs_error.size, color='#ff7f0e', linewidth=3)
        # plt.legend(['Validation','Test'], loc='upper right')
        plt.grid()
        axis = plt.gca()
        axis.set_xscale('log')
        axis.set_xlabel('Relative Error')
        axis.set_ylabel('Dataset Portion')

        # save
        savename = self.saveDir+'/'+self.name+'_errorCurve_'+self.timestamp+'.pdf'
        plt.savefig(savename)
        print("Plot\t::\t"+savename+" saved!")

    
    # def combPerfPlots(Vprediction, Vtruth, Tprediction=None, Ttruth=None, savename='pred_error'):
    #     '''
    #     Produce a set of basic sanity/validation plots
    #     arguments
    #         Vprediction, Tprediction: the validation and test predicted np.array from MLP
    #         Vtruth, Ttruth: the validation and test truth np.array
    #     '''
    #     print("Hi from Plotter!")
    #     plt.clf()

    #     # create dir
    #     saveDir = 'plots/'+today+'/'+timestamp
    #     system('mkdir -p '+saveDir)

    #     if (Tprediction is not None) and (Ttruth is not None): combined = True
    #     else: combined = False

    #     # validation
    #     val_truth_length = Vtruth*lengthNormalisation
    #     val_pred_length = Vprediction*lengthNormalisation
    #     val_error = np.divide(val_truth_length - val_pred_length, val_truth_length, out=np.zeros_like(val_truth_length - val_pred_length), where=val_truth_length!=0)
    #     val_abs_error = abs(val_error)
        
    #     if combined:
    #         test_truth_length = Ttruth*lengthNormalisation
    #         test_pred_length = Tprediction*lengthNormalisation
    #         test_error = np.divide(test_truth_length - test_pred_length, test_truth_length, out=np.zeros_like(test_truth_length - test_pred_length), where=test_truth_length!=0)
    #         test_abs_error = abs(test_error)

    #     # what portion of the predictions has at least x error?
    #     intervals = np.logspace(-4, 0, num=100)
    #     val_counts = []
    #     test_counts = []
    #     for limit in intervals: 
    #         val_counts.append(np.count_nonzero(val_abs_error>=limit))
    #         if combined: test_counts.append(np.count_nonzero(test_abs_error>=limit))
        
    #     plt.plot(intervals, np.array(val_counts)/val_abs_error.size, color='#ff7f0e', linewidth=3)
    #     if combined: plt.plot(intervals, np.array(test_counts)/test_abs_error.size, color='#1f77b4', linewidth=3)
    #     plt.legend(['Validation','Test'], loc='upper right')

    #     plt.grid()
    #     axis = plt.gca()
    #     axis.set_xscale('log')
    #     axis.set_xlabel('Relative Error')
    #     axis.set_ylabel('Dataset Portion')

    #     # save
    #     plt.savefig(saveDir+'/'+savename+'_'+timestamp+'.pdf')
    #     print(savename+'_'+timestamp+".pdf Saved!")