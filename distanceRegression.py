import os,pdb,argparse,sys
import tensorflow as tf
from models import *
from loadData import *
from plotUtils import Plot
import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
from os import system
import h5py,csv,pickle
import json
from datetime import date,datetime
from tensorflow.keras.optimizers import *
timestamp = str(datetime.now().year)+str("{:02d}".format(datetime.now().month))+str("{:02d}".format(datetime.now().day))+str("{:02d}".format(datetime.now().hour))+str("{:02d}".format(datetime.now().minute))
today = str(date.today())

# limit full GPU memory allocation by gradually allocating memory as needed
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

def logConfiguration(dictionary):
    log = json.dumps(dictionary)
    f = open('data/modelConfig_'+timestamp+'.json','w')
    f.write(log)
    f.close()

# define the input arguments
parser = argparse.ArgumentParser(description='Regress distance to the boundary of a unit cube from 3D points and directions.')
parser.add_argument('--trainData', help='Train dataset.', required=False, default=None)
parser.add_argument('--validationData', help='Train dataset.', required=True)
parser.add_argument('--testData', help='Test dataset.', required=False)
parser.add_argument('--plots', help='Produce sanity plots.', default=False, action='store_true')
parser.add_argument('--model', help='Use a previously trained and saved MLP model.', default=None, required=False)
parser.add_argument('--test', help='Model testing environment. Do not save.', default=False, action='store_true')

if __name__ == '__main__':

    # parse the arguments
    args = parser.parse_args()
    if args.model is not None and args.trainData is not None: sys.exit("You can't load a pre-trained '--model' and '--trainData' at the same time!")

    # set normalisations
    # lengthNormalisation = 5543
    lengthNormalisation = 1
    # positionNormalisation = 1600
    positionNormalisation = 1
    
    # define your settings
    settings = {
        'Description'       :    "simple sequential feed-forward network to estimate the distance to the surface of a unit sphere",
        # 'Structure'         :    {'Position' : [512,512,512], 'Direction' : [], 'Concatenated' : [1024,1024]},
        'Structure'         :    [256,256,128],
        'Activation'        :    'relu',
        'OutputActivation'  :    'relu',
        'Loss'              :    'getNoOverestimateLossFunction', #noOverestimateLossFunction #mae #mse
        'negExp'            :    1.2, # only for noOverestimateLossFunction
        'Optimizer'         :    'Adam',
        'LearningRate'      :    0.00100,
        'Batch'             :    1024,
        'Epochs'            :    5
    }
    # this is needed along with eval to convert the str to function call
    dispatcher = {'getNoOverestimateLossFunction':getNoOverestimateLossFunction, 'Adam':Adam}

    # load the validation data, which are needed either you train or not
    valX, valY  = getG4Arrays(args.trainData)
    valDataset = getDatasets(valX, valY)

    # create MLP model
    if args.model is None:

        # get some data to train & validate on
        # load everything in memory
        trainX, trainY  = getG4Arrays(args.trainData)
        trainDataset = getDatasets(trainX, trainY, batch_size=settings['Batch'])

        # get the DNN model
        mlp_model = getSimpleMLP(settings['Structure'],
                                 activation=settings['Activation'],
                                 output_activation=settings['OutputActivation'],
                                 loss=eval(settings['Loss']+'('+str(settings['negExp'])+')', {'__builtins__':None}, dispatcher) if 'getNoOverestimateLossFunction' in settings['Loss'] else settings['Loss'],
                                 optimizer=eval(settings['Optimizer']+'('+str(settings['LearningRate'])+')', {'__builtins__':None}, dispatcher))

        # fit model
        history = mlp_model.fit(trainDataset,
                                epochs=settings['Epochs'],
                                validation_data=valDataset)

        # save model and print learning rate
        if not args.test: 
            mlp_model.save('data/mlp_model_'+timestamp+'.h5')
            logConfiguration(settings)
            # Plot training & validation loss values
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.ylabel('Loss (MSE)')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper right')
            system('mkdir -p plots/'+today+'/'+timestamp)
            plt.savefig('plots/'+today+'/'+timestamp+'/learning_'+timestamp+'.pdf')
            print("Trained model saved! Timestamp:", timestamp)

    # load MLP model
    else:
        timestamp = [item.split('_')[-1].split('.')[0] for item in args.model.split('/') if 'mlp' in item][0]
        # unfortunately the negExp is not saved but I can retrieve it from the logs
        logDic = json.load(open('data/modelConfig_'+timestamp+'.json'))
        retValNegExp = logDic['negExp'] if 'negExp' in logDic.keys() else 1

        mlp_model = tf.keras.models.load_model(args.model, custom_objects={'noOverestimateLossFunction':getNoOverestimateLossFunction(negExp=retValNegExp)})
        # print model summary
        mlp_model.summary()

    # predict on validation data
    print("Calculating validation predictions for %i points..." % len(valX))
    pred_valY = mlp_model.predict(valDataset)

    # that would be the test dataset - WIP
    if args.testData is not None:
        print("Loading all test data in memory...")
        testX, testY = getG4Datasets(args.testData)
        # predict on test data
        print("Calculating test predictions for %i points..." % len(testX))
        pred_testX = mlp_model.predict(testX)

    # plot
    if args.plots: 
        validationPlots = Plot('validation', timestamp, truth=valY, prediction=pred_valY)
        # trainPlots = Plot('training', timestamp, inputFeatures=trainX.numpy(), truth=trainY.numpy())
        validationPlots.plotPerformance()
        # trainPlots.plotInputs()
    
    print("Done!")
