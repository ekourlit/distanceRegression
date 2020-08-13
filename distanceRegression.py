import os,pdb,argparse,sys
import tensorflow as tf
tf.random.set_seed(123)
#tf.config.threading.set_inter_op_parallelism_threads(36)
from models import *
from loadData import *
from plotUtils import Plot,plotTrainingMetrics
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from os import system
import h5py,csv,pickle
import json
from datetime import date,datetime
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import EarlyStopping,LearningRateScheduler
import config

from sklearn.model_selection import train_test_split

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

def congigureAndGetModel(**config):
    if args.distribute: 
            with strategy.scope():
                model = getSimpleMLP_DH(num_layers=config['layers'],
                                        nodes=config['nodes'],
                                        activation=config['activation'],
                                        output_activation=config['output_activation'],
                                        loss=config['loss'],
                                        optimizer=config['optimizer'])
    else:
        model = getSimpleMLP_DH(num_layers=config['layers'],
                                nodes=config['nodes'],
                                activation=config['activation'],
                                output_activation=config['output_activation'],
                                loss=config['loss'],
                                optimizer=config['optimizer'])

    return model

# define the input arguments
parser = argparse.ArgumentParser(description='Regress distance to the boundary of a unit cube from 3D points and directions.')
parser.add_argument('--trainValData', help='Train dataset.', required=False, default=None)
parser.add_argument('--valFrac', help='Fraction of data to be used for validation', type=float, required=False, default=0.2)
parser.add_argument('--trainData', help='Train dataset.', required=False, default=None)
parser.add_argument('--validationData', help='Train dataset.', required=False)
parser.add_argument('--testData', help='Test dataset.', required=False)
parser.add_argument('--plots', help='Produce sanity plots.', default=False, action='store_true')
parser.add_argument('--model', help='Use a previously trained and saved MLP model.', default=None, required=False)
parser.add_argument('--test', help='Model testing environment. Do not save.', default=False, action='store_true')
parser.add_argument('--distribute', help='Enable distributed training on available GPUs.', default=False, action='store_true')

if __name__ == '__main__':

    # parse the arguments
    args = parser.parse_args()
    if args.model is not None and args.trainData is not None: sys.exit("You can't load a pre-trained '--model' and '--trainData' at the same time!")

    # create distributed strategy
    if args.distribute: 
        strategy = tf.distribute.MirroredStrategy()
        availableGPUs = int(len(gpus))
    else:
        availableGPUs = 1

    if args.trainValData:
      trainValDataFiles = glob.glob(args.trainValData.replace("\"", ""))
      nValFiles = int(args.valFrac*len(trainValDataFiles))
      print(f'Using the first {nValFiles} files for validation out of {len(trainValDataFiles)} total files.')
      valFiles = trainValDataFiles[:nValFiles]
      trainFiles = trainValDataFiles[nValFiles:]
      trainDataset = getG4Datasets_dataAPI(trainFiles, batch_size=config.settings['Batch']*availableGPUs)
    else:
      # load the validation data, which are needed either you train or not
      trainDataset = getG4Datasets_dataAPI(args.trainData, batch_size=config.settings['Batch']*availableGPUs)
      valFiles = glob.glob(args.validationData)
      
    valX, valY = getG4Arrays(valFiles)
    valDataset = getDatasets(valX, valY)
    # create model & train it
    if args.model is None:
        # get some data to train & validate on
        # load everything in memory in advance
        # trainX, trainY  = getG4Arrays(args.trainData)
        # trainDataset = getDatasets(trainX, trainY, batch_size=config.settings['Batch']*availableGPUs)
        # or load as you train
        # if args.trainValData:
        #   trainDataset = getG4Datasets_dataAPI(args.trainValData, batch_size=config.settings['Batch']*availableGPUs)
        # else:
        #   trainDataset = getG4Datasets_dataAPI(args.trainData, batch_size=config.settings['Batch']*availableGPUs)

        # get the optimizer
        myOptimizer = eval(config.settings['Optimizer']+'('+
            'learning_rate='+str(config.settings['LearningRate']*availableGPUs)+','+
            'beta_1='+str(config.settings['b1'])+','+
            'beta_2='+str(config.settings['b2'])+')'
            ,{'__builtins__':None}, config.dispatcher)

        # get the DNN model
        mlp_model = congigureAndGetModel(layers=config.settings['Layers'],
                                         nodes=config.settings['Nodes'],
                                         activation=config.settings['Activation'],
                                         output_activation=config.settings['OutputActivation'],
                                         loss=eval(config.settings['Loss']+'('+str(config.settings['negPunish'])+')', {'__builtins__':None}, config.dispatcher) if 'getNoOverestimateLossFunction' in config.settings['Loss'] else config.settings['Loss'],
                                         optimizer=myOptimizer)

        # get lr scheduler functions
        expIncreaseTestFunc = expIncreaseTest(config.settings['LearningRate'])
        oneCycleFunc = oneCycle(config.settings['LearningRate'], 0.1, 10, 100)

        # callbacks
        callbacks_list = [
            EarlyStopping(monitor='val_mae', min_delta=0.001, patience=5, restore_best_weights=True)
            # LearningRateScheduler(oneCycleFunc)
            ]

        # fit model
        history = mlp_model.fit(trainDataset,
                                epochs=config.settings['Epochs'],
                                validation_data=valDataset,
                                callbacks=callbacks_list)

        # save model and print learning rate
        if not args.test: 
            mlp_model.save('data/mlp_model_'+timestamp+'.h5')
            logConfiguration(config.settings)
            # plot training
            plotTrainingMetrics(history.history)            
            system('mkdir -p plots/'+today+'/'+timestamp)
            plt.savefig('plots/'+today+'/'+timestamp+'/learning_'+timestamp+'.pdf')
            print("Trained model saved! Timestamp:", timestamp)

    # load MLP model
    else:
        timestamp = [item.split('_')[-1].split('.')[0] for item in args.model.split('/') if 'mlp' in item][0]
        # unfortunately the negPunish is not saved but I can retrieve it from the logs
        logDic = json.load(open('data/modelConfig_'+timestamp+'.json'))
        retValNegPunish = logDic['negPunish'] if 'negPunish' in logDic.keys() else 1

        mlp_model = tf.keras.models.load_model(args.model, 
                                               custom_objects={'noOverestimateLossFunction':getNoOverestimateLossFunction(negPunish=retValNegPunish),
                                                               'overestimationMetric':overestimationMetric})
        # print model summary
        mlp_model.summary()

    # predict on validation data
    print("Calculating validation predictions for %i points..." % len(valX))
    pred_valY = mlp_model.predict(valX)
    # pred_trainY = mlp_model.predict(trainX)

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
        validationPlots.plotPerformance()
        # validationPlots.plotInputs()
        # trainPlots = Plot('training', timestamp, inputFeatures=trainX, truth=trainY, prediction=pred_trainY)
        # trainPlots.plotInputs()
        # trainPlots.plotPerformance()
    
    print("Done!")
