import os,pdb,argparse,sys
import tensorflow as tf
from models import *
from plotUtils import Plot
# import mlflow.keras
import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
from os import system
import h5py,csv,pickle
from datetime import date,datetime
from tensorflow.keras.optimizers import *
timestamp = str(datetime.now().year)+str("{:02d}".format(datetime.now().month))+str("{:02d}".format(datetime.now().day))+str("{:02d}".format(datetime.now().hour))+str("{:02d}".format(datetime.now().minute))
today = str(date.today())

def consumeCSV(path):
    '''
    create tf.data.dataset by mapping on a tf.data.dataset containing file paths
    that's a great function but it still experimental and doesn't work with AutoGraph (not convertible)
    '''
    scv_columns = ['X','Y','Z','Xprime','Yprime','Zprime','L']
    return tf.data.experimental.make_csv_dataset(path, batch_size=1, column_names=scv_columns, label_name=scv_columns[-1], num_epochs=1)

@tf.function
def process_csv_line(line):
    '''
    Process each csv line of the dataset.
    Data preprocessing can be done nere.
    '''
    # parse csv line
    fields = tf.io.decode_csv(line, [tf.constant([np.nan], dtype=tf.float32)] * 7)
    # first 3 fields are X,Y,Z. normalize them
    position = tf.stack(fields[:3])/positionNormalisation
    # next 3 fields are Xprime,Yprime,Zprime
    momentum = tf.stack(fields[3:-1])
    # stack and flatten them
    features = tf.reshape(tf.stack([position,momentum]), [-1])
    # last field is the length. normalize it.
    length = tf.stack(fields[-1:])/lengthNormalisation
    return features, length

def getG4Datasets_dataAPI(G4FilePath, batch_size=32, shuffle_buffer_size=1000):
    '''
    Load datasets generated from Geant4.
    arguments
        G4FilePath: file path, can be wildcarded
        batch_size: the batch size to slice the dataset
        shuffle_buffer_size: the shuffle hand size
    return
        dataset: tf.data.Dataset
    '''

    # using tf.data API
    if dataAPI:
        # create a file list dataset
        file_list = tf.data.Dataset.list_files(G4FilePath)
        # create TextLineDatasets (lines) from the above list
        dataset = file_list.interleave(
            lambda path: tf.data.TextLineDataset(path).skip(15), #skip the first 15 lines as it's header
            # cycle_length=1) # the number of paths it concurrently process from file_list
            num_parallel_calls=tf.data.experimental.AUTOTUNE) 
        # parse & process csv line
        dataset = dataset.map(process_csv_line)
        # keep a hand in memory and shuffle
        dataset = dataset.shuffle(shuffle_buffer_size)
        # chop in batches and prepare in CPU 1 bach ahead before you feed into evaluation
        dataset = dataset.batch(batch_size).prefetch(1)

        return dataset

def getG4Datasets(G4FilePath, split_input=False):
    '''
    Construct the test datasets generated from Geant4.
    arguments
        G4FilePath: file path
    return
        data_input, data_output: 1 x tf.Tensor. Input shape (i,6), output shape (i,1).
    '''
    
    if '.hdf5' in G4FilePath:
        file = h5py.File(G4FilePath,'r')
        
        X = np.array(file['default_ntuples']['B4']['x']['pages'])/positionNormalisation
        Y = np.array(file['default_ntuples']['B4']['y']['pages'])/positionNormalisation
        Z = np.array(file['default_ntuples']['B4']['z']['pages'])/positionNormalisation
        Xprime = np.array(file['default_ntuples']['B4']['dx']['pages'])
        Yprime = np.array(file['default_ntuples']['B4']['dy']['pages'])
        Zprime = np.array(file['default_ntuples']['B4']['dz']['pages'])
        L = np.array(file['default_ntuples']['B4']['distance']['pages'])
        assert (np.any(L>lengthNormalisation)==False), "There are too large lengths in your dataset!"

        data_input = tf.convert_to_tensor(np.column_stack((X,Y,Z,Xprime,Yprime,Zprime)))
        # a normalisation of the output is also happening
        data_output = tf.convert_to_tensor(np.column_stack(L/lengthNormalisation).T)

        return data_input, data_output

    # csv input
    else:
        # to-do: I want to feed a wildcarded path here
	    data = np.loadtxt(G4FilePath, delimiter=',', skiprows=15)

	    L = (data[:,6]/lengthNormalisation).reshape(data[:,6].size, 1)
	    if lengthNormalisation != 1: assert (np.any(L>1)==False), "There are too large lengths in your dataset!"
	    
	    # normalise X, Y, Z 
	    positions = data[:,:3]/positionNormalisation
	    # X', Y', Z'
	    directions = data[:,3:6]

	    if split_input == True:
		    data_input = {'position' : tf.convert_to_tensor(positions), 'direction' : tf.convert_to_tensor(directions)}
	    else:
	    	inputs = np.concatenate((positions, directions), axis=1)
	    	data_input = tf.convert_to_tensor(inputs)
	    
	    data_output = tf.convert_to_tensor(L)

	    return data_input, data_output

def logConfiguration(dictionary):
    f = open('data/modelConfig_'+timestamp+'.txt','w')
    f.write( str(dictionary) )
    f.close()

# enable MLflow autologging XLA
# mlflow.keras.autolog()

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
    positionNormalisation = 1600
    
    # define your settings
    settings = {
        'Description'       :    "test simple sequential feed-forward network",
        # 'Structure'         :    {'Position' : [512,512,512], 'Direction' : [], 'Concatenated' : [1024,1024]},
        'Structure'         :    [1024,1024],
        'Activation'        :    'relu',
        'OutputActivation'	:    'relu',
        'Loss'              :    'mse',
        'Optimizer'         :    Adam(learning_rate=0.005),
        'Batch'             :    256,
        'Epochs'            :    100
    }

    # create MLP model
    if args.model is None:

        # get some data to train on
        # load everything in memory
        trainX, trainY = getG4Datasets(args.trainData)

        mlp_model = getSplitMLP(settings['Structure'],
                                activation=settings['Activation'],
                                output_activation=settings['OutputActivation'],
                                loss=settings['Loss'],
                                optimizer=settings['Optimizer'])

        # fit model
        history = mlp_model.fit(trainX, trainY,
                                epochs=settings['Epochs'],
                                batch_size=settings['Batch'],
                                validation_split=0.01) # not supported with data API pipeline but you can feed dataset
                                # validation_data=validationData)

        # save model and print learning rate
        if not args.test: 
            mlp_model.save('data/mlp_model_'+timestamp)
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
        mlp_model = tf.keras.models.load_model(args.model)
        # print model summary
        mlp_model.summary()
        timestamp = [item.split('_')[-1] for item in args.model.split('/') if 'mlp' in item][0]

    # predict on validation data
    print("Loading all validation data in memory...")
    valX, valY = getG4Datasets(args.validationData)
    print("Calculating validation predictions for %i points..." % len(valX))
    pred_valY = mlp_model.predict(valX)

    # that would be the test dataset
    if args.testData is not None:
        print("Loading all test data in memory...")
        testX, testY = getG4Datasets(args.testData)
        # predict on test data
        print("Calculating test predictions for %i points..." % len(testX))
        pred_testX = mlp_model.predict(testX)

    # plot
    if args.plots: 
        myPlots = Plot('validation', timestamp, inputFeatures=valX.numpy(), truth=valY.numpy(), prediction=pred_valY)
        myPlots.plotPerformance()
    
    print("Done!")
