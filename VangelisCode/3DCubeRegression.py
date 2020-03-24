import os,pdb,argparse
import tensorflow as tf
import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
from os import system
from datetime import date,datetime
import h5py

def getG4Datasets(HDF5File):
	'''
	Construct the test datasets generated from Geant4.
	arguments
		HDF5File: the data file
	return
		test_data_input, test_data_output: 1 x tf.Tensor. Input shape (i,6), output shape (i,1).
	'''
	
	# load data
	file = h5py.File(args.G4Dataset,'r')
	
	X = np.array(file['default_ntuples']['B4']['x']['pages'])
	Y = np.array(file['default_ntuples']['B4']['y']['pages'])
	Z = np.array(file['default_ntuples']['B4']['z']['pages'])
	Xprime = np.array(file['default_ntuples']['B4']['dx']['pages'])
	Yprime = np.array(file['default_ntuples']['B4']['dy']['pages'])
	Zprime = np.array(file['default_ntuples']['B4']['dz']['pages'])
	L = np.array(file['default_ntuples']['B4']['distance']['pages'])
	
	test_data_input = tf.convert_to_tensor(np.column_stack((X,Y,Z,Xprime,Yprime,Zprime)))
	# a normalisation of the output is also happening
	test_data_output = tf.convert_to_tensor(np.column_stack(L/math.sqrt(3)).T)

	return test_data_input, test_data_output

def getDatasets(pickleFile, validation_ratio=0.2):
	'''
	Construct the training and validation datasets.
	arguments
		pickleFile: the data file
		validation_ratio: portion of the data that comprise the validation dataset
	return
		train_data_input, train_data_output, validation_data_input, validation_data_output: 4 x tf.Tensor. Input shape (0,6), output shape (0,1).
	'''
	
	# load data
	dataset = pd.DataFrame(pd.read_pickle(pickleFile))
	# split data
	split_index = int(len(dataset) * (1 - validation_ratio))
	train_data = dataset.head(split_index)
	validation_data = dataset.tail(len(dataset) - split_index)
	
	# convert DataFrame to tf.Tensor
	train_data_input = tf.convert_to_tensor(train_data[['X','Y','Z','Xprime','Yprime','Zprime']].values)
	validation_data_input = tf.convert_to_tensor(validation_data[['X','Y','Z','Xprime','Yprime','Zprime']].values)
	# a normalisation of the output is also happening
	train_data_output = tf.convert_to_tensor(train_data[['L']].values/math.sqrt(3))
	validation_data_output = tf.convert_to_tensor(validation_data[['L']].values/math.sqrt(3))

	return train_data_input, train_data_output, validation_data_input, validation_data_output

def getMLP(inputShape, hiddenNodes, fActivation='relu', fOptimizer='adam'):
	'''
	Construct a fully connected MLP.
	arguments
		hiddenNodes: list of int. len(hiddenNodes) specifies the number of hiddel layers
		fActivation: activation function string
		fOptimizer: optimizer string
	return
		model: keras model
	'''
	
	# construct
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Dense(hiddenNodes[0], input_shape=(inputShape,), activation=fActivation))
	for layer in range(1, len(hiddenNodes)):
		model.add(tf.keras.layers.Dense(hiddenNodes[layer], activation=fActivation))
	# I don't understand why I need to put linear and not relu
	model.add(tf.keras.layers.Dense(1, activation='linear'))

	# compile
	model.compile(
		loss='mse',
		optimizer=fOptimizer,
		metrics=['mae'])
	
	return model

def printTable(dictionary):
	string = ""
	# for item in [key+': '+str(dictionary[key])+'\n' for key in dictionary]:
	for item in [key+': '+str(dictionary[key])+', ' for key in dictionary]:
		string += item
	return string.rstrip()

def sanityPlots(prediction, truth, details, savename='results'):
	'''
	Produce a set of basic sanity/validation plots
	arguments
		prediction: the predicted np.array from MLP
		truth: the truth np.array
	'''
	print("Hi from Plotter!")

	# create dir
	today = date.today()
	saveDir = 'plots/'+str(today)
	system('mkdir -p '+saveDir)

	# create subplot env with shared y axis
	fig, axs = plt.subplots(1,3)

	# truth length
	truth_length = truth*math.sqrt(3)
	axs[0].hist(truth_length, bins=100)
	axs[0].set(xlabel='Truth L', ylabel='Points')
	
	# predicted length
	pred_length = prediction*math.sqrt(3)
	axs[1].hist(pred_length, bins=100)
	axs[1].set(xlabel='Predicted L')

	# error
	error = np.divide(truth_length - pred_length, truth_length, out=np.zeros_like(truth_length - pred_length), where=truth_length!=0)
	abs_error = abs(error)
	axs[2].hist(error, bins=100, log=True)
	axs[2].set(xlabel='Truth L - Predicted L / Truth L')

	# NN details
	fig.suptitle(printTable(details), size='xx-small')
	# plt.text(0.6, 0.8, printTable(details), transform = axs[2].transAxes, size=5, bbox=dict(facecolor='white'))

	# save
	now = datetime.now()
	timestamp = int(datetime.timestamp(now))
	plt.savefig(saveDir+'/'+savename+'_'+str(timestamp)+'.pdf')
	print(savename+".pdf Saved!")

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Regress distance to the boundary of a unit cube from 3D points and directions.')
	parser.add_argument('--myDataset', help='Pickle file of the dataset.', required=True)
	parser.add_argument('--plots', help='Produce sanity plots.', default=False, action='store_true')
	parser.add_argument('--model', help='Use a previously trained and saved MLP model.', default=None)
	parser.add_argument('--G4Dataset', help='Import Geant4 generated dataset (HDF5 format).', default=None)
	args = parser.parse_args()

	# get input/output train and validation data
	trainX, trainY, valX, valY = getDatasets(args.myDataset, validation_ratio=0.1)
	# validation_split: keep a small portion of the training data blind just to calculate performance during training.
	validation_split = 0.01

	# define your settings
	# todo: I need to somehow read these when I load the model
	settings = {
		'Structure'				:	[20, 50, 20, 10],
		'Optimizer'				:	'adam',
		'Activation'			:	'relu',
		'Batch'					:	64,
		'Epochs'				:	4,
		'Training Sample Size'	:	int(trainX.shape[0]*(1-validation_split))
	}

	# create MLP model
	if args.model is None:
		mlp_model = getMLP(inputShape=trainX.shape[1], hiddenNodes=settings['Structure'], fActivation=settings['Activation'], fOptimizer=settings['Optimizer'])
		# print model summary
		mlp_model.summary()
		# fit model
		mlp_model.fit(trainX, trainY, epochs=settings['Epochs'], batch_size=settings['Batch'], validation_split=validation_split)
		# save model
		today = date.today()
		mlp_model.save('mlp_model_'+str(today))
	# load MLP model
	else:
		mlp_model = tf.keras.models.load_model(args.model)
		# print model summary
		mlp_model.summary()

	# predict on validation data
	print("Calculating predictions for %i points..." % len(valX))
	pred_valY = mlp_model.predict(valX)

	# prepare G4 dataset
	if args.G4Dataset is not None:
		g4X, g4Y = getG4Datasets(args.G4Dataset)
		# comment: I'm not so sure what to do for g4Y zero values.

	# predict on G4 data
	print("Calculating predictions for %i points..." % len(g4X))
	pred_g4Y = mlp_model.predict(g4X)

	# plot
	if args.plots: 
		sanityPlots(pred_valY, valY.numpy(), settings, savename='resultsVal')
		sanityPlots(pred_g4Y, g4Y.numpy(), settings, savename='resultsG4')
	
	print("Done!")
