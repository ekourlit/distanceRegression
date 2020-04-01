import os,pdb,argparse
import tensorflow as tf
import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
from os import system
from datetime import date,datetime
import h5py,csv,pickle

def pack(features, length):
	'''
	pack together tf.data.Dataset columns and normalise length
	generally data preprocessing can be added here
	'''
	return tf.stack(list(features.values()), axis=-1), length/normalisation

def consumeCSV(path, **kwards):
	'''
	create tf.data.dataset by mapping on a tf.data.dataset containing file paths
	generally data preprocessing can be added here
	# try to use tf.io.decode_csv
	'''
	scv_columns = ['X','Y','Z','Xprime','Yprime','Zprime','L']
	return tf.data.experimental.make_csv_dataset(path, batch_size=int(1e10), column_names=scv_columns, label_name=scv_columns[-1], num_epochs=1)

def consumePickle(path):
	'''
	similar to consumeCSV
	'''
	file = open(path.numpy(),'rb')
	data_dict = pickle.load(file)
	columns = len(data_dict.keys())
	firstColumn = next(iter(data_dict))
	entries = len(data_dict[firstColumn])
	dataset = tf.data.Dataset.from_tensor_slices(data_dict)
	return dataset, entries, columns

def tf_consumePickle(path):
	'''
	the tf function that feed into the map
	'''
	# path_shape = path.shape
	# [path,] = tf.py_function(func=consumePickle, inp=[path], Tout=[tf.float32])
	# path.set_shape(path_shape)
	# return path
	dataset, entries, columns = tf.py_function(func=consumePickle, inp=[path], Tout=[tf.float32, tf.int32, tf.int32])
	for example in dataset: break
	pdb.set_trace()
	return dataset

def getG4Datasets(G4FilePath):
	'''
	Construct the test datasets generated from Geant4.
	arguments
		HDF5File: the data file
	return
		test_data_input, test_data_output: 1 x tf.Tensor. Input shape (i,6), output shape (i,1).
	'''
	
	if '.hdf5' in G4FilePath:
		file = h5py.File(G4FilePath,'r')
		
		X = np.array(file['default_ntuples']['B4']['x']['pages'])
		Y = np.array(file['default_ntuples']['B4']['y']['pages'])
		Z = np.array(file['default_ntuples']['B4']['z']['pages'])
		Xprime = np.array(file['default_ntuples']['B4']['dx']['pages'])
		Yprime = np.array(file['default_ntuples']['B4']['dy']['pages'])
		Zprime = np.array(file['default_ntuples']['B4']['dz']['pages'])
		L = np.array(file['default_ntuples']['B4']['distance']['pages'])
		assert (np.any(L>1)==False), "There are too large lengths in your dataset!"

		test_data_input = tf.convert_to_tensor(np.column_stack((X,Y,Z,Xprime,Yprime,Zprime)))
		# a normalisation of the output is also happening
		test_data_output = tf.convert_to_tensor(np.column_stack(L/normalisation).T)

		return test_data_input, test_data_output

	elif '.csv' in G4FilePath:


		# this is my classic way to load a csv dataset into tf Tensors
		data = np.loadtxt(G4FilePath, delimiter=',')
		L = (data[:,6]/normalisation).reshape(data[:,6].size,1)
		assert (np.any(L>1)==False), "There are too large lengths in your dataset!"

		test_data_input = tf.convert_to_tensor(data[:,:6])
		test_data_output = tf.convert_to_tensor(L)

		return test_data_input, test_data_output

		# using tf.data API
		# if '*' in G4FilePath:
		# 	file_list = tf.data.Dataset.list_files(G4FilePath)
		# 	dataset = file_list.map(consumeCSV)

		# scv_columns = ['X','Y','Z','Xprime','Yprime','Zprime','L']
		# # as these data are used only for inference at the moment, the batch_size can be arbitary large and num_epochs = 1
		# dataset = tf.data.experimental.make_csv_dataset(G4FilePath, batch_size=int(1e10), column_names=scv_columns, label_name=scv_columns[-1], num_epochs=1)
		# packed_dataset = dataset.map(pack)
		# return packed_dataset

def getDatasets(pickleFile, validation_ratio=0.2):
	'''
	Construct the training and validation datasets.
	arguments
		pickleFile: the data file
		validation_ratio: portion of the data that comprise the validation dataset
	return
		train_data_input, train_data_output, validation_data_input, validation_data_output: 4 x tf.Tensor. Input shape (0,6), output shape (0,1).
	'''

	'''
	# tf.data API for parallel mutiple file loading
	if '*' in pickleFile:
		file_list = tf.data.Dataset.list_files(pickleFile)
		dataset = file_list.map(tf_consumePickle)
	'''

	# Classic workflow
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
	train_data_output = tf.convert_to_tensor(train_data[['L']].values/normalisation)
	validation_data_output = tf.convert_to_tensor(validation_data[['L']].values/normalisation)

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
	model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

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

def perfPlots(prediction, truth, details, savename='results'):
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
	truth_length = truth*normalisation
	axs[0].hist(truth_length, bins=100)
	axs[0].set(xlabel='Truth L', ylabel='Points')
	
	# predicted length
	pred_length = prediction*normalisation
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

def inputPlots(inputs, savename='inputs'):
	'''
	Produce a set of plots to inspect the inputs
	arguments
		inputs: the inputs array of shape (i,6)
	'''
	print("Hi from Plotter!")

	# create dir
	today = date.today()
	saveDir = 'plots/'+str(today)
	system('mkdir -p '+saveDir)

	# create subplot env with shared y axis
	fig, axs = plt.subplots(3,2)
	fig.tight_layout(pad=2.0)

	# X
	X = inputs[:,0]
	axs[0,0].hist(X, bins=100)
	axs[0,0].set(xlabel='X')

	# Y
	Y = inputs[:,1]
	axs[1,0].hist(Y, bins=100)
	axs[1,0].set(xlabel='Y')

	# Z
	Z = inputs[:,2]
	axs[2,0].hist(Z, bins=100)
	axs[2,0].set(xlabel='Z')

	# Xprime
	Xprime = inputs[:,3]
	axs[0,1].hist(Xprime, bins=100)
	axs[0,1].set(xlabel='X\'')

	# Yprime
	Yprime = inputs[:,4]
	axs[1,1].hist(Yprime, bins=100)
	axs[1,1].set(xlabel='Y\'')

	# Zprime
	Zprime = inputs[:,5]
	axs[2,1].hist(Zprime, bins=100)
	axs[2,1].set(xlabel='Z\'')

	# save
	now = datetime.now()
	timestamp = int(datetime.timestamp(now))
	plt.savefig(saveDir+'/'+savename+'.pdf')
	print(savename+".pdf Saved!")

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Regress distance to the boundary of a unit cube from 3D points and directions.')
	parser.add_argument('--myDataset', help='Pickle file of the dataset.', required=True)
	parser.add_argument('--G4Dataset', help='Import Geant4 generated dataset (HDF5 or CSV format).', default=None)
	parser.add_argument('--plots', help='Produce sanity plots.', default=False, action='store_true')
	parser.add_argument('--model', help='Use a previously trained and saved MLP model.', default=None)
	args = parser.parse_args()

	# set cube length normalisation.
	normalisation = math.sqrt(3)
	
	# get input/output train and validation data
	trainX, trainY, valX, valY = getDatasets(args.myDataset, validation_ratio=0.1)
	# validation_split: keep a small portion of the training data blind just to calculate performance during training.
	validation_split = 0.01

	# define your settings
	# todo: I need to somehow read these when I load the model
	settings = {
		'Structure'				:	[10, 20, 50, 20, 10],
		'Optimizer'				:	'adam',
		'Activation'			:	'relu',
		'Batch'					:	64,
		'Epochs'				:	40,
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
		mlp_model.save('data/mlp_model_'+str(today))
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
		# g4_data = getG4Datasets(args.G4Dataset) #tf.data API

		# predict on G4 data
		print("Calculating predictions for %i points..." % len(g4X))
		pred_g4Y = mlp_model.predict(g4X)

	# plot
	if args.plots: 
		perfPlots(pred_valY, valY.numpy(), settings, savename='resultsVal')
		# inputPlots(valX.numpy(), savename='inputsVal')
		if args.G4Dataset is not None: 
			perfPlots(pred_g4Y, g4Y.numpy(), settings, savename='resultsG4')
			# inputPlots(g4X.numpy(), savename='inputsG4')
	
	print("Done!")
