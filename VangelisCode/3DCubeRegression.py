import os,pdb,argparse
import tensorflow as tf
import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
from os import system
from datetime import date,datetime

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
	
	# print model summary
	model.summary()
	return model

def printTable(dictionary):
	string = ""
	for item in [key+': '+str(dictionary[key])+'\n' for key in dictionary]:
		string += item
	return string

def sanityPlots(prediction, truth, details):
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

	# predicted length
	pred_length = prediction*math.sqrt(3)
	axs[0].hist(pred_length, bins=100)
	axs[0].set(xlabel='Predicted L', ylabel='Points')

	# truth length
	pred_length = truth*math.sqrt(3)
	axs[1].hist(pred_length, bins=100)
	axs[1].set(xlabel='Truth L')

	# error
	error = (prediction - truth)/math.sqrt(3)
	abs_error = abs(error)
	axs[2].hist(error, bins=100)
	axs[2].set(xlabel='Predicted L - Truth L')

	# NN details
	plt.text(0.6, 0.8, printTable(details), transform = axs[2].transAxes, size=5, bbox=dict(facecolor='white'))

	# save
	now = datetime.now()
	timestamp = int(datetime.timestamp(now))
	plt.savefig(saveDir+'/results_'+str(timestamp)+'.pdf')
	print("results.pdf Saved!")

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Regress distance to the boundary of a unit cube from 3D points and directions.')
	parser.add_argument('--dataFile', help='Pickle file of the dataset.', required=True)
	parser.add_argument('--plots', help='Produce sanity plots.', default=False, action='store_true')
	args = parser.parse_args()

	# get input/output train and validation data
	trainX, trainY, valX, valY = getDatasets(args.dataFile, validation_ratio=0.05)
	# validation_split: keep a small portion of the training data blind just to calculate performance during training.
	validation_split = 0.01

	# define your settings
	settings = {
		'Structure' 			: 	[10, 50, 10, 6],
		'Optimizer' 			: 	'adam',
		'Activation' 			: 	'relu',
		'Batch'					:	500,
		'Epochs'				:	3,
		'Training Sample Size'	:	int(trainX.shape[0]*(1-validation_split))
	}

	# MLP model
	mlp_model = getMLP(inputShape=trainX.shape[1], hiddenNodes=settings['Structure'], fActivation=settings['Activation'], fOptimizer=settings['Optimizer'])

	# fit model
	mlp_model.fit(trainX, trainY, epochs=settings['Epochs'], batch_size=settings['Batch'], validation_split=validation_split)

	# predict
	print("Calculating predictions for %i points..." % len(valX))
	predY = mlp_model.predict(valX)

	# plot
	if args.plots: sanityPlots(predY, valY.numpy(), settings)
	print("Done!")
