import os,pdb,argparse
import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def getDatasets(pickleFile, validation_ratio=0.2):
	'''
	Construct the training and validation datasets.
	arguments
		pickleFile: the data file
		validation_ratio: portion of the data that comprise the validation dataset
	return
		train_data_input, train_data_output, validation_data_input, validation_data_output: 4 x tf.Tensor. Input shape (0,4), output shape (0,1).
	'''
	
	# load data
	dataset = pd.DataFrame(pd.read_pickle(pickleFile))
	# split data
	split_index = int(len(dataset) * (1 - validation_ratio))
	train_data = dataset.head(split_index)
	validation_data = dataset.tail(len(dataset) - split_index)
	
	# convert DataFrame to tf.Tensor
	train_data_input = tf.convert_to_tensor(train_data[['X','Y','Xprime','Yprime']].values, dtype=tf.float32)
	train_data_output = tf.convert_to_tensor(train_data[['L']].values, dtype=tf.float32)
	validation_data_input = tf.convert_to_tensor(validation_data[['X','Y','Xprime','Yprime']].values, dtype=tf.float32)
	validation_data_output = tf.convert_to_tensor(validation_data[['L']].values, dtype=tf.float32)

	return train_data_input, train_data_output, validation_data_input, validation_data_output

def getMLP(inputShape, hiddenNodes, activationFunc='relu', myOptimizer='adam'):
	'''
	Construct a fully connected MLP.
	arguments
		hiddenNodes: list of int. len(hiddenNodes) specifies the number of hiddel layers
		activationFunc: activation function string
	return
		model: keras model
	'''
	
	# construct
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Dense(hiddenNodes[0], input_shape=(inputShape,), activation=activationFunc))
	for layer in range(1, len(hiddenNodes)):
		model.add(tf.keras.layers.Dense(hiddenNodes[layer], activation=activationFunc))
	# I don't understand why I need to put linear and not relu
	model.add(tf.keras.layers.Dense(1, activation='linear'))

	# compile
	model.compile(
		loss='mse',
		optimizer=myOptimizer,
		metrics=['mae'])
	
	# print model summary
	model.summary()
	return model

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Regress distance to the boundary of a unit box from 2D points and directions.')
	parser.add_argument('--datasetFile', help='Pickle file of the dataset.', required=True)
	args = parser.parse_args()

	# get input/output train and validation data
	trainX, trainY, valX, valY = getDatasets(args.datasetFile, validation_ratio=0.1)

	# MLP model
	mlp_model = getMLP(inputShape=trainX.shape[1], hiddenNodes=[10, 10, 4])

	# fit model
	# validation_split: keep a small portion of the data blind just to calculate performance during training.
	mlp_model.fit(trainX, trainY, epochs=10, batch_size=100, validation_split=0.05)

	# predict
	print("Calculating predictions for %i points..." % len(valX))
	predY = mlp_model.predict(valX)

	# plot results
	print("Plotting results...")
	result_error = abs(predY - valY.numpy())
	hist = plt.hist(result_error, bins=100)
	ax = plt.gca()
	ax.set_xlabel('|Predicted L - Truth L|')
	ax.set_ylabel('Points')
	plt.savefig('hist.pdf')
	print("Done!")
