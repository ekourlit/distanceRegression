import os,pdb,argparse
import tensorflow as tf
import pandas as pd
import numpy as np
import math
import matplotlib as mpl
from matplotlib import pyplot as plt
from os import system
import h5py,csv,pickle
from datetime import date,datetime
timestamp = str(datetime.now().year)+str("{:02d}".format(datetime.now().month))+str("{:02d}".format(datetime.now().day))+str("{:02d}".format(datetime.now().hour))+str("{:02d}".format(datetime.now().minute))

#tf.data API WIP
def pack(features, length):
	'''
	pack together tf.data.Dataset columns and normalise length
	generally data preprocessing can be added here
	'''
	return tf.stack(list(features.values()), axis=-1), length/normalisation

#tf.data API WIP
def consumeCSV(path, **kwards):
	'''
	create tf.data.dataset by mapping on a tf.data.dataset containing file paths
	generally data preprocessing can be added here
	# try to use tf.io.decode_csv
	'''
	scv_columns = ['X','Y','Z','Xprime','Yprime','Zprime','L']
	return tf.data.experimental.make_csv_dataset(path, batch_size=int(1e10), column_names=scv_columns, label_name=scv_columns[-1], num_epochs=1)

#tf.data API WIP
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

#tf.data API WIP
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
		assert (np.any(L>normalisation)==False), "There are too large lengths in your dataset!"

		test_data_input = tf.convert_to_tensor(np.column_stack((X,Y,Z,Xprime,Yprime,Zprime)))
		# a normalisation of the output is also happening
		test_data_output = tf.convert_to_tensor(np.column_stack(L/normalisation).T)

		return test_data_input, test_data_output

	elif '.csv' in G4FilePath:

		data = np.loadtxt(G4FilePath, delimiter=',')
		L = (data[:,6]/normalisation).reshape(data[:,6].size,1)
		assert (np.any(L>1)==False), "There are too large lengths in your dataset!"

		test_data_input = tf.convert_to_tensor(data[:,:6])
		test_data_output = tf.convert_to_tensor(L)

		return test_data_input, test_data_output

		# using tf.data API WIP
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
	# tf.data API for parallel mutiple file loading WIP
	if '*' in pickleFile:
		file_list = tf.data.Dataset.list_files(pickleFile)
		dataset = file_list.map(tf_consumePickle)
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
	train_data_output = tf.convert_to_tensor(train_data[['L']].values/normalisation)
	validation_data_output = tf.convert_to_tensor(validation_data[['L']].values/normalisation)

	return train_data_input, train_data_output, validation_data_input, validation_data_output

def getMLP(inputShape, hiddenNodes, fActivation='relu', fOptimizer='adam', fOutputActivation='sigmoid', fLoss='mse'):
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
	model.add(tf.keras.layers.Dense(1, activation=fOutputActivation))

	# compile
	model.compile(
		loss=fLoss,
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
	# plt.clf()

	# create dir
	today = date.today()
	saveDir = 'plots/'+str(today)
	system('mkdir -p '+saveDir)

	# create subplot env with shared y axis
	fig, axs = plt.subplots(1,3)
	fig.tight_layout(pad=1.3)

	# truth length
	truth_length = truth*normalisation
	axs[0].hist(truth_length, bins=100, range=(0,2))
	axs[0].set(xlabel='Truth L', ylabel='Points')
	
	# predicted length
	pred_length = prediction*normalisation
	axs[1].hist(pred_length, bins=100, range=(0,2))
	axs[1].set(xlabel='Predicted L')

	# error
	error = np.divide(truth_length - pred_length, truth_length, out=np.zeros_like(truth_length - pred_length), where=truth_length!=0)
	abs_error = abs(error)
	axs[2].hist(error, bins=100, log=True, range=(-1,1))
	axis = plt.gca()
	axis.minorticks_on()
	axs[2].set(xlabel='Truth L - Predicted L / Truth L')

	# NN details
	fig.suptitle(printTable(details), size='xx-small')
	# plt.text(0.6, 0.8, printTable(details), transform = axs[2].transAxes, size=5, bbox=dict(facecolor='white'))

	# save
	plt.tight_layout()
	plt.savefig(saveDir+'/'+savename+'_'+timestamp+'.pdf')
	print(savename+'_'+timestamp+".pdf Saved!")

	# 2D truth vs predicted marginal
	# plt.clf()
	# import seaborn as sns
	# sns.set(style="ticks")
	# plot = sns.jointplot(truth_length, pred_length, kind="hex", color="#2E5D9F", joint_kws=dict(gridsize=100))
	# plot.set_axis_labels('Truth L', 'Predicted L')
	# # save
	# plt.tight_layout()
	# plt.savefig(saveDir+'/'+savename+'_cor_'+timestamp+'.pdf')
	# print(savename+'_cor_'+timestamp+".pdf Saved!")

	# scatter
	plt.clf()
	plt.scatter(truth_length, pred_length, s=0.1)
	axis = plt.gca()
	axis.set_xlabel('Truth L')
	axis.set_ylabel('Predicted L')
	# save
	plt.tight_layout()
	plt.savefig(saveDir+'/'+savename+'_scatt_'+timestamp+'.png')
	print(savename+'_scatt_'+timestamp+".pdf Saved!")

	# hist2d
	plt.clf()
	plt.hist2d(truth_length.reshape(len(truth_length),), pred_length.reshape(len(pred_length),), bins=(300,300), norm=mpl.colors.LogNorm())
	plt.grid()
	axis = plt.gca()
	axis.set_xlabel('Truth L')
	axis.set_ylabel('Predicted L')
	# save
	plt.tight_layout()
	plt.savefig(saveDir+'/'+savename+'_hist2d_'+timestamp+'.pdf')
	print(savename+'_hist2d_'+timestamp+".pdf Saved!")

def combPerfPlots(Vprediction, Vtruth, Tprediction=None, Ttruth=None, savename='pred_error'):
	'''
	Produce a set of basic sanity/validation plots
	arguments
		Vprediction, Tprediction: the validation and test predicted np.array from MLP
		Vtruth, Ttruth: the validation and test truth np.array
	'''
	print("Hi from Plotter!")
	plt.clf()

	# create dir
	today = date.today()
	saveDir = 'plots/'+str(today)
	system('mkdir -p '+saveDir)

	if (Tprediction is not None) and (Ttruth is not None): combined = True
	else: combined = False

	# validation
	val_truth_length = Vtruth*normalisation
	val_pred_length = Vprediction*normalisation
	val_error = np.divide(val_truth_length - val_pred_length, val_truth_length, out=np.zeros_like(val_truth_length - val_pred_length), where=val_truth_length!=0)
	val_abs_error = abs(val_error)
	
	if combined:
		test_truth_length = Ttruth*normalisation
		test_pred_length = Tprediction*normalisation
		test_error = np.divide(test_truth_length - test_pred_length, test_truth_length, out=np.zeros_like(test_truth_length - test_pred_length), where=test_truth_length!=0)
		test_abs_error = abs(test_error)

	# what portion of the predictions has at least x error?
	intervals = np.logspace(-4, 0, num=100)
	val_counts = []
	test_counts = []
	for limit in intervals: 
		val_counts.append(np.count_nonzero(val_abs_error>=limit))
		if combined: test_counts.append(np.count_nonzero(test_abs_error>=limit))
	
	plt.plot(intervals, np.array(val_counts)/val_abs_error.size, color='#ff7f0e', linewidth=3)
	if combined: plt.plot(intervals, np.array(test_counts)/test_abs_error.size, color='#1f77b4', linewidth=3)
	plt.legend(['Validation','Test'], loc='upper right')

	plt.grid()
	axis = plt.gca()
	axis.set_xscale('log')
	axis.set_xlabel('Relative Error')
	axis.set_ylabel('Dataset Portion')

	# save
	plt.savefig(saveDir+'/'+savename+'_'+timestamp+'.pdf')
	print(savename+'_'+timestamp+".pdf Saved!")

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
	validation_split = 0.05

	# define your settings
	# todo: I need to somehow read these when I load the model
	settings = {
		'Structure'				:	[1024, 1024],
		'Optimizer'				:	'adam',
		'Activation'			:	'relu',
		'OutputActivation'		:	'sigmoid',
		'Loss'					:	'mse',
		'Batch'					:	128,
		'Epochs'				:	50,
		'Training Sample Size'	:	int(trainX.shape[0]*(1-validation_split))
	}

	# create MLP model
	if args.model is None:
		mlp_model = getMLP(inputShape=trainX.shape[1], hiddenNodes=settings['Structure'], fActivation=settings['Activation'], fOptimizer=settings['Optimizer'], fOutputActivation=settings['OutputActivation'], fLoss=settings['Loss'])
		# print model summary
		mlp_model.summary()
		# fit model
		history = mlp_model.fit(trainX, trainY, epochs=settings['Epochs'], batch_size=settings['Batch'], validation_split=validation_split)

		# Plot training & validation loss values
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.ylabel('Loss (MSE)')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Validation'], loc='upper right')
		today = str(date.today())
		system('mkdir -p plots/'+today)
		plt.savefig('plots/'+today+'/learning_'+timestamp+'.pdf')
		mlp_model.save('data/mlp_model_'+timestamp)

	# load MLP model
	else:
		mlp_model = tf.keras.models.load_model(args.model)
		# print model summary
		mlp_model.summary()
		for item in args.model.split('/'):
			if 'mlp' in item: timestamp = item.split('_')[-1]

	# predict on validation data
	print("Calculating predictions for %i points..." % len(valX))
	pred_valY = mlp_model.predict(valX)
	
	# prepare G4 dataset
	if args.G4Dataset is not None:
		g4X, g4Y = getG4Datasets(args.G4Dataset)
		# g4_data = getG4Datasets(args.G4Dataset) #tf.data API WIP

		# predict on G4 data
		print("Calculating predictions for %i points..." % len(g4X))
		pred_g4Y = mlp_model.predict(g4X)

	# plot
	if args.plots: 
		perfPlots(prediction=pred_valY, truth=valY.numpy(), details=settings, savename='resultsVal')
		inputPlots(valX.numpy(), savename='inputsVal')
		if args.G4Dataset is not None: 
			perfPlots(prediction=pred_g4Y, truth=g4Y.numpy(), details=settings, savename='resultsG4')
			inputPlots(g4X.numpy(), savename='inputsG4')
		
		# combined plots
		if args.G4Dataset is not None: combPerfPlots(Vprediction=pred_valY, Vtruth=valY.numpy(), Tprediction=pred_g4Y, Ttruth=g4Y.numpy())
		else: combPerfPlots(Vprediction=pred_valY, Vtruth=valY.numpy())		
	
	print("Done!")
