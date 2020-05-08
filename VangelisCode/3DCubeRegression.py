import os,pdb,argparse
import tensorflow as tf
import mlflow.keras
import pandas as pd
import numpy as np
import math
import matplotlib as mpl
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

def getG4Datasets(G4FilePath, batch_size=32, shuffle_buffer_size=1000, dataAPI=False):
	'''
	Construct the test datasets generated from Geant4.
	arguments
		HDF5File: the data file
	return
		data_input, data_output: 1 x tf.Tensor. Input shape (i,6), output shape (i,1).
	'''
	
	if '.hdf5' in G4FilePath:
		file = h5py.File(G4FilePath,'r')
		
		X = np.array(file['default_ntuples']['B4']['x']['pages'])/positionNormalisation # convert these to -1 - 1 range
		Y = np.array(file['default_ntuples']['B4']['y']['pages'])/positionNormalisation # convert these to -1 - 1 range
		Z = np.array(file['default_ntuples']['B4']['z']['pages'])/positionNormalisation # convert these to -1 - 1 range
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

		# using tf.data API
		if dataAPI:
			# create a file list dataset
			file_list = tf.data.Dataset.list_files(G4FilePath)
			# create TextLineDatasets (lines) from the above list
			dataset = file_list.interleave(
				lambda path: tf.data.TextLineDataset(path).skip(15), #skip the first 15 lines as it's header
				cycle_length=1) # the number of paths it concurrently process from file_list
			# parse & process csv line
			dataset = dataset.map(process_csv_line)
			# keep a hand in memory and shuffle
			dataset = dataset.shuffle(shuffle_buffer_size)
			# chop in batches and prepare in CPU 1 bach ahead before you feed into evaluation
			dataset = dataset.batch(batch_size).prefetch(1)

			return dataset

		# just load everything in memory
		else:
			data = np.loadtxt(G4FilePath, delimiter=',', skiprows=15)

			L = (data[:,6]/lengthNormalisation).reshape(data[:,6].size,1)
			assert (np.any(L>1)==False), "There are too large lengths in your dataset!"
			# normalise X, Y, Z and concatenate with X', Y', Z'
			inputs = np.concatenate((data[:,:3]/positionNormalisation, data[:,3:6]), axis=1)
			
			data_input = tf.convert_to_tensor(inputs)
			data_output = tf.convert_to_tensor(L)

			return data_input, data_output

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

	# convert X, Y, Z from -0.5 - 0.5 to -1 - 1
	# these lines might spit a warning but it still works fine
	train_data[['X','Y','Z']] = train_data[['X','Y','Z']].apply(lambda x: x/0.5)
	validation_data[['X','Y','Z']] = validation_data[['X','Y','Z']].apply(lambda x: x/0.5)

	# convert DataFrame to tf.Tensor
	train_data_input = tf.convert_to_tensor(train_data[['X','Y','Z','Xprime','Yprime','Zprime']].values)
	validation_data_input = tf.convert_to_tensor(validation_data[['X','Y','Z','Xprime','Yprime','Zprime']].values)
	# a normalisation of the output is also happening
	train_data_output = tf.convert_to_tensor(train_data[['L']].values/lengthNormalisation)
	validation_data_output = tf.convert_to_tensor(validation_data[['L']].values/lengthNormalisation)

	return train_data_input, train_data_output, validation_data_input, validation_data_output

def getMLP(fhiddenNodes, inputShape=6, fActivation='relu', fOptimizer='adam', fOutputActivation='sigmoid', fLoss='mse'):
	'''
	Construct a fully connected MLP.
	return
		model: keras model
	'''
	
	# construct
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Dense(fhiddenNodes[0], input_shape=(inputShape,), activation=fActivation))
	for layer in range(1, len(fhiddenNodes)):
		model.add(tf.keras.layers.Dense(fhiddenNodes[layer], activation=fActivation))
	model.add(tf.keras.layers.Dense(1, activation=fOutputActivation))
	# check kernel_initializer='he_uniform'. it doesn't learn.

	# compile
	model.compile(
		loss=fLoss,
		optimizer=fOptimizer,
		metrics=['mae'])
	
	return model

def logConfiguration(dictionary):
	f = open('data/modelConfig_'+timestamp+'.txt','w')
	f.write( str(dictionary) )
	f.close()

def perfPlots(prediction, truth, savename='results'):
	'''
	Produce a set of basic sanity/validation plots
	arguments
		prediction: the predicted np.array from MLP
		truth: the truth np.array
	'''
	print("Hi from Plotter!")

	# create dir
	saveDir = 'plots/'+today+'/'+timestamp
	system('mkdir -p '+saveDir)

	# create subplot env with shared y axis
	fig, axs = plt.subplots(1,3)
	fig.tight_layout(pad=1.3)

	# truth length
	truth_length = truth*lengthNormalisation
	axs[0].hist(truth_length, bins=100)
	axs[0].set(xlabel='Truth L', ylabel='Points')
	
	# predicted length
	pred_length = prediction*lengthNormalisation
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

	# hist2d Truth vs Predicted
	plt.clf()
	plt.hist2d(truth_length.reshape(len(truth_length),), pred_length.reshape(len(pred_length),), bins=(200,200), norm=mpl.colors.LogNorm())
	plt.grid()
	axis = plt.gca()
	axis.set_xlabel('Truth L')
	axis.set_ylabel('Predicted L')
	# save
	plt.tight_layout()
	plt.savefig(saveDir+'/'+savename+'_truthVSpred_'+timestamp+'.pdf')
	print(savename+'_truthVSpred_'+timestamp+".pdf Saved!")

	# hist2d Error vs Truth
	plt.clf()
	plt.hist2d(truth_length.reshape(len(truth_length),), error.reshape(len(error),), bins=(50,50),  norm=mpl.colors.LogNorm())
	axis = plt.gca()
	axis.set_xlabel('Truth L')
	axis.set_ylabel('Truth L - Predicted L')
	# save
	plt.tight_layout()
	plt.savefig(saveDir+'/'+savename+'_truthVSerror_'+timestamp+'.pdf')
	print(savename+'_truthVSerror_'+timestamp+".pdf Saved!")

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
	saveDir = 'plots/'+today+'/'+timestamp
	system('mkdir -p '+saveDir)

	if (Tprediction is not None) and (Ttruth is not None): combined = True
	else: combined = False

	# validation
	val_truth_length = Vtruth*lengthNormalisation
	val_pred_length = Vprediction*lengthNormalisation
	val_error = np.divide(val_truth_length - val_pred_length, val_truth_length, out=np.zeros_like(val_truth_length - val_pred_length), where=val_truth_length!=0)
	val_abs_error = abs(val_error)
	
	if combined:
		test_truth_length = Ttruth*lengthNormalisation
		test_pred_length = Tprediction*lengthNormalisation
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
	saveDir = 'plots/'+today+'/'+timestamp
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
	plt.savefig(saveDir+'/'+savename+'.pdf')
	print(savename+".pdf Saved!")

# enable MLflow autologging XLA
mlflow.keras.autolog()

# define the input arguments
parser = argparse.ArgumentParser(description='Regress distance to the boundary of a unit cube from 3D points and directions.')
parser.add_argument('--trainData', help='Train dataset.', required=False)
parser.add_argument('--validationData', help='Train dataset.', required=True)
parser.add_argument('--testData', help='Test dataset.', required=False)
parser.add_argument('--plots', help='Produce sanity plots.', default=False, action='store_true')
parser.add_argument('--model', help='Use a previously trained and saved MLP model.', default=None)
parser.add_argument('--test', help='Model testing environment. Do not save.', default=False, action='store_true')

if __name__ == '__main__':

	# parse the arguments
	args = parser.parse_args()

	# set normalisations
	lengthNormalisation = 5543
	positionNormalisation = 1600
	
	# define your settings
	settings = {
		'Structure'				:	[1024,1024],
		'Optimizer'				:	Adam(learning_rate=0.005),
		'Activation'			:	'relu',
		'OutputActivation'		:	'sigmoid',
		'Loss'					:	'mse',
		'Batch'					:	64,
		'Epochs'				:	2
		# 'Training Sample Size'	:	int(trainX.shape[0]*(1-validation_split))
	}

	# get some train/validation data
	if args.model is None: trainData = getG4Datasets(args.trainData, batch_size=settings['Batch'], dataAPI=True)
	validationData = getG4Datasets(args.validationData, batch_size=settings['Batch'], shuffle_buffer_size=1, dataAPI=True)

	# create MLP model
	if args.model is None:
		mlp_model = getMLP(fhiddenNodes=settings['Structure'],
						   fActivation=settings['Activation'],
						   fOptimizer=settings['Optimizer'],
						   fOutputActivation=settings['OutputActivation'],
						   fLoss=settings['Loss'])

		# print model summary
		mlp_model.summary()
		# fit model
		history = mlp_model.fit(trainData,
			                    epochs=settings['Epochs'],
			                    validation_data=validationData)
			                    # batch_size=settings['Batch'],
			                    # validation_split=validation_split) # not supported with data API pipeline but you could feed dataset

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
		perfPlots(prediction=pred_valY, truth=valY.numpy(), savename='resultsVal')
		inputPlots(valX.numpy(), savename='inputsVal')
		if args.testData is not None: 
			perfPlots(prediction=pred_testX, truth=testY.numpy(), savename='resultsTest')
			inputPlots(g4X.numpy(), savename='inputsG4')
		
		# combined plots
		if args.testData is not None: combPerfPlots(Vprediction=pred_valY, Vtruth=valY.numpy(), Tprediction=pred_testX, Ttruth=testY.numpy())
		else: combPerfPlots(Vprediction=pred_valY, Vtruth=valY.numpy())		
	
	print("Done!")
