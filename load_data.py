import numpy as np
import tensorflow as tf

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

def getG4Datasets(G4FilePath, lengthNormalisation=1, positionNormalisation=1):
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

		L = (data[:,6]/lengthNormalisation).reshape(data[:,6].size,1)
		if lengthNormalisation != 1: assert (np.any(L>1)==False), "There are too large lengths in your dataset!"
		# normalise X, Y, Z and concatenate with X', Y', Z'
		inputs = np.concatenate((data[:,:3]/positionNormalisation, data[:,3:6]), axis=1)
		
		data_input = tf.convert_to_tensor(inputs)
		data_output = tf.convert_to_tensor(L)

		return data_input, data_output

def getG4SplitInputDatasets(G4FilePath, lengthNormalisation=1, positionNormalisation=1):
	'''
	Construct the test datasets generated from Geant4.
	arguments
		G4FilePath: file path
	return
		data_input, data_output: 1 x tf.Tensor. Input shape (i,6), output shape (i,1).
	'''

	# to-do: I want to feed a wildcarded path here
	data = np.loadtxt(G4FilePath, delimiter=',', skiprows=15)

	L = (data[:,6]/lengthNormalisation).reshape(data[:,6].size, 1)
	if lengthNormalisation != 1: assert (np.any(L>1)==False), "There are too large lengths in your dataset!"
	# normalise X, Y, Z 
	positions = data[:,:3]/positionNormalisation
	# X', Y', Z'
	directions = data[:,3:6]

	data_input = {'position' : tf.convert_to_tensor(positions), 'direction' : tf.convert_to_tensor(directions)}
	data_output = tf.convert_to_tensor(L)

	return data_input, data_output

def load_data(trainDataFPath='/projects/atlas_aesp/whopkins/GeantData/geom_type_1_nsensors_2_dense_1/train_merged.csv', valDataFPath='/projects/atlas_aesp/whopkins/GeantData/geom_type_1_nsensors_2_dense_1/validation_6.csv'):
	"""! 
	Load some data!
	"""
	lengthNormalisation = 1
	positionNormalisation = 1600
	
	trainX, trainY = getG4Datasets(trainDataFPath, lengthNormalisation=lengthNormalisation, positionNormalisation=positionNormalisation)
	valX, valY = getG4Datasets(valDataFPath, lengthNormalisation=lengthNormalisation, positionNormalisation=positionNormalisation)

	return (trainX, trainY), (valX, valY)


if __name__ == '__main__':
	load_data()
