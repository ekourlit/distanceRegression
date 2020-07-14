import tensorflow as tf
import numpy as np
import h5py, pandas
import config
import glob
import os

@tf.function
def process_csv_line(line):
    '''
    Process each csv line of the dataset.
    Data preprocessing can be done nere.
    '''
    # parse csv line
    fields = tf.io.decode_csv(line, [tf.constant([np.nan], dtype=tf.float32)] * 7)
    # first 3 fields are X,Y,Z. normalize them
    position = tf.stack(fields[:3])/config.positionNormalisation
    # next 3 fields are Xprime,Yprime,Zprime
    momentum = tf.stack(fields[3:-1])
    # stack and flatten them
    features = tf.reshape(tf.stack([position,momentum]), [-1])
    # last field is the length. normalize it.
    length = tf.stack(fields[-1:])/config.lengthNormalisation
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

def getG4Arrays(G4FilePath, split_input=False):
    '''
    Construct the test datasets generated from Geant4.
    arguments
        G4FilePath: file path
    return
        data_input, data_output: 1 x tf.Tensor. Input shape (i,6), output shape (i,1).
    '''
    if '.hdf5' in G4FilePath:
        file = h5py.File(G4FilePath,'r')
        
        X = np.array(file['default_ntuples']['B4']['x']['pages'])/config.positionNormalisation
        Y = np.array(file['default_ntuples']['B4']['y']['pages'])/config.positionNormalisation
        Z = np.array(file['default_ntuples']['B4']['z']['pages'])/config.positionNormalisation
        Xprime = np.array(file['default_ntuples']['B4']['dx']['pages'])
        Yprime = np.array(file['default_ntuples']['B4']['dy']['pages'])
        Zprime = np.array(file['default_ntuples']['B4']['dz']['pages'])
        L = np.array(file['default_ntuples']['B4']['distance']['pages'])
        assert (np.any(L>config.lengthNormalisation)==False), "There are too large lengths in your dataset!"

        data_input = tf.convert_to_tensor(np.column_stack((X,Y,Z,Xprime,Yprime,Zprime)))
        # a normalisation of the output is also happening
        data_output = tf.convert_to_tensor(np.column_stack(L/config.lengthNormalisation).T)

        return data_input, data_output

    # csv input
    else:
        # path or wildcarded files
        if not os.path.isfile(G4FilePath):
            arrays = [np.loadtxt(file, delimiter=',', skiprows=12) for file in glob.glob(G4FilePath)]
            data = np.concatenate(arrays)

        # single file
        else:
            data = np.loadtxt(G4FilePath, delimiter=',', skiprows=12)

        L = (data[:,6]/config.lengthNormalisation).reshape(data[:,6].size, 1)
        if config.lengthNormalisation != 1: assert (np.any(L>1)==False), "There are too large lengths in your dataset!"

        # normalise X, Y, Z 
        positions = data[:,:3]/config.positionNormalisation
        # X', Y', Z'
        directions = data[:,3:6]

        # prepare output
        if split_input == True:
            data_input = {'position' : positions, 'direction' : directions}
        else:
            data_input = np.concatenate((positions, directions), axis=1)
        
        data_output = L

        return data_input, data_output

def getDatasets(data_input, data_output, batch_size=2048, shuffle_buffer_size=1000):
    # create and return TF dataset
    dataset = tf.data.Dataset.from_tensor_slices((data_input,data_output))
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.batch(batch_size)
    return dataset

def getCustomArrays(pickleFile):
    '''
    Construct the training and validation datasets.
    arguments
        pickleFile: the data file
    return
        data_input, data_output. Input shape (0,6), output shape (0,1).
    '''

    # load data
    dataset = pd.DataFrame(pd.read_pickle(pickleFile))
    # split data
    train_data = dataset

    # potentially normalize X, Y, Z data
    # these lines might spit a warning but it still works fine
    train_data[['X','Y','Z']] = train_data[['X','Y','Z']].apply(lambda x: x/1.0)

    # convert DataFrame to tf.Tensor
    train_data_input = tf.convert_to_tensor(train_data[['X','Y','Z','Xprime','Yprime','Zprime']].values)
    # a normalisation of the output might also happening
    train_data_output = tf.convert_to_tensor(train_data[['L']].values/config.lengthNormalisation)

    return train_data_input, train_data_output