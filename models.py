import tensorflow as tf
import keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers


def regression(inputShape, outputDim, savePath, outputAct, hiddenSpaceSize=[50], activations=['relu']):
    model = keras.Sequential()
    model.add(layers.Dense(hiddenSpaceSize[0], input_shape=inputShape, activation=activations[0]))
    for layerI in range(1, len(hiddenSpaceSize)):
        model.add(layers.Dense(hiddenSpaceSize[layerI], activation=activations[layerI]))

    model.add(layers.Dense(outputDim, activation=outputAct))
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.0001), metrics=['mae', 'mse'])
    # Store model to file
    model_json = model.to_json()
    with open(savePath, "w") as json_file:
        json_file.write(model_json)
    model.summary()
    return model


def noOverestimateLossFunction(y_true, y_pred):
    """! 
    Hopefully this will prevent us from overestimating the boundary to the next volume.
    !"""
    negIndex = ((y_true-y_pred) < 0)
    nNegs = K.sum(K.cast(negIndex, dtype='float32'))
    # This parameter needs to be tuned. Need to figure out how to make this a function parameter with keras.
    negPunish = 20.;
    negLoss = K.mean(K.square(y_true[negIndex]-y_pred[negIndex]))
    posLoss = K.mean(K.square(y_true[~negIndex]-y_pred[~negIndex]))
    mse = K.mean(K.square(y_true-y_pred))
    totalLoss = negPunish*negLoss+posLoss
    print(totalLoss, negLoss, posLoss, mse)
    print(nNegs, y_true.shape)
    return totalLoss

def getBiasedMLP(hiddenNodes, input_shape=6, **settings):
    '''
    Construct a fully connected MLP.
    return
        model: keras model
    '''
    
    # construct
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(hiddenNodes[0], input_shape=(input_shape,), activation=settings['activation']))
    for layer in range(1, len(hiddenNodes)):
        model.add(tf.keras.layers.Dense(hiddenNodes[layer], activation=settings['activation']))
    model.add(tf.keras.layers.Dense(1, activation=settings['output_activation']))
    # check kernel_initializer='he_uniform'. it doesn't learn.

    # compile
    model.compile(
        loss=noOverestimateLossFunction,
        optimizer=settings['optimizer'],
        metrics=['mae','mse'])

    # print model summary
    model.summary()
    
    return model
 
def getSimpleMLP(hiddenNodes, input_shape=6, **settings):
    '''
    Construct a fully connected MLP.
    return
        model: keras model
    '''
    
    # construct
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(hiddenNodes[0], input_shape=(input_shape,), activation=settings['activation']))
    for layer in range(1, len(hiddenNodes)):
        model.add(tf.keras.layers.Dense(hiddenNodes[layer], activation=settings['activation']))
    model.add(tf.keras.layers.Dense(1, activation=settings['output_activation']))
    # check kernel_initializer='he_uniform'. it doesn't learn.

    # compile
    model.compile(
        loss=settings['loss'],
        optimizer=settings['optimizer'],
        metrics=['mae'])

    # print model summary
    model.summary()
    
    return model