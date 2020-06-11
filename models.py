import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pdb

def getNoOverestimateLossFunction(negExp=1):
    def noOverestimateLossFunction(y_true, y_pred):
        """! 
        Hopefully this will prevent us from overestimating the boundary to the next volume.
        !"""
        negIndex = ((y_true-y_pred) < 0)
        # nNegs = K.sum(K.cast(negIndex, dtype='float32'))
        # This parameter needs to be tuned. Need to figure out how to make this a function parameter with keras.
        posLoss = tf.reduce_mean(tf.square(y_true[negIndex]-y_pred[negIndex]))
        negLoss = tf.reduce_mean(tf.pow(y_true[~negIndex]-y_pred[~negIndex], tf.constant([2*negExp], dtype='float32')))
        # mse = K.mean(K.square(y_true-y_pred))
        totalLoss = negLoss+posLoss
        # print(totalLoss, negLoss, posLoss, mse)
        # print(nNegs, y_true.shape)
        return totalLoss
    return noOverestimateLossFunction
    
def regression(inputShape, outputDim, savePath, outputAct, hiddenSpaceSize=[50], activations=['relu']):
    model = keras.Sequential()
    model.add(layers.Dense(hiddenSpaceSize[0], input_shape=inputShape, activation=activations[0]))
    for layerI in range(1, len(hiddenSpaceSize)):
        model.add(layers.Dense(hiddenSpaceSize[layerI], activation=activations[layerI]))

    model.add(layers.Dense(outputDim, activation=outputAct))
    model.compile(loss=noOverestimateLossFunction, optimizer=tf.keras.optimizers.Adam(lr=0.0001), metrics=['mse'])
    # Store model to file
    model_json = model.to_json()
    with open(savePath, "w") as json_file:
        json_file.write(model_json)
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
    model.add(tf.keras.layers.Dense(hiddenNodes[0], input_shape=(input_shape,), activation=settings['activation'], kernel_initializer='he_uniform'))
    for layer in range(1, len(hiddenNodes)):
        model.add(tf.keras.layers.Dense(hiddenNodes[layer], activation=settings['activation'], kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Dense(1, activation=settings['output_activation'], kernel_initializer='he_uniform'))

    # compile
    model.compile(
        loss=settings['loss'],
        optimizer=settings['optimizer'],
        metrics=['mae','mse'])

    # print model summary
    model.summary()
    
    return model

def getSplitMLP(hiddenNodes, input_shape=3, **settings):
    '''
    Construct a regression MLP which takes two inputs and concatenates them further on.

              |                 |
    Position  | . . . . . . . . | 
              |                 |
                    Concatenate | . . . | Distance
              |                 |
    Direction | . . . . . . . . |
              |                 |

    arguments:
        {hiddenNodes['Position']    : [possition_hidden_layers],
        hiddenNodes['Direction']    : [direction_hidden_layers],
        hiddenNodes['Concatenated'] : [concatenated_hidden_layers]}
    return
        model: keras model
    '''
    
    # two independent input layers
    input_P = tf.keras.layers.Input(shape=(input_shape,), name='position')
    input_D = tf.keras.layers.Input(shape=(input_shape,), name='direction')

    # two independent hidden layers paths
    hidden_P = tf.keras.layers.Flatten(name='dummy_position_path')(input_P)
    for layer in range(0, len(hiddenNodes['Position'])):
        hidden_P = tf.keras.layers.Dense(hiddenNodes['Position'][layer], activation=settings['activation'])(hidden_P)
    # 
    hidden_D = tf.keras.layers.Flatten(name='dummy_direction_path')(input_D)
    for layer in range(0, len(hiddenNodes['Direction'])):
        hidden_D = tf.keras.layers.Dense(hiddenNodes['Direction'][layer], activation=settings['activation'])(hidden_D)

    # concatenate the two paths
    hidden_C = tf.keras.layers.concatenate([hidden_P, hidden_D])
    
    # final common hidden layers
    for layer in range(0, len(hiddenNodes['Concatenated'])):
        hidden_C = tf.keras.layers.Dense(hiddenNodes['Concatenated'][layer], activation=settings['activation'])(hidden_C)

    # output layer
    output = tf.keras.layers.Dense(1, activation=settings['output_activation'], name='distance')(hidden_C)

    # construct the model
    model = tf.keras.Model(inputs=[input_P, input_D], outputs=[output])

    # compile
    model.compile(
        loss=settings['loss'],
        optimizer=settings['optimizer'],
        metrics=['mse','mae'])

    # print model summary
    model.summary()
    
    return model
