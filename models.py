import tensorflow as tf
import keras.backend as K

from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def noOverestimateLossFunction(y_true, y_pred):
    """! 
    Hopefully this will prevent us from overestimating the boundary to the next volume.
    !"""
    print(y_true)
    negIndex = ((y_true-y_pred) < 0)
    nNegs =K.sum(K.cast(negIndex, dtype='float64'))
    negPunish = 1;
    negLoss = nNegs*negPunish
    posLoss = K.mean(K.square(y_true[~negIndex]-y_pred[~negIndex]))
    totalLoss = posLoss+negLoss
    return totalLoss

    
def regression(inputShape, outputDim, savePath, outputAct, hiddenSpaceSize=[50], activations=['relu']):
    model = keras.Sequential()
    model.add(layers.Dense(hiddenSpaceSize[0], input_shape=inputShape, activation=activations[0]))
    for layerI in range(1, len(hiddenSpaceSize)):
        model.add(layers.Dense(hiddenSpaceSize[layerI], activation=activations[layerI]))

    model.add(layers.Dense(outputDim, activation=outputAct))
    model.compile(loss=noOverestimateLossFunction, optimizer=tf.keras.optimizers.Adam(lr=0.0001), metrics=['mae', 'mse'])
    # Store model to file
    model_json = model.to_json()
    with open(savePath, "w") as json_file:
        json_file.write(model_json)
    model.summary()
    return model


