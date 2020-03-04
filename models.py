import tensorflow as tf

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


