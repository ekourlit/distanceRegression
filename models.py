import tensorflow as tf
from tensorflow.keras import layers

def getNoOverestimateLossFunction(negPunish=1.0):
    @tf.function
    def noOverestimateLossFunction(y_true, y_pred):
        """
        Hopefully this will prevent us from overestimating the boundary to the next volume.
        """

        negIndex = ((y_true-y_pred) < 0)
        
        # punish on exponent
        # negLoss = tf.reduce_mean(tf.pow(y_true[~negIndex]-y_pred[~negIndex], tf.constant([2*negPunish], dtype='float32')))
        # posLoss = tf.reduce_mean(tf.square(y_true[negIndex]-y_pred[negIndex]))

        # punish by weight
        negLoss = tf.reduce_mean(tf.square(y_true[negIndex]-y_pred[negIndex]))
        posLoss = tf.reduce_mean(tf.square(y_true[~negIndex]-y_pred[~negIndex]))
        totalLoss = negPunish*negLoss+posLoss

        return totalLoss

    return noOverestimateLossFunction

@tf.function
def overestimationMetric(y_true, y_pred):
    """
    This is a metric to estimate the amount of overestimation.
    Along with MAE can be used as an objective to the DH scans.
    """
    negIndex = ((y_true-y_pred) < 0)

    overPred = tf.reduce_max(y_pred[negIndex]-y_true[negIndex])
    # underPred = tf.reduce_mean(y_pred[~negIndex])
    # ratio = overPred/underPred

    return overPred
    
def regression(inputShape, outputDim, savePath, outputAct, hiddenSpaceSize=[50], activations=['relu']):
    model = tf.keras.Sequential()
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

def getSimpleMLP_DH(**config):
    '''
    Construct a fully connected MLP.
    return
        model: keras model
    '''
    
    # construct
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(config['nodes'], input_shape=(6,), activation=config['activation'], kernel_initializer='he_uniform'))
    for layer in range(1, config['num_layers']):
        model.add(tf.keras.layers.Dense(config['nodes'], activation=config['activation'], kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Dense(1, activation=config['output_activation'], kernel_initializer='he_uniform'))

    # compile
    model.compile(
        loss=config['loss'],
        optimizer=config['optimizer'],
        metrics=['mae','mse', overestimationMetric])

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
