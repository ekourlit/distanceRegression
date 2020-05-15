import argparse, pickle, matplotlib
import numpy as np
from models import *
from tensorflow.keras.models import model_from_json

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def produceToyData(minX, maxX, minY, maxY, nPoints):
    """! 
    Produce data in two dimension. 
    The target data will be the distance of a randomly generated point to the max. 
    """

    # If we weren't given a pdf, simply produced evenly spaced data between the min ans max
    xPoints = np.random.uniform(minX, maxX, nPoints)
    yPoints = np.random.uniform(minY, maxY, nPoints)
    
    xDirection = np.random.uniform(minX, maxX, nPoints)
    yDirection = np.random.uniform(minY, maxY, nPoints)

    targets = np.zeros(points.shape)
    for targetI in range(points.shape[0]):
        if directions[targetI] > 0:
            targets[targetI] = maxX-points[targetI]
        else:
            targets[targetI] = points[targetI]-minX

    inputs = np.column_stack((points,directions))
    s = np.arange(inputs.shape[0])
    np.random.shuffle(s)
    inputs = inputs[s]
    targets = targets[s]
    return (inputs, targets)


def train(x, y, epochs=1000, batchSize=100, modelFName='models/kerasReg.json', weightsSavePath='models/weightsReg.h5', historyPath='history.pkl', validationSplit=0.2):
    """! 
    Train a regression network.
    """
    model = regression([len(x.shape)], len(y.shape), modelFName, 'linear')
    history = model.fit(x, y, epochs=epochs, batch_size=batchSize, validation_split=validationSplit, verbose=0, shuffle=True)

    model.save_weights(weightsSavePath)
    pickle.dump(history.history, open(historyPath, 'wb'))

    
def analyze(inputs, targes, modelFName='models/kerasReg.json', weightsSavePath='models/weightsReg.h5', historyPath='history.pkl'):
    """! 
    Let's analyze the result of the training.
    """
    json_file        = open(modelFName, 'r')
    model_json       = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    # Load weights into new model
    model.load_weights(weightsSavePath)

    preds = model.predict(inputs)
    points = inputs[:,0]
    history = pickle.load(open(historyPath, 'rb'))
   
    plt.clf()
    epochs = np.arange(len(history['mse']))
    fig, ax = plt.subplots()
    ax.plot(epochs, history['mse'], label='Train')
    ax.plot(epochs, history['val_mse'], label='Val')
    ax.set_xlabel('MSE')
    ax.set_ylabel('epoch')
    ax.legend(loc="best", borderaxespad=0.)
    plt.savefig('mse.pdf', bbox_inches='tight')    

    plt.clf()
    epochs = np.arange(len(history['mae']))
    fig, ax = plt.subplots()
    ax.plot(epochs, history['mae'], label='Train')
    ax.plot(epochs, history['val_mse'], label='Val')
    ax.set_xlabel('MAE')
    ax.set_ylabel('epoch')
    ax.legend(loc="best", borderaxespad=0.)
    plt.savefig('mae.pdf', bbox_inches='tight')    

    plt.clf
    delta = np.squeeze(preds)-targets
    fig, ax = plt.subplots()
    ax.plot(points,delta, '.')
    ax.set_xlabel('x position')
    ax.set_ylabel('$d_{\\mathrm{pred}}-d_{\\mathrm{true}}$')
    plt.savefig('predVsTrue.pdf', bbox_inches='tight')

    plt.close('all');

   
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train a NN for regression.')
    parser.add_argument('trainAnalyze', help="Perform training, analyze, or both.")
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=10000)
    parser.add_argument('--batchSize', type=int, help='Batch sizes', default=100)
    parser.add_argument('--nTrainingPoints', type=int, help='Number of points to use for training', default=100)
    parser.add_argument('--trainTestRatio', type=float, help='Ratio of training to testing sample', default=10)
    args = parser.parse_args()

    inputs, targets = produceToyData(0, 1, args.nTrainingPoints)
    if 'train' in args.trainAnalyze:
        train(inputs, targets, epochs=args.epochs, batchSize=args.batchSize)
        
    if 'analyze' in args.trainAnalyze:
        inputs, targets = produceToyData(0, 1, int(args.nTrainingPoints*args.trainTestRatio))
        analyze(inputs, targets)
