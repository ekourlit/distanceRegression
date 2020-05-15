import argparse, pickle, matplotlib
import numpy as np
from models import *
from tensorflow.keras.models import model_from_json

matplotlib.use('Agg')
import matplotlib.pyplot as plt
seed=1984
np.random.seed(seed)
import tensorflow as tf
tf.random.set_seed(seed)

def edgeExponential(x, offset=0.0, beta=1.):
    return np.exp(beta*(x+offset))


def produceToyData(minX, maxX, nPoints, enhanceEdges=False):
    """! 
    Produce data in one dimension with a minimum and a maximum. 
    The target data will be the distance of a randomly generated point to the max. 
    """

    # If we weren't given a pdf, simply produced evenly spaced data between the min ans max
    points = np.random.uniform(minX, maxX, nPoints)
    directions = np.random.choice([-1,1], nPoints)
    if enhanceEdges:
        nEdgePoints = 5*nPoints
        beta = 100.
        tempPoints = np.random.uniform(minX, minX+0.05*(maxX-minX), nEdgePoints)
        lowEdge = edgeExponential(tempPoints, offset=minX, beta=-1.*beta)
        choicesLowEdge = np.random.choice(tempPoints, size=tempPoints.size, p=lowEdge/np.sum(lowEdge))

        points = np.append(points, choicesLowEdge)
        #directions = np.append(directions, np.random.choice([-1,1], nEdgePoints))
        directions = np.append(directions, -1.*np.ones(choicesLowEdge.shape))

        plt.clf()
        fig, ax = plt.subplots()
        ax.plot(tempPoints,lowEdge, '.')
        ax.set_xlabel('x position')
        ax.set_ylabel('lowEdge')
        plt.savefig('lowEdgeTest.pdf', bbox_inches='tight')

        plt.clf()
        fig, ax = plt.subplots()
        plt.hist(choicesLowEdge, bins=20)
        ax.set_xlabel('x position')
        ax.set_ylabel('N')
        plt.savefig('lowChoiceTest.pdf', bbox_inches='tight')


        tempPoints = np.random.uniform(.95*(maxX-minX), maxX, nEdgePoints)
        highEdge = edgeExponential(tempPoints, offset=maxX, beta=beta)
        choicesHighEdge = np.random.choice(tempPoints, size=tempPoints.size, p=highEdge/np.sum(highEdge))

        points = np.append(points, choicesHighEdge)
        #directions = np.append(directions, np.random.choice([-1,1], nEdgePoints))
        directions = np.append(directions, np.ones(choicesHighEdge.shape))
        plt.clf()
        fig, ax = plt.subplots()
        ax.plot(tempPoints,highEdge, '.')
        ax.set_xlabel('x position')
        ax.set_ylabel('highEdge')
        plt.savefig('highEdgeTest.pdf', bbox_inches='tight')

        plt.clf()
        fig, ax = plt.subplots()
        plt.hist(choicesHighEdge, bins=20)
        ax.set_xlabel('x position')
        ax.set_ylabel('N')
        plt.savefig('highChoiceTest.pdf', bbox_inches='tight')
    
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
    history = model.fit(x, y, epochs=epochs, batch_size=batchSize, validation_split=validationSplit, verbose=1, shuffle=True)

    model.save_weights(weightsSavePath)
    pickle.dump(history.history, open(historyPath, 'wb'))

    
def analyze(inputs, targets, modelFName='models/kerasReg.json', weightsSavePath='models/weightsReg.h5', historyPath='history.pkl'):
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
    epochs = np.arange(len(history['loss']))
    fig, ax = plt.subplots()
    ax.plot(epochs, history['loss'], label='Train')
    ax.plot(epochs, history['val_loss'], label='Val')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.legend(loc="best", borderaxespad=0.)
    plt.savefig('loss.pdf', bbox_inches='tight')    

    plt.clf
    delta = targets-np.squeeze(preds)
    fig, ax = plt.subplots()
    ax.plot(points,delta, '.')
    ax.set_xlabel('x position')
    ax.set_ylabel('$d_{\\mathrm{true}}-d_{\\mathrm{pred}}$')
    plt.savefig('err.pdf', bbox_inches='tight')

    plt.clf
    fig, ax = plt.subplots()
    ax.hist(delta, histtype='step', bins=100)
    ax.set_xlabel('$d_{\\mathrm{true}}-d_{\\mathrm{pred}}$')
    ax.set_ylabel('Occurances')
    plt.savefig('errHist.pdf', bbox_inches='tight')
    ax.set_yscale('log')
    plt.savefig('errHist_log.pdf', bbox_inches='tight')

    plt.close('all');

    noOverestimateLossFunction(targets, np.squeeze(preds))
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train a NN for regression.')
    parser.add_argument('trainAnalyze', help="Perform training, analyze, or both.")
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=1000)
    parser.add_argument('--batchSize', type=int, help='Batch sizes', default=100)
    parser.add_argument('--nTrainingPoints', type=int, help='Number of points to use for training', default=10000)
    parser.add_argument('--trainTestRatio', type=float, help='Ratio of training to testing sample', default=10)
    args = parser.parse_args()

    inputs, targets = produceToyData(0, 1, args.nTrainingPoints, enhanceEdges=False)
    if 'train' in args.trainAnalyze:
        train(inputs, targets, epochs=args.epochs, batchSize=args.batchSize)
        
    if 'analyze' in args.trainAnalyze:
        inputs, targets = produceToyData(0, 1, int(args.nTrainingPoints*args.trainTestRatio))
        analyze(inputs, targets)
