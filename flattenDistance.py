#!/usr/bin/env python
import argparse, matplotlib
import pandas as pd
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
font = {'size':16}
matplotlib.rc('font', **font)


def flattenOmit(dataset, bins, lengthColumnName='L', outName='flatData.pkl'):
    """!
    Bin the lengths and then remove events so that the length distribution is flat.
    inPath: path to Geant4 macro file.
    bins: bins to use to flatten length distribution.
    """
    lengths = dataset[lengthColumnName]
    fig, ax = plt.subplots()

    plt.clf()
    plt.hist(lengths, histtype='step', bins=bins)
    ax.set_xlabel('Lengths')
    plt.savefig('LengthsPre.pdf', bbox_inches='tight')
        
    hist, binEdges = np.histogram(lengths, bins=bins)
    binIndeces = np.digitize(lengths, bins=bins)
    minBinContent = np.min(hist)
    totalEvents = minBinContent*(len(bins)-1)
    print(f'You have chosen a binning that will result in each bin containing:\n{minBinContent} events.\nWhich will result in a total of\n{totalEvents} events.')

    newDataset = pd.DataFrame(columns=dataset.columns)
    for binI in range(1, len(bins)):
        selectedBins = binIndeces==binI
        binData = dataset.iloc[selectedBins]
        dataInBin = binData.sample(n=minBinContent, random_state=1)
        newDataset = newDataset.append(dataInBin, ignore_index=True)

    plt.clf()
    plt.hist(newDataset[lengthColumnName], histtype='step', bins=bins)
    ax.set_xlabel('Lengths')
    plt.savefig('LengthsPost.pdf', bbox_inches='tight')
        
    newDataset.to_pickle(outName)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flatten a distance distribution')
    parser.add_argument('-i', '--inPath', type=str, help="Input file")
    parser.add_argument('-n', '--nbins', type=int, default=50, help="number of bins")
    parser.add_argument('-t', '--inputType', type=str, default='pickle', choices=['pickle', 'csv'])
    parser.add_argument('-l', '--lengthName', type=str, default='L', help='Name of length column in input file.')
    parser.add_argument('-o', '--outName', type=str, default='flatData.pkl', help='Output file name')
    parser.add_argument('--maxVal', type=float, default=1., help="Maximum length value")
    parser.add_argument('--minVal', type=float, default=0., help="Minimum length value")


    args = parser.parse_args()

    if args.inputType == 'pickle':
        dataset = pd.DataFrame(pd.read_pickle(args.inPath))
    else:
        dataset = pd.DataFrame(pd.read_csv(args.inPath))
    
    dx = (args.maxVal-args.minVal)/args.nbins
    bins = [args.minVal+dx*i for i in range(args.nbins+1)]
    print('args.minVal, args.maxVal, dx, dataset shape')
    print(args.minVal, args.maxVal, dx, dataset.shape)
    flattenOmit(dataset, bins, lengthColumnName=args.lengthName, outName=args.outName)
