#!/usr/bin/env python
import argparse
import pandas as pd
import numpy as np

def flattenOmit(dataset, bins, lengthColumnName='L', outName='flatData.pkl'):
    """!
    Bin the lengths and then remove events so that the length distribution is flat.
    inPath: path to Geant4 macro file.
    bins: bins to use to flatten length distribution.
    """
    lengths = dataset[lengthColumnName]
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

    newDataset.to_pickle(outName)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flatten a distance distribution')
    parser.add_argument('-i', '--inPath', type=str, help="Input file")
    parser.add_argument('-n', '--nbins', type=int, default=50, help="number of bins")
    parser.add_argument('-t', '--inputType', type=str, default='pickle', choices=['pickle', 'csv'])
    parser.add_argument('-l', '--lengthName', type=str, default='L', help='Name of length column in input file.')
    parser.add_argument('-o', '--outName', type=str, default='flatData.pkl', help='Output file name')


    args = parser.parse_args()

    if args.inputType == 'pickle':
        dataset = pd.DataFrame(pd.read_pickle(args.inPath))
    else:
        dataset = pd.DataFrame(pd.read_csv(args.inPath))

    minVal = np.min(dataset[args.lengthName]);
    maxVal = np.max(dataset[args.lengthName]);
    dx = (maxVal-minVal)/args.nbins
    bins = [minVal+dx*i for i in range(args.nbins+1)]
    print('minVal, maxVal, dx, dataset shape')
    print(minVal, maxVal, dx, dataset.shape)
    flattenOmit(dataset, bins, lengthColumnName=args.lengthName, outName=args.outName)
