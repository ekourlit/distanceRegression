import argparse
import math
import numpy as np
import pickle
from shapely.geometry import Point, LineString, Polygon, box

def extendedLineString(points, toX=10):
	'''
	Get a pair of point and return an arbitary extended line.
	arguments
		points: np.array, shape = (2, 2)
		toX:	int toX
	return
		longLine: shapely.LineString
	'''
	
	# alpha = y2-y1/x2-x1
	alpha = (points[1,:][1] - points[0,:][1]) / (points[1,:][0] - points[0,:][0])
	# beta = y1 - alpha * x1
	beta = points[0,:][1] - alpha * points[0,:][0]

	longLine = LineString([Point(toX, alpha*toX+beta), Point(-toX, -alpha*toX+beta)])
	return(longLine)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Create a dataset of 2D points, directions and distance to the boundary of a unit box.')
	parser.add_argument('--sampleSize', type=int, help='Number of data points.', default=1000)
	parser.add_argument('--savePickle', help='Save the dataset in a pickle file.', default=False, action='store_true')
	parser.add_argument('--fileName', help='Pickle file name.', default="2Ddata")
	args = parser.parse_args()

	# boundary of box from 0 to 1 in both x & y axes
	boundary = LineString([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)])

	# output dictionary
	if args.savePickle: dataset = {'i':[], 'X':[], 'Y':[], 'Xprime':[], 'Yprime':[], 'L':[]}

	for i in range(args.sampleSize):

		# pair of random points in [0,1)
		A = np.random.rand(2,2) # A[0,:] is np.array([x1,y1]), A[1,:] is np.array([x2,y2])
		line = extendedLineString(A)

		# a collection of two points where the line intercepts the boundary
		intercepts = line.intersection(boundary)
		assert len(intercepts) == 2, "More than two intersection points found! Something wrong?"

		# vector from points A (x2-x1, y2-y1)
		v = np.array([A[1,:][0] - A[0,:][0], A[1,:][1] - A[0,:][1]])
		# vector from A to I1
		AI1 = np.array([intercepts.geoms[0].x - A[0,:][0], intercepts.geoms[0].y - A[0,:][1]])
		# vector from A to I2
		AI2 = np.array([intercepts.geoms[1].x - A[0,:][0], intercepts.geoms[1].y - A[0,:][1]])

		# index of I point in the direction of vector v
		Iindex = 0 if v.dot(AI1) > 0 else 1
		lineSegment = LineString([Point(A[0,:][0], A[0,:][1]), Point(intercepts.geoms[Iindex].x, intercepts.geoms[Iindex].y)])

		# legth from A to I in the direction of vector v
		lengthToBoundary = lineSegment.length
		assert lengthToBoundary < math.sqrt(2), "The length is longer than physically allowed!"

		# print some data
		if (i % (args.sampleSize/10) == 0): print("i: %i X: %f, Y: %f, X\': %f, Y\': %f, Length: %f" % (i, A[0,:][0], A[0,:][1], A[1,:][0], A[1,:][1], lengthToBoundary))

		if args.savePickle:
			# save to dataset
			dataset['i'].append(i)
			dataset['X'].append(A[0,:][0])
			dataset['Y'].append(A[0,:][1])
			dataset['Xprime'].append(A[1,:][0])
			dataset['Yprime'].append(A[1,:][1])
			dataset['L'].append(lengthToBoundary)

	if args.savePickle: 
		with open(args.fileName+'.pickle', 'wb') as f: pickle.dump(dataset, f)