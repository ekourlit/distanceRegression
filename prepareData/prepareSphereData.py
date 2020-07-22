import argparse
import math
import numpy as np
import pickle
from geo import *
import pdb
from tqdm import tqdm
import h5py
from os import system

def getPlaneNormal(fPlane):
	'''
	Get two points on the plane normal and calculate the vector
	'''
	norm_p1 = np.array(list(fPlane.normal().points()[0].coordinates()))
	norm_p2 = np.array(list(fPlane.normal().points()[1].coordinates()))
	norm_vector = norm_p2 - norm_p1
	
	return norm_vector

def getPlanePoint(fPlane):
	'''
	Return a point on the plane
	'''
	return np.array(list(fPlane.points()[0].coordinates()))

def getLineDirection(fLine):
	'''
	Get the two points of the line and calculate the vector. Direction from P[0,:] to P[1,:]
	'''
	line_point1 = np.array(list(fLine.points()[0].coordinates()))
	line_point2 = np.array(list(fLine.points()[1].coordinates()))
	lineDirection = line_point2 - line_point1
	
	return lineDirection

def getLinePoint(fLine):
	'''
	Return a point on the Line
	'''
	return np.array(list(fLine.points()[0].coordinates()))

def getLinePlaneIntersection(fLine, fPlane, epsilon=1e-10):
	'''
	Calculate the intersection of a line with a plane (if there is any).
	Algorithm from https://rosettacode.org/wiki/Find_the_intersection_of_a_line_with_a_plane#Python
	'''
	# Plane normal vector
	n = getPlaneNormal(fPlane)
	# Point on plane
	plane_point = getPlanePoint(fPlane)
	# Line direction vector
	u = getLineDirection(fLine)
	# Point on line
	line_point = getLinePoint(fLine)

	ndotu = n.dot(u)
	# there is not an intersection (you can also check if any point of L is contained in P, if yes, discard this datum)
	if abs(ndotu) < epsilon: return None
	# otherwise there is an intersection
	w = line_point - plane_point
	si = - n.dot(w) / ndotu
	if si >= 0 : 
		# Intersection point
		intersex = w + si * u + plane_point	
		# the intersection should be within [shift, 1+shift)
		if not (np.all(intersex>=shift) and np.all(intersex<=1+shift)): return None
		return intersex

def saveData(index, points, direction, length=1.0):
	if not args.HDF5:
		# save to dataset dictionary for pickling
		dataset['i'].append(index)
		dataset['X'].append(points[0])
		dataset['Y'].append(points[1])
		dataset['Z'].append(points[2])
		dataset['Xprime'].append(direction[0])
		dataset['Yprime'].append(direction[1])
		dataset['Zprime'].append(direction[2])
		dataset['L'].append(length)
	else:
		group = hdf.create_group('point'+str(index))
		group.create_dataset('position', data=points[0,:], compression='gzip')
		group.create_dataset('direction', data=direction, compression='gzip')
		group.create_dataset('length', data=np.array([length]), compression='gzip')

parser = argparse.ArgumentParser(description='Create a dataset of a pair of 3D points (position and direction unit vecor) and the distance to the boundary of a unit sphere.')
parser.add_argument('--sampleSize', type=int, help='Number of data points.', default=1000)
parser.add_argument('--saveFile', help='Save the dataset in a pickle file.', default=False, action='store_true')
parser.add_argument('--HDF5', help='Save the dataset in a HDF5 file.', default=False, action='store_true')
parser.add_argument('--filename', help='Pickle file name.', default="sphereData")

if __name__ == '__main__':

	# parse the arguments
	args = parser.parse_args()

	# create data dir
	system('mkdir -p data')

	# outputs
	if not args.HDF5: dataset = {'i':[], 'X':[], 'Y':[], 'Z':[], 'Xprime':[], 'Yprime':[], 'Zprime':[], 'L':[]}
	else: hdf = h5py.File('data/'+args.filename+'.h5', 'w')

	# probably an infinite loop here and count the number of times you call saveData
	i = 0
	while True:
		if i >= args.sampleSize: break

		# sample uniformly a 3D point in [-1, 1) range
		P = Point(np.random.uniform(-1, 1, 3)) # P is np.array([x1,y1,z1])
		P0 = Point(np.zeros(3))

		# the point P should be inside the unit sphere
		distanceToOrigin = P.distance_to(P0)
		if distanceToOrigin > 1.0: continue
		i += 1
		if i%(args.sampleSize/100) == 0: print("Point number %i" % i)

		line = Line(P0, P)
		# line direction vector and its unit vector
		u = getLineDirection(line)
		u_hat = u/np.linalg.norm(u)

		# unit sphere lengthToBoundary calculations
		lengthToBoundary = 1.0 - distanceToOrigin
		assert lengthToBoundary < 1, "The length is longer than physically allowed!"

		# save to dataset
		saveData(i, np.array(list(P.coordinates())), u_hat, lengthToBoundary)

	# write out
	if args.saveFile:
		if not args.HDF5: 
			with open('data/'+args.filename+'.pickle', 'wb') as f: pickle.dump(dataset, f)
			print("\nFile %s saved!" % (args.filename+'.pickle'))
		else:
			hdf.close()
			print("\nFile %s saved!" % (args.filename+'.h5'))

