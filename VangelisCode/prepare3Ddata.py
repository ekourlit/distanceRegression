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
		dataset['X'].append(points[0,0])
		dataset['Y'].append(points[0,1])
		dataset['Z'].append(points[0,2])
		dataset['Xprime'].append(direction[0])
		dataset['Yprime'].append(direction[1])
		dataset['Zprime'].append(direction[2])
		dataset['L'].append(length)
	else:
		group = hdf.create_group('point'+str(index))
		group.create_dataset('position', data=points[0,:], compression='gzip')
		group.create_dataset('direction', data=direction, compression='gzip')
		group.create_dataset('length', data=np.array([length]), compression='gzip')

parser = argparse.ArgumentParser(description='Create a dataset of a pair of 3D points (position and direction unit vecor) and the distance to the boundary of a unit cube.')
parser.add_argument('--sampleSize', type=int, help='Number of data points.', default=1000)
parser.add_argument('--saveFile', help='Save the dataset in a pickle file.', default=False, action='store_true')
parser.add_argument('--HDF5', help='Save the dataset in a HDF5 file.', default=False, action='store_true')
parser.add_argument('--filename', help='Pickle file name.', default="3Ddata")
parser.add_argument('--custom', help='Not a unit cube but custom range.', default=False, action='store_true')

if __name__ == '__main__':

	# parse the arguments
	args = parser.parse_args()

	# create data dir
	system('mkdir -p data')

	# outputs
	if not args.HDF5: dataset = {'i':[], 'X':[], 'Y':[], 'Z':[], 'Xprime':[], 'Yprime':[], 'Zprime':[], 'L':[]}
	else: hdf = h5py.File('data/'+args.filename+'.h5', 'w')

	shift = -0.5
	# Unit centered at (0.5+shift,0.5+shift,0.5+shift).
	boundaries = {
		'base' 	: Plane(Point(0+shift,0+shift,0+shift), Point(1+shift,0+shift,0+shift), Point(0+shift,1+shift,0+shift)),
		'top' 	: Plane(Point(0+shift,0+shift,1+shift), Point(1+shift,0+shift,1+shift), Point(0+shift,1+shift,1+shift)),
		'face' 	: Plane(Point(1+shift,0+shift,0+shift), Point(1+shift,1+shift,0+shift), Point(1+shift,0+shift,1+shift)),
		'back' 	: Plane(Point(0+shift,0+shift,0+shift), Point(0+shift,1+shift,0+shift), Point(0+shift,0+shift,1+shift)),
		'east' 	: Plane(Point(0+shift,1+shift,0+shift), Point(1+shift,1+shift,0+shift), Point(1+shift,1+shift,1+shift)),
		'west' 	: Plane(Point(0+shift,0+shift,0+shift), Point(1+shift,0+shift,0+shift), Point(0+shift,0+shift,1+shift))
	}

	for i in tqdm(range(args.sampleSize)):
		
		if not args.custom:
			# sample uniformly a pair of 3D points in [shift, 1+shift)
			points = np.random.uniform(shift, 1+shift, 6)
			points.shape = (2,3)
			P = points # P[0,:] is np.array([x1,y1,z1]), P[1,:] is np.array([x2,y2,z2])
		
		else:
			# or sample a pair of custom 3D points
			x = np.random.uniform(-2070., 2070., 2)
			y = np.random.uniform(-2070., 2070., 2)
			z = np.random.uniform(-3050.5, -2420.5, 2)
			P = np.column_stack((x,y,z)) # P[0,:] is np.array([x1,y1,z1]), P[1,:] is np.array([x2,y2,z2])

		line = Line(Point(P[0,:]), Point(P[1,:]))
		# line direction vector and its unit vector
		u = getLineDirection(line)
		u_hat = u/np.linalg.norm(u)


		if not args.custom:
			# unit cube lengthToBoundary calculations

			count = 0
			for plane in boundaries:
				# Line - Plane Intersection point
				I = getLinePlaneIntersection(line, boundaries[plane], epsilon=1e-50)
				if I is not None:
					count =+ 1
					lengthToBoundary = Point(P[0,:]).distance_to(Point(I))
					assert lengthToBoundary < math.sqrt(3), "The length is longer than physically allowed!"
					lengthToBoundaryNorm = lengthToBoundary/math.sqrt(3)
			assert count < 2, "More than one intersection found! Unphysical!"
			
			# In some cased I dont' find intersection. I don't exactly know why but I can just drop these points
			if count == 1: saveData(i, P, u_hat, lengthToBoundary)

		# just save the points			
		else: saveData(i, P, u_hat)

	# write out
	if args.saveFile:
		if not args.HDF5: 
			with open('data/'+args.filename+'.pickle', 'wb') as f: pickle.dump(dataset, f)
			print("\nFile %s saved!" % (args.filename+'.pickle'))
		else:
			hdf.close()
			print("\nFile %s saved!" % (args.filename+'.h5'))

	# plot
	# from matplotlib import pyplot as plt
	# plt.hist(np.array(dataset['L']), bins=100, range=(0,2))
	# plt.savefig('distance.pdf')
