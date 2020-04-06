import argparse
import math
import numpy as np
import pickle
from geo import *
import pdb
from tqdm import tqdm
import h5py

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

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Create a dataset of a pair of 3D points (position and direction unit vecor) and the distance to the boundary of a unit cube.')
	parser.add_argument('--sampleSize', type=int, help='Number of data points.', default=1000)
	parser.add_argument('--saveFile', help='Save the dataset in a pickle file.', default=False, action='store_true')
	parser.add_argument('--HDF5', help='Save the dataset in a HDF5 file.', default=False, action='store_true')
	parser.add_argument('--filename', help='Pickle file name.', default="3Ddata")
	args = parser.parse_args()

	# outputs
	if not args.HDF5: dataset = {'i':[], 'X':[], 'Y':[], 'Z':[], 'Xprime':[], 'Yprime':[], 'Zprime':[], 'L':[]}
	else: hdf = h5py.File(args.filename+'.h5', 'w')

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
		
		# sample uniformly a pair of 3D points in [shift, 1+shift)
		points = np.random.uniform(shift, 1+shift, 6)
		points.shape = (2,3)
		P = points # P[0,:] is np.array([x1,y1,z1]), P[1,:] is np.array([x2,y2,z2])
		
		# or sample a pair of custom 3D points
		# x = np.random.uniform(-3820., 3820., 2)
		# y = np.random.uniform(-3820., 3820., 2)
		# z = np.random.uniform(-3521.1, 3521.1, 2)
		# P = np.column_stack((x,y,z)) # P[0,:] is np.array([x1,y1,z1]), P[1,:] is np.array([x2,y2,z2])

		line = Line(Point(P[0,:]), Point(P[1,:]))
		# line direction vector and its unit vector
		u = getLineDirection(line)
		u_hat = u/np.linalg.norm(u)

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
		if count == 1:
			# print some data
			# if (i % (args.sampleSize/10) == 0): print("i: %i, X: %f, Y: %f, Z: %f, Xprime: %f, Yprime: %f, Zprime: %f, L: %f" % (i, P[0,0], P[0,1], P[0,2], u_hat[0], u_hat[1], u_hat[2], lengthToBoundary))

			if not args.HDF5:
				# save to dataset dictionary for pickling
				dataset['i'].append(i)
				dataset['X'].append(P[0,0])
				dataset['Y'].append(P[0,1])
				dataset['Z'].append(P[0,2])
				dataset['Xprime'].append(u_hat[0])
				dataset['Yprime'].append(u_hat[1])
				dataset['Zprime'].append(u_hat[2])
				dataset['L'].append(lengthToBoundary)
			else:
				group = hdf.create_group('point'+str(i))
				group.create_dataset('position', data=P[0,:], compression='gzip')
				group.create_dataset('direction', data=u_hat, compression='gzip')
				group.create_dataset('length', data=np.array([lengthToBoundary]), compression='gzip')

	if args.saveFile:
		if not args.HDF5: 
			with open('data/'+args.filename+'.pickle', 'wb') as f: pickle.dump(dataset, f)
			print("\nFile %s saved!" % (args.filename+'.pickle'))
		else:
			hdf.close()
			print("\nFile %s saved!" % (args.filename+'.h5'))

	# plot some shit
	from matplotlib import pyplot as plt
	plt.hist(np.array(dataset['L']), bins=100, range=(0,2))
	plt.savefig('distance.pdf')
