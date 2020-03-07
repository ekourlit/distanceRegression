import argparse
import math
import numpy as np
import pickle
from geo import *
import pdb

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
		
		# the intersection should be within [0-1]
		if not (np.all(intersex>=0) and np.all(intersex<=1)): return None
		return intersex

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Create a dataset of 3D points, directions and distance to the boundary of a unit cube.')
	parser.add_argument('--sampleSize', type=int, help='Number of data points.', default=1000)
	parser.add_argument('--savePickle', help='Save the dataset in a pickle file.', default=False, action='store_true')
	parser.add_argument('--fileName', help='Pickle file name.', default="3Ddata")
	args = parser.parse_args()

	# output dictionary
	if args.savePickle: dataset = {'i':[], 'X':[], 'Y':[], 'Z':[], 'Xprime':[], 'Yprime':[], 'Zprime':[], 'L':[]}

	# create the cube planes
	boundaries = {
		'base' 	: Plane(Point(0,0,0), Point(1,0,0), Point(0,1,0)),
		'top' 	: Plane(Point(0,0,1), Point(1,0,1), Point(0,1,1)),
		'face' 	: Plane(Point(1,0,0), Point(1,1,0), Point(1,0,1)),
		'back' 	: Plane(Point(0,0,0), Point(0,1,0), Point(0,0,1)),
		'east' 	: Plane(Point(0,1,0), Point(1,1,0), Point(1,1,1)),
		'west' 	: Plane(Point(0,0,0), Point(1,0,0), Point(0,0,1))
	}

	for i in range(args.sampleSize):
		
		# pair of random 3D points in [0,1)
		P = np.random.rand(2,3) # P[0,:] is np.array([x1,y1,z1]), P[1,:] is np.array([x2,y2,z2])
		line = Line(Point(P[0,:]), Point(P[1,:]))

		count = 0
		for plane in boundaries:
			
			# Line - Plane Intersection point
			I = getLinePlaneIntersection(line, boundaries[plane])
			if I is not None:
				count =+ 1
				lengthToBoundary = Point(P[0,:]).distance_to(Point(I))
				assert lengthToBoundary < math.sqrt(3), "The length is longer than physically allowed!"

				# if (i % (args.sampleSize/10) == 0): 
				print("X: %f, Y: %f, Z: %f, Xprime: %f, Yprime: %f, Zprime: %f, L: %f" % (P[0,0], P[0,1], P[0,2], P[1,0], P[1,1], P[1,2], lengthToBoundary) )

		assert count < 2, "More than one intersection found! Unphysical!"

		# if (i % (args.sampleSize/10) == 0): print("Done with point %i" % i)
