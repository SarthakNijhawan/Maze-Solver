# A Maze Solver Algorithm implemented over an image derived using Image processing Algorithms
# Author: Sarthak Nijhawan

import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

def maze_extraction(img):
	"Takes a greyscale image as an input."
	
	# Locally Adaptive Binarisation # TODO : using OTSU's
	adapthresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2) #TODO
	
	# Median Filtered
	median = cv2.medianBlur(adapthresh, 1)

	# Invert the image to get all the regions
	invert = 255 - median

	# Region Labeling and extract the two larger perimeters 
	connectivity = 4
	output = cv2.connectedComponentsWithStats(invert, connectivity, cv2.CV_32S)
	# The first cell is the number of labels
	num_labels = output[0]
	# The second cell is the label matrix
	labels = output[1]
	# The third cell is the stat matrix
	stats = output[2]
	# The fourth cell is the centroid matrix
	centroids = output[3]

	# Find all the closed contours of all the labels
	arc_length = []
	for i in range(num_labels):
		region = cv2.threshold(labels, i+1, 255, cv2.THRESH_BINARY)
		image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		arc_length += [cv2.arc_length(image, contours)]

	# Find the the 2 most valued arguments in the list


	# Retrieve each region


	# Overlay both the maze walls to generate the final image 


	return [adapthresh, median, invert, labels] , ["Adaptively Thesholded", "Median Filtered", "Inverted", "Labels"]

def solutionpath_extraction(img):

	# Use a convex operation to extract all the points which lies under the region covered by the two walls as a whole


	# Find the contour of the region and Draw it and threhold the rest to 0


	# Final image with a black background and white solution path

	raise NotImplementedError

def morphological_thinning(img):
	"Performs morphological operations to reduce the solution space to a one pixel simply connected path without any loops."
	
	# Apply Zhang Seun thinning Algorithm


	raise NotImplementedError

def pruning_and_overlay(img):
	
	raise NotImplementedError

def algo(img):

	raise NotImplementedError

if __name__=="__main__":
	
	# input grayscale image
	input_img = cv2.imread(sys.argv[1], 0)
	
	# Capturing and Filtering the input image
	maze_extaction_steps, maze_extaction_titles = maze_extraction(input_img)
	
	# Image and Title Reservoir
	steps = [input_img, ] + maze_extaction_steps
	titles = ["Input Image", ] + maze_extaction_titles
	
	for i in range(len(steps)):
		plt.subplot(2, 4, i+1)
		plt.imshow(steps[i], cmap="gray")
		plt.title(titles[i])
	
	plt.show()
	
