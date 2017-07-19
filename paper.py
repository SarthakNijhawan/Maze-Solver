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

	# Find the regions with the maximum perimeter as maze walls


	return [adapthresh, median, invert, labels] , ["Adaptively Thesholded", "Median Filtered", "Inverted", "Labels"]

def solutionpath_extraction(img):
	raise NotImplementedError

def morphological_thinning(img):
	"Performs morphological operations to reduce the solution space and build a succinct skeleton further processing"
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
	
