# A Maze Solver Algorithm implemented over an image derived using Image processing Algorithms
# Author: Sarthak Nijhawan

import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

def processed_img(img):
	"Takes a greyscale image as an input."
	# Locally Adaptive Binarisation 
	adap_thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2) #TODO
	
	# Noise Reduction
	median = cv2.meadianBlur(adap_thresh, 3)
	
	# Region Labeling
	
		
	
	# Morphological Thinning
	raise NotImplementedError

def algo(img):
	raise NotImplementedError

if __name__=="__main__":
	input_img = cv2.imread(sys.argv[1], 0)
	pass
