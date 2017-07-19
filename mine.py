# A Maze Solver Algorithm implemented over an image derived using Image processing Algorithms
# Author: Sarthak Nijhawan

import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

def maze_extraction(img):
        "Takes a greyscale image as an input."
        
        # OTSU's method
        _, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        # Median Filtered
        median = cv2.medianBlur(otsu, 5)

        kernel = np.ones((3,3),np.uint8)
        
        # Morplogical opening to revive thin maze lines before removing black patches
        opening = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel)

        # Morphological Closing to eliminate small balck patches
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        # Median Filtered
        median2 = cv2.medianBlur(closing, 5)

        # Invert the image to get all the regions
        invert = 255 - median2

        # Region Labeling and extract the larger perimeter regions as maze walls
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
        

        return [otsu, median, opening, closing, median2, invert, labels] , ["Otsu", "Median Filtered", "Opening", "Closing", "Median", "Invert", "Labeled"]

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
