import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

img = cv2.imread(sys.argv[1], 0)

adap = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY, 11, 2)

gauss = cv2.GaussianBlur(adap, (5,5), 0)

_, otsu = cv2.threshold(gauss, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

plt.subplot(1,2,1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.subplot(1,2,2)
plt.imshow(otsu, cmap="gray")
plt.title("Processed")
plt.show()
