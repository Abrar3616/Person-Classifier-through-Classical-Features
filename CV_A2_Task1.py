#!/usr/bin/env python
import glob
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

inputfolder: str = "/home/abrar/A2_images"

img1 = cv2.imread(inputfolder + '/book.jpg', 0)
img2 = cv2.imread(inputfolder + '/book_person_holding.jpg', 0)
img3 = cv2.imread(inputfolder + '/building_1.jpg', 0)
img4 = cv2.imread(inputfolder + '/building_2.jpg', 0)
img5 = cv2.imread(inputfolder + '/building_3.jpg', 0)
img6 = cv2.imread(inputfolder + '/roma_1.jpg', 0)
img7 = cv2.imread(inputfolder + '/roma_2.jpg', 0)

# Initiate ORB detector
#orb = cv2.ORB_create()
#sift = cv2.xfeatures2d.SIFT_create()
fast = cv2.FastFeatureDetector_create()
br = cv2.BRISK_create()

# find the keypoints and descriptors
kp1 = fast.detect(img1, None)
kp2 = fast.detect(img2, None)
kp3 = fast.detect(img3, None)
kp4 = fast.detect(img4, None)
kp5 = fast.detect(img5, None)
kp6 = fast.detect(img6, None)
kp7 = fast.detect(img7, None)

kp1, des1 = br.compute(img1,  kp1)
kp2, des2 = br.compute(img2,  kp2)
kp3, des3 = br.compute(img3,  kp3)
kp4, des4 = br.compute(img4,  kp4)
kp5, des5 = br.compute(img5,  kp5)
kp6, des6 = br.compute(img6,  kp6)
kp7, des7 = br.compute(img7,  kp7)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match descriptors.
matches1 = bf.match(des1, des2)
matches2 = bf.match(des3, des4)
matches3 = bf.match(des3, des5)
matches4 = bf.match(des4, des5)
matches5 = bf.match(des6, des7)



# Sort them in the order of their distance.
matches1 = sorted(matches1, key = lambda x:x.distance)
matches2 = sorted(matches2, key = lambda x:x.distance)
matches3 = sorted(matches3, key = lambda x:x.distance)
matches4 = sorted(matches4, key = lambda x:x.distance)
matches5 = sorted(matches5, key = lambda x:x.distance)

# Draw first 10 matches.
imgres1 = cv2.drawMatches(img1, kp1, img2, kp2, matches1[:10], outImg = None, flags=2)
imgres2 = cv2.drawMatches(img3, kp3, img4, kp4, matches2[:10], outImg = None, flags=2)
imgres3 = cv2.drawMatches(img3, kp3, img5, kp5, matches3[:10], outImg = None, flags=2)
imgres4 = cv2.drawMatches(img4, kp4, img5, kp5, matches4[:10], outImg = None, flags=2)
imgres5 = cv2.drawMatches(img6, kp6, img7, kp7, matches5[:10], outImg = None, flags=2)

cv2.imwrite(inputfolder + "/fast1.jpg", imgres1)
cv2.imwrite(inputfolder + "/fast2.jpg", imgres2)
cv2.imwrite(inputfolder + "/fast3.jpg", imgres3)
cv2.imwrite(inputfolder + "/fast4.jpg", imgres4)
cv2.imwrite(inputfolder + "/fast5.jpg", imgres5)
plt.imshow(imgres5)
plt.show()

