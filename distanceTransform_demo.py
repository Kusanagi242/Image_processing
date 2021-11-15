import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('Photo/2cell.jpg', cv.IMREAD_COLOR)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

b_img = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)[1]

# distance transform
d_img = cv.distanceTransform(b_img, cv.DIST_L2,3)
# print(np.max(d_img))

# normalize 
cv.normalize(d_img, d_img, 0,  1.0, cv.NORM_MINMAX)
# print(np.max(d_img))

out = cv.threshold(d_img, 0.85, 1.0, cv.THRESH_BINARY)[1]

cv.imshow('result image', out)
cv.waitKey(0)
cv.destroyAllWindows()