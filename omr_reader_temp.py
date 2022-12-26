import cv2
import numpy as np


cv2.namedWindow('image')
img = cv2.imread('omr_4.jpeg')

r = 500/img.shape[1]
dim = (500,int(img.shape[0]*r))
resized = cv2.resize(img, dim, cv2.INTER_AREA)


dst = cv2.fastNlMeansDenoisingColored(resized,None,10,10,7,21)

gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray,15,22,20)


thresh = cv2.adaptiveThreshold(gray, 230, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11 , 2)
edges = cv2.Canny(thresh,30,200)
(x, cnts, _) = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for c in cnts:
    print(int(cv2.contourArea(c)))



cv2.imshow('image', thresh)
cv2.imshow('random',edges)

cv2.waitKey(0)
cv2.destroyAllWindows()