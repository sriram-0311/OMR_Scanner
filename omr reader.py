import cv2
import numpy as np

img = cv2.imread('omr_8.jpeg')

r = 500/img.shape[1]
dim = (500, int(img.shape[0] * r))

resized = cv2.resize(img, dim, cv2.INTER_AREA)

cv2.imwrite('resize_8.jpg',resized)

pts1 = np.float32([[48,123],[425,106],[12,739],[481,733]])
pts2 = np.float32([[0,0],[500,0],[0,750],[500,750]])

M = cv2.getPerspectiveTransform(pts1,pts2)

dst1 = cv2.warpPerspective(resized, M, (500,808))
dst = cv2.fastNlMeansDenoisingColored(dst1,None,7,10,7,21)

gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

gray = cv2.GaussianBlur(gray,(5,5),0)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
cv2.imshow('resized', resized)
cv2.imshow('trans',thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()