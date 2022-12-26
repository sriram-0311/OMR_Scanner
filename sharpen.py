import cv2
import numpy as np
import imutils

aprrox = []

img = cv2.imread('WhatsApp Image 2018-05-16 at 3.44.03 PM.jpeg')


img = imutils.resize(img,500)

gray = cv2.GaussianBlur(img, (9,9),0)



ret, thresh = cv2.threshold(gray,130,255,cv2.THRESH_BINARY)



kernel_sharpen_3 = np.array([[-1,-1,-1,-1,-1],
                             [-1,2,2,2,-1],
                             [-1,2,8,2,-1],
                             [-1,2,2,2,-1],
                             [-1,-1,-1,-1,-1]]) / 8.0

# applying different kernels to the input image
output_3 = cv2.filter2D(gray, -1, kernel_sharpen_3)
edges2 = cv2.Canny(output_3,60,200)
z, cnts, _ = cv2.findContours(edges2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # if our approximated contour has four points, then
    # we can assume that we have found our screen
    if len(approx) == 4:
        print(approx)
        print(len(approx),cv2.contourArea(approx))
        screenCnt = approx

        cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)
        pts1 = np.float32([approx[1], approx[0], approx[2], approx[3]])
        pts2 = np.float32([[0, 0], [500, 0], [0, 750], [500, 750]])

        M = cv2.getPerspectiveTransform(pts1, pts2)

        dst1 = cv2.warpPerspective(img, M, (500, 750))
        #dst1 = cv2.adaptiveThreshold(dst1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        break


cv2.imshow('Original', img)
cv2.imshow('Result', dst1)
cv2.waitKey(0)
cv2.destroyAllWindows()