import cv2
import numpy as np

mtx = np.array([[9.179617641817715139e+02,0.000000000000000000e+00,3.151351463595592008e+02],
[0.000000000000000000e+00,9.179708317058672264e+02,5.759832564317299557e+02],
[0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00]])

dist = np.array([[ 2.559945793970949124e-01,-1.976534805999403233e+00,-2.087487763142407624e-03,-2.751098535238335422e-03,4.774180081608886184e+00
]])

print(mtx)
print(dist)

img = cv2.imread('C:/Users/Sriram Ramesh/PycharmProjects/TRS/camera_calib_2/WhatsApp Image 2018-05-17 at 2.02.05 PM.jpeg')
h, w = img.shape[:2]


newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# undistort
mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
print ("ROI: ", x, y, w, h)

cv2.imshow('undistorted', dst)
cv2.imwrite('test-calib.jpg',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()