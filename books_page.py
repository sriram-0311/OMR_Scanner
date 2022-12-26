import cv2, time
start_time = time.time()
img = cv2.imread('20180411110436196_027.JPG')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

r = 500/gray.shape[1]
dim = (500, int(gray.shape[0] * r))

resized = cv2.resize(gray,dim, cv2.INTER_AREA)
cv2.namedWindow('summa')
cv2.imshow('summa', resized)

print( "time taken is ", (time.time() - start_time))
cv2.waitKey(0)
cv2.destroyAllWindows()