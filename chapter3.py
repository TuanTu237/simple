import cv2
import numpy as np
#phong to phan cua buc anh
img = cv2.imread("toycar.jpg")
print(img.shape)
imgResize = cv2.resize(img,(300,200))
print(imgResize.shape)
imgCropped = img[0:200, 200:500]
cv2.imshow("imgae cropted", imgCropped)
cv2.waitKey(0)
#30.46 time