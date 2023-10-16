#joining images
import cv2
import numpy as np
#ghep 2 anh thanh 1

img = cv2.imread("quanco.png")
imgHor = np.hstack((img,img) )
imgVer = np.vstack((img, img))

cv2.imshow("Horizontal",imgHor)
cv2.imshow("Vertical",imgVer)
cv2.waitKey(0)