
'''
1 Read images videos and webcams
2 Basic Function
3 Resizing and cropping 
Thay đổi kích thước và cắt xén
4 shapes and text
5 warp prespective 
biến dạng dọc
6 joining image
7 color detection 
phát hiện màu sắc
8 contour / shape detection
9 Face Detection
   
Project
1. virtual paint ve ao
2. paper Scanner  may quet giay
3. number plate Detector may do bien so
'''
#read image, video, webcam
#image
#print("packed imported")
#img = cv2.imread("toycar.jpg")
#cv2.imshow("output", img)
#cv2.waitKey(0)

#video
''' 
cap = cv2.VideoCapture("sp.mp4")

while True:
    success, img = cap.read()
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''
''' 
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10,100)

while True:
    success, img = cap.read()
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''
import cv2 
import numpy as np 
    #basic function
img = cv2.imread("toycar.jpg")
kernel = np.ones((5,5),np.uint8)
img = cv2.resize(img,(500,550))

imgGray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

imgBlur = cv2.GaussianBlur(imgGray, (7,7), 0)
imgCanny = cv2.Canny(img, 100,100)
imgDialation = cv2.dilate(imgCanny,kernel, iterations=1)
imgEroded = cv2.erode(imgDialation, kernel, iterations=1) #iterations: lap lai

cv2.imshow("Gray image", imgGray) #lam den anh
cv2.imshow("Blur image", imgBlur) #lam xam anh va mo
cv2.imshow("Canny image", imgCanny) # lay nhung duong ke cua anh
cv2.imshow("dialation image", imgDialation) # lay cac duong ke
cv2.imshow("eroded image", imgEroded) # lay anh mo , xoi mon
cv2.waitKey(0)