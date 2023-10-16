#face detection
#Landmarks detector
#Facial Landmark Detection with Mediapipe & Creating Animated Snapchat filters
#mediapie
'''
import cv2
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray,1.1,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h), (255,0,0),2)
        cv2.putText(img, "TU", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

'''
'''
import cv2

# Load the face cascade classifier
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load the image of a beard or mask that you want to add to the face
beard = cv2.imread("beard.png")

# Open a video capture stream from the default camera (usually 0)
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) which is the face
        roi = img[y:y+h, x:x+w]
        
        # Resize the beard image to match the size of the detected face
        resized_beard = cv2.resize(beard, (w, h))
        
        # Paste the resized beard onto the face
        img[y:y+h, x:x+w] = resized_beard
        
        cv2.imshow("Video", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''
import cv2
import face_recognition
 
imgElon = face_recognition.load_image_file('hoangbatu.png')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('hoangbatutest.png')

imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
 
faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
 
faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)
 
results = face_recognition.compare_faces([encodeElon],encodeTest)
faceDis = face_recognition.face_distance([encodeElon],encodeTest)
print(results,faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
 
cv2.imshow('Elon Musk',imgElon)
cv2.imshow('Elon Test',imgTest)
cv2.waitKey(0)