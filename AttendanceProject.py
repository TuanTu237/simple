import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Đường dẫn tới thư mục chứa hình ảnh
path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

# Khai báo biến lưu ngày điểm danh gần nhất
lastAttendanceDate = None

# Hàm điểm danh
def markAttendance(name):
    global lastAttendanceDate
    now = datetime.now()
    current_date = now.strftime('%Y-%m-%d')

    if lastAttendanceDate is None or lastAttendanceDate != current_date:
        dtString = now.strftime('%Y-%m-%d %H:%M:%S')

        with open('Attendance.csv', 'a') as f:
            f.write(f'{name},{dtString}\n')

        lastAttendanceDate = current_date
    else:
        print("Ngày hôm nay đã điểm danh. Hãy quay lại vào ngày mai.")

# Định nghĩa hàm để vẽ "râu" lên hình ảnh
def addMustache(img, faceLoc):
    y1, x2, y2, x1 = faceLoc
    mustache_width = x2 - x1
    mustache_height = int(0.2 * (y2 - y1))
    mustache = cv2.imread('mustache.png')  # Đọc hình ảnh "râu" (BGR, không có kênh alpha)
    mustache = cv2.resize(mustache, (mustache_width, mustache_height))

    roi = img[y1 + mustache_height:y1 + mustache_height * 2, x1:x1 + mustache_width]

    # Thêm "râu" vào hình ảnh
    img[y1 + mustache_height:y1 + mustache_height * 2, x1:x1 + mustache_width] = mustache

    return img

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()

            img = addMustache(img, faceLoc)
            markAttendance(name)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow('Webcam', img)

cap.release()
cv2.destroyAllWindows()
