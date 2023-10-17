import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Khai báo các thư viện và module cần thiết.

path = 'ImagesAttendance'
images = []  # Danh sách để lưu trữ các hình ảnh khuôn mặt
classNames = []  # Danh sách để lưu trữ tên của người trong hình ảnh
myList = os.listdir(path)  # Lấy danh sách các tệp trong thư mục 'ImagesAttendance'
print(myList)

# Duyệt qua danh sách các tệp và thêm hình ảnh và tên vào danh sách 'images' và 'classNames'
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')  # Đọc hình ảnh từ tệp
    images.append(curImg)  # Thêm hình ảnh vào danh sách 'images'
    classNames.append(os.path.splitext(cl)[0])  # Thêm tên người từ tên tệp vào danh sách 'classNames'

print(classNames)

# Khai báo một biến để lưu ngày điểm danh gần nhất
lastAttendanceDate = None
#ham diem danh
def markAttendance(name):
    global lastAttendanceDate  # Sử dụng biến global để lưu ngày điểm danh gần nhất
    now = datetime.now()
    current_date = now.strftime('%Y-%m-%d')

    # Kiểm tra xem đã điểm danh vào ngày hiện tại chưa
    if lastAttendanceDate is None or lastAttendanceDate != current_date:
        dtString = now.strftime('%Y-%m-%d %H:%M:%S')  # Định dạng thời gian

        # Ghi thông tin điểm danh (tên và thời gian) vào tệp "Attendance.csv"
        with open('Attendance.csv', 'a') as f:  # Mở tệp ở chế độ ghi (append)
            f.write(f'{name},{dtString}\n')

        lastAttendanceDate = current_date  # Cập nhật ngày điểm danh gần nhất
    else:
        print("Ngày hôm nay đã điểm danh. Hãy quay lại ngày mai.")
# Định nghĩa hàm 'findEncodings' để mã hóa các khuôn mặt trong danh sách hình ảnh 'images'
def findEncodings(images):
    encodeList = []  # Danh sách để lưu trữ các mã hóa khuôn mặt
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển đổi không gian màu BGR sang RGB
        encode = face_recognition.face_encodings(img)[0]  # Mã hóa khuôn mặt đầu tiên trong hình ảnh
        encodeList.append(encode)  # Thêm mã hóa vào danh sách
    return encodeList

# Gọi hàm 'findEncodings' để mã hóa các khuôn mặt và lưu vào 'encodeListKnown'
encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Khởi tạo một đối tượng video capture để sử dụng webcam
cap = cv2.VideoCapture(0)

# Bắt đầu vòng lặp chính để xử lý hình ảnh từ webcam
while True:
    success, img = cap.read()  # Đọc một khung hình từ webcam
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # Thay đổi kích thước hình ảnh
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)  # Chuyển đổi không gian màu

    # Tìm vị trí của các khuôn mặt trong hình ảnh
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    # Duyệt qua danh sách các khuôn mặt trong hình ảnh
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)  # So sánh với danh sách khuôn mặt đã biết
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)  # Tính khoảng cách giữa khuôn mặt

        matchIndex = np.argmin(faceDis)  # Tìm chỉ số của khuôn mặt gần nhất

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()  # Lấy tên từ danh sách 'classNames'

            # Vẽ hộp xung quanh khuôn mặt và hiển thị tên
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            # Ghi thông tin điểm danh vào tệp "Attendance.csv"
            markAttendance(name)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Nếu người dùng nhấn phím 'q', thoát khỏi vòng lặp
        break

    cv2.imshow('Webcam', img)  # Hiển thị hình ảnh từ webcam

# Kết thúc vòng lặp, giải phóng tài nguyên và đóng cửa sổ hiển thị
cap.release()
cv2.destroyAllWindows()
