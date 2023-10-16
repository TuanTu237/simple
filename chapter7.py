import cv2
import numpy as np

# Hàm callback cho các Trackbars, không thực hiện gì cả
def empty(a):
    pass
#chinh mau anh theo cac H  Hue S Sat V val
img = cv2.imread("quanco.png")
cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 640, 240)
cv2.createTrackbar("Hue Min", "Trackbars", 0, 179, empty)
cv2.createTrackbar("Hue Max", "Trackbars", 179, 179, empty)
cv2.createTrackbar("Sat Min", "Trackbars", 0, 255, empty)
cv2.createTrackbar("Sat Max", "Trackbars", 255, 255, empty)
cv2.createTrackbar("Val Min", "Trackbars", 0, 255, empty)
cv2.createTrackbar("Val Max", "Trackbars", 255, 255, empty)

while True:
    # Đọc giá trị của các Trackbars
    h_min = cv2.getTrackbarPos("Hue Min", "Trackbars")
    h_max = cv2.getTrackbarPos("Hue Max", "Trackbars")
    s_min = cv2.getTrackbarPos("Sat Min", "Trackbars")
    s_max = cv2.getTrackbarPos("Sat Max", "Trackbars")
    v_min = cv2.getTrackbarPos("Val Min", "Trackbars")
    v_max = cv2.getTrackbarPos("Val Max", "Trackbars")

    # Tạo một mặt nạ (mask) dựa trên các giá trị ngưỡng HSV
    #HSV la  Hue (Màu sắc), Saturation (Độ bão hòa), và Value (Giá trị sáng
    lower_bound = np.array([h_min, s_min, v_min])
    upper_bound = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(img, lower_bound, upper_bound)

    # Áp dụng mặt nạ vào hình ảnh gốc để tạo hình ảnh kết quả
    result = cv2.bitwise_and(img, img, mask=mask)

    # Hiển thị hình ảnh gốc, hình ảnh kết quả và mặt nạ
    cv2.imshow("Original", img)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)

    # Thoát khỏi vòng lặp khi nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
