import cv2


# 检测函数
def face_detect_demo(image):
    # 灰度转换
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 加载分类器
    face_detect = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt.xml')

    face = face_detect.detectMultiScale(gray_img)

    for x, y, w, h in face:
        cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
    cv2.imshow('result', image)


# 读取摄像头，打开本地摄像头
cap = cv2.VideoCapture(0)
# 循环
while True:
    ret, img = cap.read()
    if ret:
        face_detect_demo(img)

    if cv2.waitKey(1) == ord('q'):
        break
# 释放内存
cv2.destroyAllWindows()
# 释放摄像头
cap.release()
