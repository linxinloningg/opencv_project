import cv2


# 检测函数
def face_detect(image):
    # 生成灰度图，提高检测效率
    # 灰度转换
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 加载分类器
    face_detect = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt.xml')

    face = face_detect.detectMultiScale(gray_img, scaleFactor=1.01, minNeighbors=5, flags=0, minSize=(10, 10),
                                        maxSize=(100, 100))

    for x, y, w, h in face:
        cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
    cv2.imshow('result', image)


# 读取图像
srcImage = cv2.imread('face.jpg')

# 修改图片尺寸
srcImage = cv2.resize(srcImage, dsize=(200, 355))

# 检测函数
face_detect(srcImage)

# 等待
while True:
    if cv2.waitKey(1) == ord('q'):
        break
# 释放内存
cv2.destroyAllWindows()
