import cv2

# 读取图片
img = cv2.imread('face.jpg')

while True:
    # 显示图片
    cv2.imshow('img', img)

    if cv2.waitKey(1) == ord('q'):
        break
# 释放内存
cv2.destroyAllWindows()
