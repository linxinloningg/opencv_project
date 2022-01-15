import cv2

# 读取图片
img = cv2.imread('face.jpg')
# 灰度转换
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

while True:
    # 显示图片
    cv2.imshow('gray', gray_img)

    if cv2.waitKey(1) == ord('q'):
        break

# 释放内存
cv2.destroyAllWindows()

# 保存灰度图片
# cv.imwrite('gray_face.jpg', gray_img)
