import cv2

# 读取图片
img = cv2.imread('face.jpg')
# 坐标
x, y, w, h = 100, 100, 100, 100
# 绘制矩形
cv2.rectangle(img, color=(0, 0, 255), thickness=1, pt1=(x, y), pt2=(x + w, y + h))
# 绘制圆形
cv2.circle(img, center=((x + w), y + h), radius=100, color=(255, 0, 0), thickness=5)

while True:
    # 显示图片
    cv2.imshow('img', img)

    if cv2.waitKey(1) == ord('q'):
        break
# 释放内存
cv2.destroyAllWindows()
