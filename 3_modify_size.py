import cv2

# 读取图片
img = cv2.imread('face.jpg')
# 修改尺寸
resize_img = cv2.resize(img, dsize=(int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)))
# 显示原图
cv2.imshow('img', img)
# 显示修改后的
cv2.imshow('resize_img', resize_img)
# 打印原图尺寸大小
print('未修改：', img.shape)
# 打印修改后的大小
print('修改后：', resize_img.shape)
# 等待
while True:
    # 显示图片
    cv2.imshow('img', img)

    if cv2.waitKey(1) == ord('q'):
        break
# 释放内存
cv2.destroyAllWindows()
# 保存改变尺寸的图片
# cv.imwrite('resize_img.jpg', resize_img)