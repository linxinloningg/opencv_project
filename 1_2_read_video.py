import cv2

# 读取视频
cap = cv2.VideoCapture('test.mp4')

while True:
	ret, img = cap.read()
	 # 显示图片
	if ret:
        cv2.imshow('img', img)

    if cv2.waitKey(1) == ord('q'):
        break
# 释放内存
cv2.destroyAllWindows()