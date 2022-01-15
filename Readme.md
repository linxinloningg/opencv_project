### opencv保姆级别教入门，不会就把我头打爆。

全部代码链接:https://github.com/linxinloningg/opencv_project

#### 环境配置：

```bash
pip install opencv-python
```

#### opencv使用步骤：

归结起来就只是三步：

读取-->> 调整 -->> 检测

* 读取：
  * 读取图片
  * 读取视频
  * 读取摄像头
* 调整
  * 灰度调整
  * 大小调整
* 检测
  * 标记（画一个框、圆圈、or something来显示检测的东c）
  * 调用不同的模型文件检测
    * 图片检测
    * 视频检测
    * 摄像头检测
    * 等等

#### 代码链接：

* 读取

  * 图片

    ```python
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
    ```

  * 视频

    ```python
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
    ```

  * 摄像头

    ```python
    import cv2
    
    # 读取摄像头
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, img = cap.read()
        # 显示图片
        if ret:
            cv2.imshow('img', img)
    
        if cv2.waitKey(1) == ord('q'):
            break
    # 释放内存
    cv2.destroyAllWindows()
    ```

* 调整

  * 灰度调整

    ```python
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
    ```

  * 大小调整

    ```python
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
    ```

* 检测

  * 画框

    ```python
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
    ```

  * 调用不同的模型文件检测

    * ```python
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
      ```

    * ```python
      import cv2
      
      
      # 检测函数
      def face_detect(image):
          # 灰度转换
          gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
          # 加载分类器
          face_detect = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt.xml')
      
          face = face_detect.detectMultiScale(gray_img)
      
          for x, y, w, h in face:
              cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
          cv2.imshow('result', image)
      
      
      # 读取视频
      cap = cv2.VideoCapture('test.mp4')
      # 循环
      while True:
          ret, img = cap.read()
          if ret:
              face_detect(img)
      
          if cv2.waitKey(1) == ord('q'):
              break
      # 释放内存
      cv2.destroyAllWindows()
      # 释放摄像头
      cap.release()
      ```

    * ```python
      import cv2
      
      
      # 检测函数
      def face_detect(image):
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
              face_detect(img)
      
          if cv2.waitKey(1) == ord('q'):
              break
      # 释放内存
      cv2.destroyAllWindows()
      # 释放摄像头
      cap.release()
      ```

* 增强

  * GPU应用

    ```python
    """
    cpu_gpu.py
    An OpenCL-OpenCV-Python CPU vs GPU comparison
    """
    import cv2
    import timeit
    
    
    # A simple image pipeline that runs on both Mat and Umat
    def img_cal(img, mode):
        if mode == 'UMat':
            img = cv2.UMat(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (7, 7), 1.5)
        img = cv2.Canny(img, 0, 50)
        if type(img) == 'cv2.UMat':
            img = cv2.UMat.get(img)
        return img
    
    
    # Timing function
    def run(processor, function, n_threads, N):
        cv2.setNumThreads(n_threads)
        t = timeit.timeit(function, globals=globals(), number=N) / N * 1000
        print('%s avg. with %d threads: %0.2f ms' % (processor, n, t))
        return t
    
    
    img = cv2.imread('face.jpg')
    N = 1000
    threads = [1, 16]
    
    processor = {'GPU': "img_cal(img_UMat)",
                 'CPU': "img_cal(img)"}
    results = {}
    n_threads = 1
    
    for n in n_threads:
        for pro in processor.keys():
            results[pro, n] = run(processor=pro,
                                  function=processor[pro],
                                  n_threads=n, N=N)
    
    print('\nGPU speed increase over 1 CPU thread [%%]: %0.2f' % \
          (results[('CPU', 1)] / results[('GPU', 1)] * 100))
    print('CPU speed increase on 4 threads versus 1 thread [%%]: %0.2f' % \
          (results[('CPU', 1)] / results[('CPU', 16)] * 100))
    print('GPU speed increase versus 4 threads [%%]: %0.2f' % \
          (results[('CPU', 4)] / results[('CPU', 1)] * 100))
    ```

* 训练