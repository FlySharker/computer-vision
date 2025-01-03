# 导入必要的库
import cv2
import numpy as np

# 打开名为test.avi的视频文件
camera = cv2.VideoCapture("test.avi")

# 获取视频的帧率
fps = camera.get(cv2.CAP_PROP_FPS)

# 获取视频帧的尺寸
size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# 创建名为test_result.avi的视频写入对象
# 设置视频编解码方式为'I', '4', '2', '0'
ViideoWrite = cv2.VideoWriter("test_result.avi", cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)

# 定义形态学操作的结构元素
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))
background = None

# 循环读取摄像头帧并处理
while True:

    grabbed, frame_lwpCV = camera.read()  # 读取摄像头帧

    gray_lwpCV = cv2.cvtColor(frame_lwpCV, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
    gray_lwpCV = cv2.GaussianBlur(gray_lwpCV, (21, 21), 0)  # 高斯模糊

    if background is None:
        background = gray_lwpCV
        continue

    diff = cv2.absdiff(background, gray_lwpCV)  # 计算背景差
    diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]  # 二值化阈值处理
    diff = cv2.dilate(diff, es, iterations=2)  # 形态学膨胀

    # 寻找轮廓并标记
    contours, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < 1500:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame_lwpCV, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 显示标记后的帧和处理后的差异图像
    cv2.imshow('contours', frame_lwpCV)
    ViideoWrite.write(frame_lwpCV)
    cv2.imshow('dis', diff)
    background = gray_lwpCV
    key = cv2.waitKey(20) & 0xFF

    # 检测按键，如果是'q'则退出循环
    if key == ord('q'):
        break

# 释放摄像头并关闭所有窗口
camera.release()
cv2.destroyAllWindows()