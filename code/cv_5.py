import cv2
import os
from matplotlib import pyplot as plt

# 加载预训练的Haar分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 读取测试图像
folder_path="D:\\p_y\\cv\\WiderFace\\WIDER_train\\images\\28--Sports_Fan\\"
# 获取文件夹中所有文件的名称
image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
cnt=0
# 遍历文件并读取图片
for file in image_files:
    if cnt==5:
        break
    # 构建图片完整路径
    image_path = os.path.join(folder_path, file)
    # 使用OpenCV读取图片
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 10)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
    cnt=cnt+1

