import cv2
import numpy as np
import os
import joblib
from scipy.cluster.vq import *
from sklearn import preprocessing

# 设置训练样本路径和聚类个数
train_path = "D:\\p_y\\cv\\animal\\"  # 训练样本文件夹路径
training_names = os.listdir(train_path)
numWords = 1000  # 聚类中心数

# 将所有训练图片路径放到一个列表中
image_paths = []  # 所有图片路径
ImageSet = {}
for name in training_names:
    ls = os.path.join(train_path, name)
    image_path = ls
    image_paths.append(ls)
    ImageSet[name] = 1

# 提取图片的sift特征
sift_det = cv2.SIFT_create()
des_list = []  # 特征描述
for name, count in ImageSet.items():
    dir = train_path + name
    print("从 " + name + " 中提取特征")
    img = cv2.imread(dir)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kp, des = sift_det.detectAndCompute(gray, None)
    des_list.append((image_path, des))

# 将提取到的sift特征聚合起来
descriptors = des_list[0][1]
print('生成向量数组')
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))

# k-means
print("开始 k-means 聚类: %d words, %d key points" % (numWords, descriptors.shape[0]))
voc, variance = kmeans(descriptors, numWords, 1)

# 计算特征的频率直方图
im_features = np.zeros((len(image_paths), numWords), "float32")
for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1], voc)
    for w in words:
        im_features[i][w] += 1

# Tf-Idf矢量化
nbr_occurences = np.sum((im_features > 0) * 1, axis=0)
idf = np.array(np.log((1.0 * len(image_paths) + 1) / (1.0 * nbr_occurences + 1)), 'float32')

# L2正则化
im_features = im_features * idf
im_features = preprocessing.normalize(im_features, norm='l2')

print('保存词袋模型文件')
joblib.dump((im_features, image_paths, idf, numWords, voc), "bow.pkl", compress=3)
