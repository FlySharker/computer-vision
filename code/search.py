import cv2
import joblib
from scipy.cluster.vq import *
from sklearn import preprocessing
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
from PIL import Image

# 获取目标图片路径
image_path = "D:\\p_y\\cv\\animal_image_ dataset\\archive\\animals\\animals\\antelope\\6aa06f252d.jpg"

# 加载词袋模型
im_features, image_paths, idf, numWords, voc = joblib.load("bow.pkl")

# 提取目标图片sift特征
sift_det = cv2.SIFT_create()
des_list = []
im = cv2.imread(image_path)
gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
kp, des = sift_det.detectAndCompute(gray, None)
des_list.append((image_path, des))

# 聚合所有特征点
descriptors = des_list[0][1]

#计算特征的频率直方图
test_features = np.zeros((1, numWords), "float32")
words, distance = vq(descriptors, voc)
for w in words:
    test_features[0][w] += 1

# Tf-Idf 矢量化 and L2 正则化
test_features = test_features * idf
test_features = preprocessing.normalize(test_features, norm='l2')

score = np.dot(test_features, im_features.T)
rank_ID = np.argsort(-score)

# 结果可视化
figure('基于OpenCV的图像检索')
subplot(5, 5, 1)  #
title('目标图片')
imshow(im[:, :, ::-1])
axis('off')
for i, ID in enumerate(rank_ID[0][0:20]):
    img = Image.open(image_paths[ID])
    subplot(5, 5, i + 6)
    imshow(img)
    title('第%d相似' % (i + 1))
    axis('off')

show()