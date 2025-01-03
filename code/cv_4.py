import os
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

def img_read(path):
    # 初始化一个空列表来存储图片的向量表示
    image_vectors = []
    # 遍历根文件夹中的所有子文件夹和文件
    for root, dirs, files in os.walk(path):
        # 遍历当前目录下的所有文件
        for file in files:
            # 检查文件是否以.bmp结尾
            if file.endswith('.bmp'):
                # 构建文件的完整路径
                image_path = os.path.join(root, file)
                print(image_path)
                # 打开并读取图片
                img = Image.open(image_path)
                # 将图片转换为numpy数组
                img_array = np.array(img)
                # 将numpy数组转换为向量（flatten）
                img_vector = img_array.flatten()
                # 将向量添加到列表中
                image_vectors.append(img_vector)

                # 将列表转换为numpy二维数组（矩阵）
    image_matrix = np.array(image_vectors)

    # 输出矩阵的形状，以验证结果
    print(image_matrix)
    print(image_matrix.shape)

    return image_matrix

# def pca(X, n_components):
#     """
#     PCA实现
#     :param X: 数据集，形状为 (num_samples, num_features)
#     :param n_components: 要保留的主成分数量
#     :return: 降维后的数据
#     """
#     # 1. 数据标准化（均值为0，方差为1）
#     X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
#
#     # 2. 计算协方差矩阵
#     covariance_matrix = np.cov(X_std.T)
#
#     # 3. 计算协方差矩阵的特征值和特征向量
#     eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
#
#     # 4. 将特征向量按对应特征值大小排序
#     idx = eigenvalues.argsort()[::-1]
#     eigenvectors = eigenvectors[:, idx]
#
#     # 5. 选择前n个主成分
#     eigenvectors_pca = eigenvectors[:, :n_components]
#
#     # 6. 将原始数据投影到主成分上
#     X_pca = np.dot(X_std, eigenvectors_pca)
#
#     return X_pca

def train(data_pca):
    target = []
    for i in range(40):
        for j in range(10):
            target.append(i)
    X_train, X_test, y_train, y_test = train_test_split(data_pca, target, test_size=0.3, random_state=42)
    # 创建KNN分类器实例
    knn = KNeighborsClassifier(n_neighbors=1)
    # 拟合模型
    knn.fit(X_train, y_train)
    # 预测测试集结果
    y_pred = knn.predict(X_test)
    # 输出模型评估结果
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    image_folder = 'D:\p_y\cv\ORL人脸数据库'
    data=img_read(image_folder)
    pca=PCA(n_components=50)
    data_pca = pca.fit_transform(data)
    print(data_pca)
    print(data_pca.shape)
    train(data_pca)




