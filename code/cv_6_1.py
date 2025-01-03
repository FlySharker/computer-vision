import glob
import os
import cv2
import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 设置数据集的路径
positives_folder = 'D:\\p_y\\computer vision\\INRIADATA\\INRIADATA\\normalized_images\\train\\pos\\'
negatives_folder = 'D:\\p_y\\computer vision\\INRIADATA\\INRIADATA\\normalized_images\\train\\neg\\'
train_path = "D:\\p_y\\computer vision\\INRIADATA\\INRIADATA\\normalized_images\\train\\"
test_path = "D:\\p_y\\computer vision\\INRIADATA\\INRIADATA\\original_images\\train\\"
path = "D:\\p_y\\computer vision\\"


def load_data_set():
    # 提取正样本
    pos=[]
    neg=[]
    test=[]
    pos_dir = os.path.join(train_path, 'pos')
    if os.path.exists(pos_dir):
        pos = os.listdir(pos_dir)

    # 提取负样本
    neg_dir = os.path.join(train_path, 'neg')
    if os.path.exists(neg_dir):
        neg = os.listdir(neg_dir)

    # 提取测试集
    test_dir = os.path.join(test_path, 'pos')
    if os.path.exists(test_dir):
        test = os.listdir(test_dir)

    return pos, neg, test

#合并正样本与负样本并添加标签
def load_train_samples(pos, neg):
    pos_dir = os.path.join(train_path, 'pos')
    neg_dir = os.path.join(train_path, 'neg')

    samples = []
    labels = []
    for f in pos:
        file_path = os.path.join(pos_dir, f)
        if os.path.exists(file_path):
            samples.append(file_path)
            labels.append(1.)

    for f in neg:
        file_path = os.path.join(neg_dir, f)
        if os.path.exists(file_path):
            samples.append(file_path)
            labels.append(-1.)

    labels = np.int32(labels)
    labels_len = len(pos) + len(neg)
    labels = np.resize(labels, (labels_len, 1))

    return samples, labels

#提取图像的hog特征
def extract_hog(samples):
    train = []
    num = 0.
    total = len(samples)
    for f in samples:
        num += 1.
        # hog = cv2.HOGDescriptor((64,128), (16,16), (8,8), (8,8), 9)
        hog = cv2.HOGDescriptor()
        img = cv2.imread(f, 1)
        img = cv2.resize(img, (64, 128))
        img = np.array(img, dtype=np.uint8)
        descriptors = hog.compute(img)
        train.append(descriptors)

    train = np.float32(train)
    train = np.resize(train, (total, 3780))

    return train

#创建svm分类器
def get_svm_detector(svm):
    sv = svm.getSupportVectors()
    rho, _, _ = svm.getDecisionFunction(0)
    sv = np.transpose(sv)
    return np.append(sv, [[-rho]], 0)

#设置svm参数
def train_svm(train, labels):
    svm = cv2.ml.SVM_create()
    svm.setCoef0(0.0)
    svm.setDegree(3)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3)
    svm.setTermCriteria(criteria)
    svm.setGamma(0)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setNu(0.5)
    svm.setP(0.1)  # for EPSILON_SVR, epsilon in loss function?
    svm.setC(0.01)  # From paper, soft classifier
    svm.setType(cv2.ml.SVM_EPS_SVR)

    svm.train(train, cv2.ml.ROW_SAMPLE, labels)

    model_path = os.path.join(path, 'svm.xml')
    svm.save(model_path)

    return get_svm_detector(svm)

#行人检测
def test_hog_detect(test, svm_detector):
    hog = cv2.HOGDescriptor()
    # hog.setSVMDetector(svm_detector)
    # opencv自带的训练好了的分类器
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    test_dir = os.path.join(test_path, 'pos')
    cv2.namedWindow('Detect')
    for f in test:
        file_path = os.path.join(test_dir, f)
        img = cv2.imread(file_path)
        rects, _ = hog.detectMultiScale(img, winStride=(4, 4), padding=(8, 8), scale=1.05)
        for (x, y, w, h) in rects:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imshow('Detect', img)
        c = cv2.waitKey(0) & 0xff
        if c == 27:
            break
    cv2.destroyAllWindows()


# 加载INRIADATA数据集
def load_inria_data(positives_folder, negatives_folder):
    positives = []
    negatives = []
    filenames = os.listdir(positives_folder)
    for f in filenames:
        pos_path = os.path.join(positives_folder, f)
        positive = cv2.imread(pos_path, 0)
        positives.append(positive)

    filenames = os.listdir(negatives_folder)
    for f in filenames:
        neg_path = os.path.join(negatives_folder, f)
        negative = cv2.imread(neg_path, 0)
        negative=negative[:160,:96]
        negatives.append(negative)

    X = positives + negatives
    y = [1] * len(positives) + [0] * len(negatives)  # 1 for pedestrians, 0 for non-pedestrians
    return np.array(X), np.array(y)


# 提取HOG特征
def extract_hog_features(img):
    test = hog.compute(img)
    return hog.compute(img)


def draw(fpr, tpr):
    # 绘制ROC曲线
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    pos, neg, test = load_data_set()
    samples, labels = load_train_samples(pos, neg)
    train = extract_hog(samples)
    svm_detector = train_svm(train, labels)
    test_hog_detect(test, svm_detector)

    # 加载数据
    X, y = load_inria_data(positives_folder, negatives_folder)
    # 划分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 初始化HOG特征提取器
    hog = cv2.HOGDescriptor()
    X_train_hog = np.array([extract_hog_features(img).flatten() for img in X_train])
    X_test_hog = np.array([extract_hog_features(img).flatten() for img in X_test])
    # 训练SVM分类器
    clf = svm.SVC(probability=True)
    clf.fit(X_train_hog, y_train)
    # 预测测试集
    y_score = clf.predict_proba(X_test_hog)[:, 1]
    y_pred = clf.predict(X_test_hog)
    # 计算性能指标
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    # 画图
    draw(fpr, tpr)
