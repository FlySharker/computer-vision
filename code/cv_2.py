import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def img_change(path):
    # 读取彩色图像(BGR)
    img = cv.imread(path)
    rows, cols, ch = img.shape
    # dx=100 向右偏移量, dy=150 向下偏移量
    dx, dy = 100, 150
    # 构造平移变换矩阵
    MAT = np.float32([[1, 0, dx], [0, 1, dy]])
    # 仿射变换
    dst = cv.warpAffine(img, MAT, (cols, rows))

    plt.figure(figsize=(9, 6))
    plt.subplot(121), plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)), plt.title("Original")
    plt.subplot(122), plt.imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB)), plt.title("Move")
    plt.show()


def smaller(path):
    # 读取彩色图像(BGR)
    img = cv.imread(path)
    # 对图片进行放缩
    img1 = cv.resize(img, (1024, 768))
    img2 = cv.resize(img, None, fx=0.6, fy=0.6, interpolation=cv.INTER_AREA)

    plt.figure(figsize=(12, 7))
    plt.subplot(221), plt.title("img:original")
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.subplot(222), plt.title("img1: 1024*768")
    plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
    plt.subplot(223), plt.title("img2: fx=0.6,fy=0.6")
    plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
    plt.show()


def flip(path):
    img = cv.imread(path)  # 读取彩色图像(BGR)
    imgFlip1 = cv.flip(img, 0)  # 垂直翻转
    imgFlip2 = cv.flip(img, 1)  # 水平翻转
    imgFlip3 = cv.flip(img, -1)  # 水平和垂直翻转

    plt.figure(figsize=(9, 6))
    plt.subplot(221), plt.axis('off'), plt.title("Original")
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))  # 原始图像
    plt.subplot(222), plt.axis('off'), plt.title("Flipped Horizontally")
    plt.imshow(cv.cvtColor(imgFlip2, cv.COLOR_BGR2RGB))  # 水平翻转
    plt.subplot(223), plt.axis('off'), plt.title("Flipped Vertically")
    plt.imshow(cv.cvtColor(imgFlip1, cv.COLOR_BGR2RGB))  # 垂直翻转
    plt.subplot(224), plt.axis('off'), plt.title("Flipped Horizontally & Vertically")
    plt.imshow(cv.cvtColor(imgFlip3, cv.COLOR_BGR2RGB))  # 水平垂直翻转
    plt.show()


def rotate(path):
    img = cv.imread(path)  # 读取彩色图像(BGR)
    height, width = img.shape[:2]  # 图片的高度和宽度
    theta1 = 30  # 顺时针旋转角度，单位为角度
    x0, y0 = width // 2, height // 2  # 以图像中心作为旋转中心
    MAR1 = cv.getRotationMatrix2D((x0, y0), theta1, 1.0)  # 求解旋转变换矩阵
    imgR1 = cv.warpAffine(img, MAR1, (width, height))  # 旋转变换，默认为黑色填充

    plt.figure(figsize=(10, 6))
    plt.subplot(121), plt.axis('off'), plt.title(r"$Origin$")
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.subplot(122), plt.axis('off'), plt.title(r"$(0.5*width,0.5*height),Rotation {}^o$".format(theta1))
    plt.imshow(cv.cvtColor(imgR1, cv.COLOR_BGR2RGB))
    plt.show()


def smallest(path):
    img = cv.imread(path)
    img_small = cv.resize(img, None, fx=0.2, fy=0.2)#放缩
    width, height = img_small.shape[:2]
    x, y = 0, 0#设置放置的左上角坐标
    img[x:x + width, y:y + height] = img_small#放置
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()


def grey_chip(path):
    img = cv.imread(path, 0)#以灰度图形式读取
    img = cv.resize(img, (700, 700))#放缩成正方形
    width, height = img.shape[:2]
    #设置圆形遮罩
    mask1 = np.zeros(np.shape(img), dtype=np.uint8)
    cv.circle(mask1, (width // 2, height // 2), 200, (255, 255, 255), -1)
    #提取圆形ROI
    img_make = cv.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask1)
    cv.imshow("img", img_make)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    img_change('D:\\1b91d753cbb64883aee1dacb31bea7bd.png')
    smaller('D:\\1b91d753cbb64883aee1dacb31bea7bd.png')
    flip('D:\\1b91d753cbb64883aee1dacb31bea7bd.png')
    rotate('D:\\1b91d753cbb64883aee1dacb31bea7bd.png')
    smallest('D:\\1b91d753cbb64883aee1dacb31bea7bd.png')
    grey_chip('D:\\1b91d753cbb64883aee1dacb31bea7bd.png')
