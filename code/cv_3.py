import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def transform(path):
    img = cv.imread(path)
    # 将rgb图片转换到hsv空间
    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)), plt.title('original')
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    plt.subplot(122), plt.imshow(cv.cvtColor(img_hsv, cv.COLOR_HSV2RGB)), plt.title('hsv')
    plt.show()
    # 将rgb和hsv的通道分离
    b, g, r = cv.split(img)
    h, s, v = cv.split(img_hsv)
    plt.figure(figsize=(10, 5))
    plt.subplot(231), plt.imshow(r), plt.title('r')
    plt.subplot(232), plt.imshow(g), plt.title('g')
    plt.subplot(233), plt.imshow(b), plt.title('b')
    plt.subplot(234), plt.imshow(h), plt.title('h')
    plt.subplot(235), plt.imshow(s), plt.title('s')
    plt.subplot(236), plt.imshow(v), plt.title('v')
    plt.show()

    # 创建一个3D图
    fig = plt.figure(figsize=(15, 6))

    # 绘制R通道的三维图
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    # ax1=Axes3D(fig)
    x, y = np.arange(img.shape[1]), np.arange(img.shape[0])
    X, Y = np.meshgrid(x, y)
    Z = r.ravel()
    Z = Z.reshape(img.shape[0], img.shape[1])
    ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=False)
    ax1.set_title('R Channel')
    # plt.show()
    # 绘制G通道的三维图
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    Z = g.ravel()
    Z = Z.reshape(img.shape[0], img.shape[1])
    ax2.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=False)
    ax2.set_title('G Channel')

    # 绘制B通道的三维图
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    Z = b.ravel()
    Z = Z.reshape(img.shape[0], img.shape[1])
    ax3.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=False)
    ax3.set_title('B Channel')

    # 设置x, y, z轴的标签
    for ax in fig.get_axes():
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Intensity')

        # 显示图形
    plt.show()


def his(path):
    # 1. 读取彩色图像home_color
    home_color = cv.imread(path)

    # 2. 画出灰度化图像home_gray的灰度直方图，并拼接原灰度图与结果图
    home_gray = cv.cvtColor(home_color, cv.COLOR_BGR2GRAY)

    # 计算灰度直方图
    hist_gray = cv.calcHist([home_gray], [0], None, [256], [0, 256])

    # 归一化直方图
    hist_gray_norm = hist_gray.ravel() / hist_gray.max()

    # 创建直方图图像
    Q = hist_gray_norm.shape[0]
    bin_width = 256 / Q
    bin_centers = np.arange(bin_width / 2, 256, bin_width)

    plt.figure(figsize=(10, 7))

    # 绘制灰度直方图
    plt.subplot(2, 2, 1)
    plt.bar(bin_centers, hist_gray_norm, width=bin_width)
    plt.title('Grayscale Histogram')
    plt.xlabel('Pixel Values')
    plt.ylabel('Frequency')

    # 拼接原灰度图与直方图
    plt.subplot(2, 2, 2)
    plt.imshow(home_gray, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')

    # 3. 画出彩色home_color图像的直方图，并拼接原彩色图与结果图
    # 计算彩色直方图
    plt.subplot(2, 2, 3)
    colors = ('b', 'g', 'r')
    for i, col in enumerate(colors):
        histr = cv.calcHist([home_color], [i], None, [256], [0, 256])
        cv.normalize(histr, histr, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
        q = histr.shape[0]
        bin_width = 256 / q
        bin_centers = np.arange(bin_width / 2, 256, bin_width)

        plt.plot(bin_centers, histr, color=col)
        plt.xlim([0, 256])

    # 拼接原彩色图与直方图
    plt.subplot(2, 2, 4)
    plt.imshow(cv.cvtColor(home_color, cv.COLOR_BGR2RGB))
    plt.title('Color Image')
    plt.axis('off')

    # 显示所有图像
    plt.show()

    # 4. 定义ROI区域
    x_start, x_end = 50, 100
    y_start, y_end = 100, 200

    # 创建ROI的mask图
    roi_mask = np.zeros_like(home_color[:, :, 0])  # 使用单通道图像创建mask
    roi_mask[y_start:y_end, x_start:x_end] = 255

    # 使用mask图从原图中提取ROI
    roi = cv.bitwise_and(home_color, home_color, mask=roi_mask)

    # 计算ROI的直方图
    roi_pixels = roi[y_start:y_end, x_start:x_end, :].ravel()  # 获取ROI的像素值
    roi_pixels = roi_pixels.reshape(-1, 3)
    roi_hist, _ = np.histogram(roi_pixels[:, 0], bins=256, range=[0, 256])
    roi_hist_norm = roi_hist.astype('float') / roi_hist.max()

    # 创建直方图图像
    Q = roi_hist_norm.shape[0]
    bin_width = 256 / Q
    bin_centers = np.arange(bin_width / 2, 256, bin_width)

    # 在一个窗口中显示原图、ROI的mask图、提取后的ROI图以及ROI的直方图
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))

    # 显示原图
    axes[0, 0].imshow(cv.cvtColor(home_color, cv.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # 显示ROI的mask图
    axes[0, 1].imshow(roi_mask, cmap='gray')
    axes[0, 1].set_title('ROI Mask')
    axes[0, 1].axis('off')

    # 显示ROI提取后的图
    axes[1, 0].imshow(cv.cvtColor(roi, cv.COLOR_BGR2RGB))
    axes[1, 0].set_title('Extracted ROI')
    axes[1, 0].axis('off')

    # 显示ROI的直方图
    axes[1, 1].bar(bin_centers, roi_hist_norm, width=bin_width, color='gray')
    axes[1, 1].set_title('ROI Histogram')
    axes[1, 1].set_xlabel('Pixel Values')
    axes[1, 1].set_ylabel('Frequency')

    # 调整子图间距
    plt.tight_layout()

    # 显示所有图像
    plt.show()


def equal(path):
    img = cv.imread(path, 0)
    plt.figure(figsize=(12, 6))
    plt.subplot(121), plt.imshow(img, cmap='grey'), plt.title('original')
    height, width = img.shape

    num_pixel = np.zeros(256)  # 创建长度为256的列表，用来保存每个像素值的数目，初始化为0

    # 统计列表中某个值出现的次数
    for i in range(height):
        for j in range(width):
            k = img[i][j]  # 得到当前像素值
            num_pixel[k] = num_pixel[k] + 1  # 当前像素值的个数增加1(k的大小对应num_pixel的下标）

    # 计算原图灰度分布频率
    prob_pixel = np.zeros(256)
    for i in range(0, 256):
        prob_pixel[i] = num_pixel[i] / (height * width)  # 出现次数/总数

    # 计算原图累积分布频率
    cum_pixel = np.cumsum(prob_pixel)

    # 将累计分布中的元素乘以(L-1)(此处为255），再四舍五入，以使得均衡化后的图像的灰度级与原始图像一致。
    for i in range(len(cum_pixel)):
        cum_pixel[i] = int(cum_pixel[i] * 255 + 0.5)

    # 根据cum_pixel得到形式和原图相同的输出
    out_img = img  # 保持和原图格式相同
    for m in range(0, height):
        for n in range(0, width):
            k = img[m][n]
            out_img[m][n] = cum_pixel[img[m][n]]  # out_img中新的值，保存在cum_pixel中，由映射关系知，img[m][n]为对应下标

    plt.subplot(122), plt.imshow(out_img, cmap='grey'), plt.title('equal')
    plt.show()


if __name__ == '__main__':
    # transform('D:\\test.jpg')
    # his('D:\\home_color.png')
    equal('D:\\wallhaven-1k2g1v_1920x1080.png')
