import cv2
import numpy
import matplotlib
from PIL import Image, ImageDraw, ImageFont


def read_image(path):
    #使用Image库中的open模块打开图片
    img = Image.open(path)
    return img


def show_image(image):
    #设置文字内容
    text = '尹林峰 21122867'
    #使用ImageDraw的Draw模块创建可绘制对象的draw
    draw = ImageDraw.Draw(image)
    #设置字体与大小
    tfont = ImageFont.truetype("C:\Windows\Fonts\SimHei.ttf", 40)
    #在图片的指定位置添加对应颜色的文字
    draw.text((100, 60), text, fill="yellow", font=tfont)
    #显示图像
    image.show()
    #保存图片
    image.save('D:\p_y\cv\img.png')


def read_video():
    #创建读取视频的对象cap
    cap = cv2.VideoCapture('Waymo.mp4')
    #判断对象是否读取成功
    while cap.isOpened():
        #获取每一帧的图像与结果
        ret, frame = cap.read()
        if ret == True:
            #显示图像
            cv2.imshow('frame', frame)
        #设置每一帧间隔25ms，且可以按回车退出
        if cv2.waitKey(25) == 13:
            break
    #释放视频
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    img = read_image('D:\\wallhaven-1k2g1v_1920x1080.png')
    show_image(img)
    read_video()
