import os
from tqdm import tqdm
from PIL import Image

path = r"D:/p_y/cv/INRIADATA/INRIADATA/normalized_images/train/pos/"
fileList = os.listdir(path)
for i in tqdm(fileList):
    img=Image.open(path+i)
    img.save(path+i)

path = r"D:/p_y/cv/INRIADATA/INRIADATA/normalized_images/train/neg/"
fileList = os.listdir(path)
for i in tqdm(fileList):
    img=Image.open(path+i)
    img.save(path+i)

path = r"D:/p_y/cv/INRIADATA/INRIADATA/original_images/train/pos/"
fileList = os.listdir(path)
for i in tqdm(fileList):
    img=Image.open(path+i)
    img.save(path+i)
