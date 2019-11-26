import os
from pathlib import Path
import cv2
import numpy as np
import random

exe_dir_path = os.path.dirname(__file__)

SIZE = 100


def generate_data(dir_name, n=1):
    directory = os.path.join(exe_dir_path, dir_name)
    if not os.path.isdir(directory):
        os.mkdir(directory)
    radius = 2
    for i in range(n):
        img = np.zeros((SIZE,SIZE),np.uint8)
        random_coords = (random.uniform(radius,SIZE-radius),random.uniform(radius,SIZE-radius))
        cv2.circle(img,(int(random_coords[0]),int(random_coords[1])),radius,255,thickness=-1)
        cv2.imwrite(os.path.join(dir_name,str(random_coords[0])+','+str(random_coords[1]) + '.jpg'), img)


def load_data(folder):
    x = []
    y = []
    dir = os.path.join(exe_dir_path,'data',folder)
    for file in os.listdir(dir):
        x.append(cv2.imread(os.path.join(dir,file), cv2.IMREAD_GRAYSCALE))
        label = file[:-4].split(sep=',')
        label_tpl = (int(float(label[0])), int(float(label[1])))
        y.append(label_tpl)
    x = np.array(x)
    y = np.array(y) / SIZE
    return x, y


def load_img(image_path, img_rows, img_cols):
    img = np.array(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))

    img = img.reshape(-1, img_rows, img_cols, 1)
    img = img/255
    return img


