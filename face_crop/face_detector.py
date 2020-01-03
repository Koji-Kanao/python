import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys, os
from PIL import Image

INPUT_PATH = 'dataset/celeb/'
OUTPUT_PATH = 'output'

def get_files(file_path):
    filenames = os.listdir(file_path)
    return filenames


pics = get_files(INPUT_PATH)

for i in pics:
    images = cv2.imread(INPUT_PATH + i)
    cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt.xml")
    face_list = cascade.detectMultiScale(images, scaleFactor=1.1, minNeighbors=1, minSize=(1,1))
    no = 1
    for rect in face_list:
        x = rect[0]
        y = rect[1]
        width = rect[2]
        height = rect[3]
        dst = images[y: y + height, x: x + width]
        save_path = OUTPUT_PATH + '/' + 'output_' + str(i) + '.jpg'
        result = cv2.imwrite(save_path, dst)
        #plt.show(plt.imshow(np.asarray(Image.open(save_path))))
        print(no)
        no += 1

