import cv2
import numpy as np
import os
from os.path import isfile, join

flag = 5
name = 'th3'
pathIn= '/Users/pratcr7/Desktop/pythonP/DLAP-Morato/CV/' + str(name) + '/' + str(name) + '_' + str(flag) + '/'
pathOut = 'Stitched_vids/' + str(flag) + '/110video_'+ str(name) + '.mp4'
fps = 15
frame_array = []
files = sorted([f for f in os.listdir(pathIn) if f != '.DS_STORE'],key=lambda f: f.lower())

frame_array = []
for i in range(len(files)):
    filename=pathIn + str(flag) + str(name) + '_' + str(i) + ".jpg"
    print(filename)
    #reading each files
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    
    #inserting the frames into an image array
    frame_array.append(img)
out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])
out.release()