## HW 7-8, Pratik Jawahar
import cv2
import os

#directory path, where my video images will be stored
#os.mkdir(r'/Users/pratcr7/Desktop/pythonP/DLAP-Morato/HW78/parsedFrames')

#video number   
flag = 7

vidcap = cv2.VideoCapture('/Users/pratcr7/Desktop/pythonP/DLAP-Morato/HW78/TestVids/vid7.mp4')
def getFrame(sec,pathOut):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    #height, width, channels = image.shape
    if hasFrames:
    	#if height < width:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        #image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(os.path.join(pathOut, str(flag)+"frame"+str(count)+".jpg"), image)
    return hasFrames
sec = 0
frameRate = 1/15 #//it will capture image in each 0.5 second
count=1
success = getFrame(sec, 'parsedFrames')
while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec, 'parsedFrames')
