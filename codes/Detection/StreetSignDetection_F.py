import numpy as np
import math
import cv2
import scipy.ndimage as ndimage
import matplotlib.pyplot as plot
import os
from os.path import isfile, join


def bwareafilt ( image ):
############Area filtering on binarized image to get rid of any noise########################

    #Binary image type force
    image = image.astype(np.uint8)      #Type force to UNIT8 for easier processing

    #Extracting features of all 4-connected pixels from the binary image
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4) 
    sizes = stats[:, -1]

    max_label = 1
    max_size = sizes[1]

    #Filtering out the largest 4-connected image region
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i               #
            max_size = sizes[i]         #Getting coordinates to feed into BBox algo

    img2 = np.zeros(output.shape)
    img2[output == max_label] = 255     #Removing other 4-connected regions, thereby acheiving significant 'Salt and Pepper' Noise Reduction

    return img2


def detect_signs ( pathIn,pathOut,flag ):
############Calculation for the image unknowns########################
	
	#Reading all parsed images extracted from the  video
	files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
	files.sort(key = lambda x: x[5:-4])                       #Sorting image frames
	files.sort()

	for i in range(len(files)):
		filename=pathIn + str(flag) + "frame" + str(i+1) + ".jpg"
		print(filename)
		print("len = ", len(files))
		
		#reading each file
		img = cv2.imread(filename)
		img1 = cv2.resize(img,(360,640))
		
		#Applying gaussian blurring filter to reduce noise in the image 	
		img = cv2.GaussianBlur(img1,ksize=(11,11),sigmaX=1)
		
		imgHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)           #Convert BGR image to HLS image.
		imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)           #Convert BGR image to HSV image.

		Lchannel = (imgHLS[:,:,1])

		Hchannel = imgHLS[:,:,0]

		Schannel = imgHLS[:,:,2]

		tot = np.sum(Lchannel)                                  #This is the sum of values for all the pixels in the Luminance channel
		a,b=np.shape(Lchannel)                                  #Find the shape of the numpy array in order to compute the average of pixels.
		mean = (tot)/(a*b)                                      #Find the mean of the pixels

		norm_mean = mean/256                                    #This is the normalized mean.
		threshold = math.exp(-norm_mean)

		############Calculation for the refernce values#############################

		Hlr = np.full((360,640),123).T
		Hlr = (2*math.pi/256)*Hlr                   #Convert to radian for proessing using numpy.

		cosHR = np.cos(Hlr)                                 #This is cosH1
		sinHR = np.sin(Hlr)                                 #This is sinH1

		Slr = np.full((360,640),255).T
		S1 = Slr/256                                        #This is the normalized Saturation value scale for reference.

		

		Hlr = np.around(((255/180)*Hchannel))
		Hradian = (2*math.pi/256)*Hlr                      #Convert the hue values for the image to corresponding angles in radians
		S2 = Schannel/256                                       #Normalize the S channel for the image

		cosH2 = np.cos(Hradian)                                 #The unknown colours cos values
		sinH2 = np.sin(Hradian)                                 #The unknown colours sin values

		result1 = np.multiply(S2,cosH2) - np.multiply(S1,cosHR)     #numpy.mat will calculate the element wise product for the given matrices.
		result2 = np.multiply(S2,sinH2) - np.multiply(S1,sinHR)

		result = np.square(result1) + np.square(result2)

		#This is the Euclidean Distance the thesis mentions. It is a numpy matrix in which all values represent the euclidean distance
		# of the given colour from the reference colour.
		result = np.sqrt(result)

		final = result #Final thresholded image

		ret3,th3 = cv2.threshold(imgHLS,threshold,255,cv2.THRESH_BINARY)
		# th4 = bwareafilt(th3) 
		th4 = th3

		#This is setting the values above the threshold to zero.
		final[final > threshold] = 0

		
		#Setting the paths and storing the processed images in the respective folders
		path_th3 = os.path.join(pathOut, "th3" + str(flag))
		cv2.imwrite(os.path.join(path_th3, str(flag)+"th3_"+str(i)+".jpg"), th3)
		
		path_th4 = os.path.join(pathOut, "th4" + str(flag))
		cv2.imwrite(os.path.join(path_th4, str(flag)+"th4_"+str(i)+".jpg"), th4)

		path_ori = os.path.join(pathOut, "original" + str(flag))
		cv2.imwrite(os.path.join(path_ori, str(flag)+"ori_"+str(i)+".jpg"), img1)		
		
		path_hls = os.path.join(pathOut, "hls" + str(flag))
		cv2.imwrite(os.path.join(path_hls, str(flag)+"hls"+str(i)+".jpg"), imgHLS)
		
		path_hsv = os.path.join(pathOut, "hsv" + str(flag))
		cv2.imwrite(os.path.join(path_hsv, str(flag)+"hsv"+str(i)+".jpg"), imgHSV)
		

	return (path_th3,path_th4,path_ori,path_hls,path_hsv)

# def join_frames ( pathIn,pathOut ):
# ############Stitching processed frames into a single '.mp4' file########################

# 	#path of the processed frames
# 	pathIn = os.path.join(pathIn, "/")
	
# 	fps = 15            #Setting the frame rate at which the frames are put together. 
# 	frame_array = []
	
# 	#Sourcing all frames to be combined
# 	files = [f for f in os.listdir(pathIn) if not f.startswith('.') and f != "installer.failurerequests" and isfile(join(pathIn, f))]

# 	#for sorting the file names properly
# 	files.sort(key = lambda x: x[5:-4])
# 	files.sort()
# 	frame_array = []

# 	for i in range(len(files)):
# 	    filename=pathIn + files[i]
# 	    print(filename)
# 	    #reading each files
# 	    img = cv2.imread(filename)
# 	    height, width, layers = img.shape
# 	    size = (width,height)
	    
# 	    #inserting the frames into an image array
# 	    frame_array.append(img)
# 	out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
# 	for i in range(len(frame_array)):
# 	    # writing to a image array
# 	    out.write(frame_array[i])
# 	out.release()

count = 5   #Video number
pathIn = ('/Users/pratcr7/Desktop/pythonP/DLAP-Morato/CV/parsedFrames/5/') #Sourcing the parsed video frames
pathOut = '/Users/pratcr7/Desktop/pythonP/DLAP-Morato/CV/'                 #Output path to write the processed frames into

#Running the street sign detection algorithm
path_th3,path_th4,path_ori,path_hls,path_hsv = detect_signs(pathIn,pathOut,count)

#Output paths to write the video output of the frame stitching function
pathOut_th3 = os.path.join(pathOut, "/stitch_th3/")
pathOut_th4 = os.path.join(pathOut, "/stitch_th4/")
pathOut_final = os.path.join(pathOut, "/stitch_ori/")
pathOut_hls = os.path.join(pathOut, "/stitch_hls/")
pathOut_hsv = os.path.join(pathOut, "/stitch_hsv/")

#Stitching frames into video format
# join_frames(path_th3,pathOut_th3)
# join_frames(path_th4,pathOut_th4)
# join_frames(path_final,pathOut_final)
# join_frames(path_hls,pathOut_hls)
# join_frames(path_hsv,pathOut_hsv)

