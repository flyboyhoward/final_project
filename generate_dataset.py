import airsim 
import sys
import os
import numpy as np
import math
import cv2 as cv

client = airsim.CarClient()
client.confirmConnection()
# client.enableApiControl(False)
client.simSetCameraOrientation(2, airsim.to_quaternion(0, 0, -math.pi/4));
client.simSetCameraOrientation(1, airsim.to_quaternion(0, 0, math.pi/4));
n = 3828
while True:
	# left camera = 2 right camera = 1

	responses = client.simGetImages([airsim.ImageRequest("2", airsim.ImageType.Scene, False, False)])
	response = responses[0]

	# get numpy array
	img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 

	# reshape array to 4 channel image array H X W X 4
	img_rgba = img1d.reshape(response.height, response.width, 3)  

# write to png 
	img_rgba = img_rgba[50:144, :, :]
	cv.imshow('img_rgba', img_rgba)
	if cv.waitKey(100) & 0xFF == ord('q'):
		break
	cv.imwrite('D:\\finalproject\\AirSim\\PythonClient\\car\\temp\\%04d.png' % n,img_rgba)
	n = n + 1