import airsim 
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import cv2 as cv
# import torch
import road_detection as predict

client = airsim.CarClient()
client.confirmConnection()
# client.enableApiControl(False)
client.simSetCameraOrientation(2, airsim.to_quaternion(0, 0, -math.pi/4));
client.simSetCameraOrientation(1, airsim.to_quaternion(0, 0, math.pi/4));

LEFT_CAM = 2
RIGHT_CAM = 1
CENTER_CAM = 0

def get_road_image(client,camera_position):
	response = client.simGetImages([airsim.ImageRequest(str(camera_position), airsim.ImageType.Scene, False, False)])
	response = response[0]
	# get numpy array
	img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 
	# reshape array to 4 channel image array H X W X 4
	img_rgba = img1d.reshape(response.height, response.width, 3)
	img_rgba = img_rgba[50:144, :, :]

	return img_rgba

def image_auxiliary(client, steering_last):

	left_img = get_road_image(client,LEFT_CAM)
	right_img = get_road_image(client,RIGHT_CAM)

    # print('left camera prediction: ', predict.test(left_img), '......', 'right camera prediction: ', predict.test(right_img))
	# save compute time if detect offroad, next time we will detect that direction only
	if steering_last[-1] == 0.5 and sum(steering_last[0:2]) != 1:
		left_pred = predict.test(left_img)
		right_pred = 'onroad'
	elif steering_last[-1] == -0.5 and sum(steering_last[0:2]) != -1:
		left_pred = 'onroad'	
		right_pred = predict.test(right_img)
	else:	
		left_pred = predict.test(left_img)
		right_pred = predict.test(right_img)

	control_flag = 1
	steering = 0
	if left_pred == 'onroad' and right_pred == 'offroad':
		print('vehicle drives out of road, please turn ###left###', end='')
		steering = -0.5
		# car_controls.throttle = 0.8
		
	elif left_pred == 'offroad' and right_pred == 'onroad':
		print('vehicle drives out of road, please turn ###right###', end='')
		steering = 0.5
		# car_controls.throttle = 0.8
		
	elif left_pred == 'offroad' and right_pred == 'offroad':
		print('vehicle drives out of road, please ###slow down###', end='')
		# car_controls.throttle = -1
		# car_controls.steering = 0
		control_flag = 2
	else:
		print('vehicle is driving on the road', end='')
		control_flag = 0
    
	return steering, control_flag

print('loading mymodle.pth .....')

steering_last = [0,0,0]
control_flag_last = 0
while True:
	# left camera = 2 right camera = 1
	# client.getCarState()
	detection_start_time = time.time()
	steering, control_flag = image_auxiliary(client,steering_last)
	steering_last[-1] = steering

	if sum(steering_last[0:2]) == 1 and steering == 0.5:
		control_flag = 2
		print('   ,enough for turning right!')
	if sum(steering_last[0:2]) == -1 and steering == -0.5:
		control_flag = 2
		print('   ,enough for turning left!')
	print(control_flag)
	flag = open('flag.txt','w')
	if control_flag == 1:
		flag.write('1\n')        
		flag.write(str(steering))
	elif control_flag == 2:
		flag.write('2\n')        
	else:
		flag.write('0')

	if len(steering_last)>3:
		steering_last = []
		control_flag_last =[]
		
	steering_last[0:2] = steering_last[1:3]
	control_flag_last = control_flag

	detection_end_time = time.time()
	detection_time = detection_end_time - detection_start_time
	print('   ,prediction_time: ', detection_time)