import os
import sys
import airsim
import numpy as np
import road_turns as turn
flags = open('flag.txt','r')
flag = flags.readline()
print(flag == '1\n')
flag = flags.readline()
print(flag)
def read_flag_text(car_control):
	flags = open('flag.txt','r')
	flag1 = flags.readline()
	if flag1 == str(1):
		car_control.steering = float()
	else:
		pass

	return car_control

def init_road_points():
    road_points = []
    car_start_coords = [12961.722656, 6660.329102, 0]
    with open('test_road_lines.txt', 'r') as f:
        for line in f:
            points = line.split('\t')

            first_point = np.array([float(p) for p in points[0].split(',')] + [0])
            second_point = np.array([float(p) for p in points[1].split(',')] + [0])
            third_point = np.array([float(p) for p in points[2].split(',')] + [0])
            road_points.append(tuple((first_point, second_point, third_point)))
            

    return road_points
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(False)
X = [-127.6,0.4,80.4,128.4]
Y = [-129,-1,127]
def get_roadval(client):
    heading_angle = turn.get_heading_angle(client)
    car_position = client.getCarState().kinematics_estimated.position
    if heading_angle == 0 or heading_angle == 180:
        distance = [abs(car_position.y_val - y) for y in Y] 
        road_val = Y[np.argmin(distance)]
    elif heading_angle == 90 or heading_angle == 270:
        distance = [abs(car_position.x_val - x) for x in X] 
        road_val = X[np.argmin(distance)]
    return road_val
while 1:
	road_val = get_roadval(client)
	print(road_val)

nmap = [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0],
		[0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0],
		[0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0],
		[0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0],
		[0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0],
		[0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 0],
		[0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0],
		[0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0],
		[0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0],
		[0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0],
		[0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0],
		[0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0],
		[0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0],
		[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0],
		[0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0],
		[0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0],
		[0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0],
		[0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0],
		[0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0],
		[0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0],
		[0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0],
		[0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0],
		[0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0],
		[0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0],
		[0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0],
		[0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0],
		[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0],]