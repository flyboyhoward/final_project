import airsim 
import pprint
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import math
import json

import road_turns as turn

client = airsim.CarClient()
client.confirmConnection()
# client.enableApiControl(False)

def init_road_points():
    road_points = []

    with open('road_lines.txt', 'r') as f:
        for line in f:
            points = line.split('\t')
            first_point = np.array([float(p) for p in points[0].split(',')] )
            second_point = np.array([float(p) for p in points[1].split(',')] )
            road_points.append(tuple((first_point, second_point)))

    return road_points

def init_vehicle_setting():
    vehicle_quantity = 0
    vehicle_starting_points = []
    with open("C:\\Users\\flybo\\OneDrive\\Documents\\AirSim\\settings.json",encoding='utf-16', errors='ignore') as json_file:
        data = json.load(json_file, strict=False)
    vehicle_names = [key for key,value in data['Vehicles'].items()]
    vehicle_quantity = len(vehicle_names)
    for vehicle_name in vehicle_names:
        vehicle_starting_points.append([data['Vehicles'][vehicle_name]['X'],data['Vehicles'][vehicle_name]['Y']])
    return vehicle_names, vehicle_quantity, vehicle_starting_points

def show_map(clinet, path):
    path = list(path)
    car_names, car_quantity, car_starting_points = init_vehicle_setting()

    road_points = init_road_points()

    fig, ax = plt.subplots(figsize=(10,10))
    for point in road_points:
        ax.plot([point[0][1], point[1][1]], [point[0][0], point[1][0]], 'k-', alpha=0.5, lw=20)
    if path == []:
        pass
    else:
        for path_point in path:
            ax.plot(path_point[1],path_point[0], 'go',alpha = 0.7)

    line, = ax.plot([], [], 'bo', animated=True)

    def update_line(i):
        x_val = []
        y_val = []
        for indexs, car_name in enumerate(car_names):
            car_state = client.getCarState(car_name)
            x_val.append(car_state.kinematics_estimated.position.x_val + car_starting_points[indexs][0])
            y_val.append(car_state.kinematics_estimated.position.y_val + car_starting_points[indexs][1])
        plt.setp(line, 'ydata', x_val, 'xdata', y_val)
        return [line]


    ani = FuncAnimation(fig, update_line, blit=True, interval=25, frames=1000)
    plt.show()

if __name__ == '__main__':
    nmap, road_points, path_in_list, path_in_environment = turn.init_path_planning([13,13],[0,26])
    show_map(client, path_in_environment)
