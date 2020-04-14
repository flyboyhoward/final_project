import airsim
import time
import os
import numpy as np
import math
import path_planning

# connect to the AirSim simulator 
client = airsim.CarClient()
client.confirmConnection()
# client.enableApiControl(False, "Car1")
# car_controls = airsim.CarControls()
# client.reset()
LEFT = -1
RIGHT = 1
STRAIGHT = 0

def init_road_points():
    road_points = []
    with open('road_turn_points.txt', 'r') as f:
        for line in f:
            points = line.split('\t')
            first_point = np.array([float(p) for p in points[0].split(',')])

            road_points.append(tuple(first_point))
            
    return road_points

def getCarPosition(car_state):
    x_val = car_state.kinematics_estimated.position.x_val
    y_val = car_state.kinematics_estimated.position.y_val
    z_val = car_state.kinematics_estimated.position.z_val
    return x_val, y_val, z_val

def getCarOrientation(car_state):

    _, _, arc = airsim.to_eularian_angles(car_state.kinematics_estimated.orientation)

    if arc < 0:
        arc += 2*math.pi
    return arc/(2*math.pi) * 360

def getRelativeAngle(line_orientation, car_orientation):
    """得到车道线与车头之间的夹角,后续需要改进
    """
    return (car_orientation - line_orientation)

def throttleControl(car_state,set_speed):
    car_controls = airsim.CarControls()
    if (car_state.speed < set_speed):
        car_controls.throttle = 0.7
    else:
        car_controls.throttle = 0
    return car_controls

def image_auxiliary(car_control):
    flags = open('flag.txt','r')
    flag1 = flags.readline()
    if flag1 == '1\n':
        flag2 = flags.readline()
        if float(flag2) < 0:
            if car_control.steering >= 0:
                car_control.steering = -0.4
                car_control.throttle = 0
            else:
                pass
        elif float(flag2) > 0:
            if car_control.steering <= 0:
                car_control.steering = 0.4
                car_control.throttle = 0
            else:
                pass
    else:
        car_control = car_control

    return car_control

def get_heading_angle(client):
    car_state = client.getCarState()
    car_orientation = getCarOrientation(car_state)
    heading_angle = 0

    if 315 < car_orientation <= 45:
        #heading north
        heading_angle = 0
    elif 45 < car_orientation <= 135:
        #heading east
        heading_angle = 90
    elif 135 < car_orientation <= 225:
        #heading south
        heading_angle = 180
    elif 225 < car_orientation<= 315:
        #heading west
        heading_angle = 270

    return heading_angle

def get_closest_roadpoint(client,road_points):
    distance = []
    road_point = list(road_points)
    car_state = client.getCarState()
    x_val, y_val, z_val = getCarPosition(car_state)
    for x_point, y_point in road_point:
        distance.append(abs(x_point - x_val) + abs(y_point - y_val))
    index = np.argmin(distance)
    closest_roadpoint = road_point[index]
    
    return closest_roadpoint, index

def turnControl(client, turning_point, turning_direction):
    set_speed = 7  # 40km/s

    heading_angle = get_heading_angle(client)

    if turning_direction == LEFT and heading_angle == 0:
        heading_angle = 360
    #set target angle
    if turning_direction == LEFT:
        target_angle = heading_angle - 90
        turn_flag = True
        straight_flag = False
    if turning_direction == RIGHT:
        target_angle = heading_angle + 90
        turn_flag = True
        straight_flag = False
    if turning_direction == STRAIGHT:
        target_angle = heading_angle
        turn_flag = False
        straight_flag = True



    while turn_flag:
        # 1.控制车速
        car_state = client.getCarState()
        car_controls = throttleControl(car_state,set_speed)
        # client.setCarControls(car_controls)

        # 2.根据位置控制方向盘
        car_state = client.getCarState()
        x_val, y_val, z_val = getCarPosition(car_state)
        car_orientation = getCarOrientation(car_state)

        # target angle
        relative_angle = getRelativeAngle(target_angle, car_orientation)

        if abs(relative_angle) <= 5:
            turn_flag = False
            break
        else: 
            turn_flag = True

        if abs(x_val - turning_point[0]) < 10 and abs(y_val - turning_point[1]) < 10 and turning_direction == LEFT:
            car_controls.steering = -0.4

        elif abs(x_val - turning_point[0]) < 10 and abs(y_val - turning_point[1]) < 10 and turning_direction == RIGHT:
            car_controls.steering = 0.4
        else:
            car_controls.steering = 0
            turn_flag = False
        # detect road edges
        car_controls = image_auxiliary(car_controls)
        client.setCarControls(car_controls)
        # 3.打印位置信息
        print("turning: postion: %0.2f, %0.2f, throttle: %0.2f, steering: %0.2f, angle: %0.2f ,relative_angle: %0.2f" %\
             (x_val, y_val, car_controls.throttle, car_controls.steering, car_orientation,relative_angle))

    while straight_flag:
        # 1.控制车速
        car_state = client.getCarState()
        car_controls = throttleControl(car_state,set_speed)
        # client.setCarControls(car_controls)

        # 2.根据位置控制方向盘
        car_state = client.getCarState()
        x_val, y_val, z_val = getCarPosition(car_state)
        car_orientation = getCarOrientation(car_state)

        relative_angle = getRelativeAngle(heading_angle, car_orientation)
        if relative_angle > 180:
            relative_angle -= 360

        if heading_angle == 0 or heading_angle == 180:
            #heading north or south
            ref = turning_point[1]
            val = y_val
        else:
            #heading east or west
            ref = turning_point[0]
            val = x_val

        k_relative_angle = 20
        distance = abs(val-ref)
        k_distance = 15.0

        para_a = distance/k_distance
        para_b = relative_angle/k_relative_angle
        if val > ref: #车在中线右边
            if relative_angle > 0:
                car_controls.steering = - para_a - para_b
            elif relative_angle < 0:
                car_controls.steering = - para_a - para_b
        elif val < ref: #车在中线左边
            if relative_angle > 0:
                car_controls.steering = para_a - para_b
            elif relative_angle < 0:
                car_controls.steering = para_a - para_b
        # determine car postion is with in the control area
        if abs(x_val - turning_point[0]) < 10 and abs(y_val - turning_point[1]) < 10:
            straight_flag = True
        else:
            straight_flag = False
            

        client.setCarControls(car_controls)

        # 3.打印位置信息
        # print("going staight: postion: %0.2f, %0.2f, throttle: %0.2f, steering: %0.2f, relative_angle: %0.2f, para_a: %0.2f, para_b: %0.2f"  %\
        #      (x_val, y_val, car_controls.throttle, car_controls.steering, relative_angle, para_a, para_b))

def get_action(current_point, next_point,nmap):
    
    x = next_point[0] - current_point[0]
    y = next_point[1] - current_point[1]
    # left up
    if  x == -1 and y == -1:
        if nmap[current_point[0] - 1][current_point[1]] == 0: # value of near up right 
            action = LEFT
        elif nmap[current_point[0]][current_point[1]-1] == 0: # value of near down left
            action = RIGHT
    # right up
    elif x == -1 and y == 1:
        if nmap[current_point[0] - 1][current_point[1]] == 0: # value of near up left 
            action = RIGHT
        elif nmap[current_point[0]][current_point[1]+1] == 0: # value of near down right 
            action = LEFT
    # left down
    elif  x == 1 and y == -1:
        if nmap[current_point[0] + 1][current_point[1]] == 0: # value of near down right 
            action = RIGHT
        elif nmap[current_point[0]][current_point[1]-1] == 0: # value of near up left
            action = LEFT
    # right down
    elif  x == 1 and y == 1:
        if nmap[current_point[0] + 1][current_point[1]] == 0: # value of near down left 
            action = LEFT
        elif nmap[current_point[0]][current_point[1]+1] == 0: # value of near up right
            action = RIGHT
    else:
        action = STRAIGHT

    return action

def init_path_planning(start_point, goal_point):
    path_in_environment = []
    nmap = np.load('map_point.npy')
    road_points = init_road_points()
    path = path_planning.get_path(start_point, goal_point)
    path_in_list = [list(paths) for paths in path]
    for idx in range(len(path_in_list)-2):
        if path_in_list[idx][1] == path_in_list[idx+2][1] and path_in_list[idx][1] != path_in_list[idx+1][1]:
            path_in_list[idx + 1][1] = path_in_list[idx][1]
        elif path_in_list[idx][0] == path_in_list[idx+2][0] and path_in_list[idx][0] != path_in_list[idx+1][0]:
            path_in_list[idx + 1][0] = path_in_list[idx][0]
        else:
            pass
    # project point to environment
    for point in path_in_list:
        point = [-point[0]*10 + 130, point[1]*10 - 130]
        path_in_environment.append(point)
    return nmap, road_points, path_in_list, path_in_environment

def execute_turns(client, nmap, road_points, path_in_list, path_in_environment):
    
    closest_point, index = get_closest_roadpoint(client,path_in_environment)

    if index < len(path_in_list)-1:
        print(index, len(path_in_list))
        next_pathpoint = path_in_list[index + 1]
        current_pathpoint = path_in_list[index]
        action = get_action(current_pathpoint, next_pathpoint, nmap)
        current_closest_roadpoint, index = get_closest_roadpoint(client,road_points)


        car_state = client.getCarState()
        x_val, y_val, z_val = getCarPosition(car_state)
        # if abs(x_val - current_closest_roadpoint[0]) <= 10 and abs(y_val - current_closest_roadpoint[1]) <= 10:
        #     pass
        # else:
        #     current_closest_roadpoint = last_closest_roadpoint

        # last_closest_roadpoint = current_closest_roadpoint

        if abs(x_val - current_closest_roadpoint[0]) <= 10 and abs(y_val - current_closest_roadpoint[1]) <= 10:
            turnControl(client, current_closest_roadpoint, action)
            print('take action: ',action)
        else:
            pass
        # print(current_pathpoint)
        finish = False

    else:
        finish = True
        print('Congratualtions!!!!!!')
        print('You have successfully reach your target point!')
        print('Would you like another test??')
        # print(path_in_environment)
    return finish


if __name__ == "__main__":
    client = airsim.CarClient()
    client.confirmConnection()
    print('Connect succcefully！')
    client.enableApiControl(True)
    nmap, road_points, path_in_list, path_in_environment = init_path_planning([25,13],[4,0])
    print(path_in_list)
    finish = False
    while not finish:
        car_state = client.getCarState()
        x_val, y_val, z_val = getCarPosition(car_state)
        print(x_val)
        finish = execute_turns(client, nmap, road_points, path_in_list, path_in_environment)