import tensorflow as tf
import setup_path
import airsim
from collections import deque

import random
import numpy as np
import time
import os
import pickle
import math
import sys
# import road_detection as predict
import road_turns as turn
import show_map

# basic setting
ACTION_NUMS = 13        # number of valid actions
GAMMA = 0.99            # decay rate of past observations
OBSERVE = 50            # time steps to observe before training
EXPLORE = 20000.        # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.1   # starting value of epsilon
EPSILON_DECAY_START = 20
MEMORY_SIZE = 20000     # number of previous transitions to remember
MINI_BATCH = 32         # size of mini batch
MAX_EPISODE = 20000
DEPTH_IMAGE_WIDTH = 256
DEPTH_IMAGE_HEIGHT = 144

TAU = 0.001             # Rate to update target network toward primary network
flatten_len = 9216      # the input shape before full connect layer
NumBufferFrames = 4     # take the latest 4 frames as input

LEFT_CAM = 2                # front left camera ID
RIGHT_CAM = 1               # front right camera ID
CENTER_CAM = 0              # front center camera ID

LEFT = -1                   # turn left action 
RIGHT = 1                   # turn rgiht action 
STRAIGHT = 0                # go straight action 

LOOPTIME = 0.7              # define loop time of each control loop

X = [-127.6,0.4,80.4,128.4]
Y = [-129,-1,127]

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name="weights")

def bias_variable(shape):
    initial = tf.constant(0., shape=shape)
    return tf.Variable(initial, name="bias")

def conv2d(x, W, stride_h, stride_w):
    return tf.nn.conv2d(x, W, strides=[1, stride_h, stride_w, 1], padding="SAME")

class Deep_Q_Network(object):
    """docstring for ClassName"""
    def __init__(self, sess):
        # network weights and biases
        # input 144x256x4
        with tf.name_scope("Conv1"):
            W_conv1 = weight_variable([8, 8, NumBufferFrames, 32])
            variable_summaries(W_conv1)
            b_conv1 = bias_variable([32])
        with tf.name_scope("Conv2"):
            W_conv2 = weight_variable([4, 4, 32, 64])
            variable_summaries(W_conv2)
            b_conv2 = bias_variable([64])
        with tf.name_scope("Conv3"):
            W_conv3 = weight_variable([3, 3, 64, 64])
            variable_summaries(W_conv3)
            b_conv3 = bias_variable([64])
        with tf.name_scope("Value_Dense"):
            W_value = weight_variable([flatten_len, 512])
            variable_summaries(W_value)
            b_value = bias_variable([512])
        with tf.name_scope("FCAdv"):
            W_adv = weight_variable([flatten_len, 512])
            variable_summaries(W_adv)
            b_adv = bias_variable([512])
        with tf.name_scope("FCValueOut"):
            W_value_out = weight_variable([512, 1])
            variable_summaries(W_value_out)
            b_value_out = bias_variable([1])
        with tf.name_scope("FCAdvOut"):
            W_adv_out = weight_variable([512, ACTION_NUMS])
            variable_summaries(W_adv_out)
            b_adv_out = bias_variable([ACTION_NUMS])
        # input layer
        self.state = tf.placeholder("float", [None, DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH, NumBufferFrames])
        # Conv1 layer
        h_conv1 = tf.nn.relu(conv2d(self.state, W_conv1, 8, 8) + b_conv1)
        # Conv2 layer
        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2, 2) + b_conv2)
        # Conv2 layer
        h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1, 1) + b_conv3)
        h_conv3_flat = tf.layers.flatten(h_conv3)
        # FC ob value layer
        h_fc_value = tf.nn.relu(tf.matmul(h_conv3_flat, W_value) + b_value)
        value = tf.matmul(h_fc_value, W_value_out) + b_value_out
        # FC ob adv layer
        h_fc_adv = tf.nn.relu(tf.matmul(h_conv3_flat, W_adv) + b_adv)
        advantage = tf.matmul(h_fc_adv, W_adv_out) + b_adv_out
        # Q = value + (adv - advAvg)
        advAvg = tf.expand_dims(tf.reduce_mean(advantage, axis=1), axis=1)
        advIdentifiable = tf.subtract(advantage, advAvg)
        self.readout = tf.add(value, advIdentifiable)
        # define the cost function
        self.actions = tf.placeholder("float", [None, ACTION_NUMS])
        self.y = tf.placeholder("float", [None])
        self.readout_action = tf.reduce_sum(tf.multiply(self.readout, self.actions), axis=1)
        self.td_error = tf.square(self.y - self.readout_action)
        self.cost = tf.reduce_mean(self.td_error)
        self.train_step = tf.train.AdamOptimizer(1e-5).minimize(self.cost)

def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0:total_vars // 2]):
        op_holder.append(tfVars[idx + total_vars // 2].assign(
            (var.value() * tau) + ((1 - tau) * tfVars[idx + total_vars // 2].value())))
    return op_holder

def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)

def get_image(client,image_type):
    if (image_type == 'Scene'):
        # get image center front
        responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        img_rgba = img1d.reshape(response.height, response.width, 3)
        img_rgba = np.flipud(img_rgba)
        observation = img_rgba[:, :, 0:3]

    elif (image_type == 'DepthPlanner'):
        try:
            responses = client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.DepthPlanner, pixels_as_float=True)])
            response = responses[0]
            img1d = np.array(response.image_data_float, dtype=np.float)
            img1d = img1d * 3.5 + 30
            img1d[img1d > 255] = 255
            img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
            observation = img2d
        except:
            print('######### Error: I can not get a depth image correctly! #########################' )
            observation = np.ones([DEPTH_IMAGE_HEIGHT,DEPTH_IMAGE_WIDTH])
    else:
        observation = None
    observation_size = np.shape(observation)
    if(observation_size[0]==DEPTH_IMAGE_HEIGHT and observation_size[1]==DEPTH_IMAGE_WIDTH):
        return observation
    else:
        print('######### Error: The depth image shape: ',observation_size)
        return np.ones([DEPTH_IMAGE_HEIGHT,DEPTH_IMAGE_WIDTH])

def get_road_image(client,camera_position):
    response = client.simGetImages([airsim.ImageRequest(str(camera_position), airsim.ImageType.Scene, False, False)])
    response = response[0]
    # get numpy array
    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 
    img_rgba = img1d.reshape(response.height, response.width, 3)
    # crop image
    img_rgba = img_rgba[50:144, :, :]

    return img_rgba


def parse_lidarData(data):

    # reshape array of floats to array of [X,Y,Z]
    points = np.array(data.point_cloud, dtype=np.dtype('f4'))
    points = np.reshape(points, (int(points.shape[0]/3), 3))
   
    return points

def lidar_auxiliary(client,car_controls):
    lidarDataLeft = client.getLidarData("LidarSensorLeft","Car1")
    lidarDataRight = client.getLidarData("LidarSensorRight","Car1")

    if(len(lidarDataLeft.point_cloud)<3) or (len(lidarDataRight.point_cloud)<3):
        print("\tNo points received from Lidar data")
    else:
        points_left = parse_lidarData(lidarDataLeft)
        points_right = parse_lidarData(lidarDataRight)
        y_points_left, x_points_left = np.array(points_left[:, 1]),np.array(points_left[:, 0])
        y_points_right, x_points_right = np.array(points_right[:, 1]),np.array(points_right[:, 0])
        len_points_left = len(y_points_left)
        len_points_right = len(y_points_right)
        points_euler_distance_left = np.sqrt(y_points_left**2 + x_points_left**2)
        points_euler_distance_right = np.sqrt(y_points_right**2 + x_points_right**2)
        obstacle_index_left = np.argwhere(points_euler_distance_left <= 9)
        obstacle_index_right = np.argwhere(points_euler_distance_right <= 9)
        len_index_left = len(obstacle_index_left)
        len_index_right = len(obstacle_index_right)
        # print(x_points_left)
        if len_index_left > len_points_left/4 and len_index_right < len_points_right/4:
            if(car_controls.steering<=0):
                car_controls.steering = 0.5
                print('obstacles on the left, please turn ... right ...!')
            else:
                print('operate correctly')

        elif len_index_right > len_points_right/4 and len_index_left < len_points_left/4:
            if(car_controls.steering>=0):
                car_controls.steering = -0.5
                print('obstacles on the right, please turn ... left ...!')
            else:
                print('operate correctly')

        elif len_index_right > len_points_right/4 and len_index_left > len_points_left/4:
            left_distance_average = np.sum(points_euler_distance_left[obstacle_index_left])/len_index_left
            right_distance_average = np.sum(points_euler_distance_right[obstacle_index_right])/len_index_right
            
            if left_distance_average - right_distance_average < -0.4:
                if(car_controls.steering<=0):
                    car_controls.steering = 0.5
                    print('obstacles ahead, please turn ~~~ right ~~~!')
                else:
                    print('operate correctly')
            elif left_distance_average - right_distance_average > 0.4:
                if(car_controls.steering>=0):
                    car_controls.steering = -0.5
                    print('obstacles ahead, please turn ~~~ left ~~~!')
                else:
                    print('operate correctly')
            else:
                print('obstacles ahead, follow DQN action!')

        else:
            print('operate correctly')

    return car_controls

def image_auxiliary(car_control):
    flags = open('flag.txt','r')
    flag1 = flags.readline()
    reset = False
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
    elif flag1 == '2\n':
        reset = True
    else:
        car_control = car_control

    return car_control,reset

def get_roadval(client, heading_angle):
    car_position = client.getCarState().kinematics_estimated.position
    if heading_angle == 0 or heading_angle == 180:
        distance = [abs(car_position.y_val - y) for y in Y] 
        road_val = Y[np.argmin(distance)]
    elif heading_angle == 90 or heading_angle == 270:
        distance = [abs(car_position.x_val - x) for x in X] 
        road_val = X[np.argmin(distance)]
    return road_val

def env_feedback(client,last_position,goal_point):
    terminal_position = list(goal_point)
    reset = False
    collision_info = client.simGetCollisionInfo()
    car_position = client.getCarState().kinematics_estimated.position
    distance = np.sqrt((car_position.x_val-terminal_position[0])**2+(car_position.y_val-terminal_position[1])**2)
    current_position = [car_position.x_val,car_position.y_val]
    heading_angle = turn.get_heading_angle(client)
    road_val = get_roadval(client,heading_angle)
    # if not -136<car_position.x_val<137 and -136<car_position.y_val<137:
    #     reward = -100
    #     reset = True
    if collision_info.has_collided:
        reward = -100
        reset = True
    else:
        pre_distance = np.sqrt((last_position[0]-terminal_position[0])**2+(last_position[1]-terminal_position[1])**2)
        if(distance<=5):
            reward = 10
        elif(distance<pre_distance):
            reward = 5 / distance
        else: # drive further, should be negative here
            reward = 5 / distance
        # reward based on the distance to the middle line of current road
        if heading_angle == 0 or heading_angle == 180:
            distance_middle = abs(current_position[1] - road_val) 
            if distance_middle<= 2.5:
                reward += 3/(2.5 + distance_middle)
            elif 2.5 < distance_middle <= 4:
                reward -= 0.03*distance_middle
            else:
                reward -= 0.2*distance_middle  
        elif heading_angle == 90 or heading_angle == 270:
            distance_middle = abs(current_position[0] - road_val) 
            if distance_middle <= 2.5:
                reward += 3/(2.5 + distance_middle)
            elif 2.5 < distance_middle <= 4:
                reward -= 0.03*distance_middle
            else:
                reward -= 0.2*distance_middle

    if (distance <= 5):
        terminal = 1
        reset = True
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    else:
        terminal = 0
    return current_position,distance,reward, terminal,reset

def store_transition(replay_experiences,store_or_read):
    store_path = 'replay_experiences.pkl'
    if(store_or_read=='read'):
        if not os.path.exists(store_path) or os.path.getsize(store_path)==0:
            print('Not Found the pkl file!')
            return replay_experiences
        else:
            store_file = open(store_path,'rb')
            replay_experiences = pickle.load(store_file)
            store_file.close()
            memory_len = len(replay_experiences)
            print('Successfully load the replay_experiences.pkl, %05d memory'%memory_len)
            return replay_experiences
    elif(store_or_read=='store'):
        store_file = open(store_path, 'wb')
        pickle.dump(replay_experiences, store_file)
        store_file.close()
        return 1
    else:
        return 0

def excute_action(client,car_controls,steer):
    car_controls.steering = steer
    car_speed = client.getCarState().speed
    straight_range = [-0.1,0.1]
    # when swerved wildly, slow down
    BigTurn_range = [-0.6,0.6]
    if(steer>=straight_range[0] and steer<=straight_range[1]):
        car_controls.throttle = car_controls.throttle + 0.2
        car_controls.throttle = 2 if car_controls.throttle>=2 else car_controls.throttle
    elif(steer<=BigTurn_range[0] or steer>=BigTurn_range[1]):
        car_controls.throttle = 0.5
    else:
        car_controls.throttle = 1
    car_controls.throttle = 0 if car_speed>=7 else car_controls.throttle
    return car_controls

def print_action(episode,t,car_controls,distance,steer,reward):
    print('Episode:%05d,Step:%05d '%(episode,t),end='')
    if(steer<0):
        print('The car is turning left    , ',end='')
    elif(steer>0):
        print('The car is turning right   , ',end='')
    else:
        print('The car is going straightly, ',end='')
    print('throttle=%.1f,  steer=%.3f,  reward=%.2f, distance=%.2f'%(car_controls.throttle,steer,reward,distance),end='')


def init_road_points():
    road_points = []
    with open('test_road_lines.txt', 'r') as f:
        for line in f:
            points = line.split('\t')
            first_point = np.array([float(p) for p in points[0].split(',')] + [0.0])
            second_point = np.array([float(p) for p in points[1].split(',')] + [0.0])
            third_point = np.array([float(p) for p in points[2].split(',')] + [0])
            road_points.append(tuple((first_point, second_point, third_point)))
            
    return road_points

def get_next_starting_point(car_client, road_points):
        random_direction = ()
        # Get the current state of the vehicle
        car_state = car_client.getCarState()

        # Pick a random road.
        random_line_index = np.random.randint(0, high=len(road_points))

        # Compute the starting point of the car
        random_line = road_points[random_line_index]
        random_start_point = list(random_line[0])
        print('random_start_point',str(random_start_point))

        # Compute the direction that the vehicle will face
        # Vertical line
        if (np.isclose(random_line[0][1], random_line[1][1])):
            if random_line[0][0] - random_line[1][0] <= 0:
                random_direction = (0,0,0)
            else:
                random_direction = (0, 0, math.pi)
        # Horizontal line
        elif (np.isclose(random_line[0][0], random_line[1][0])):
            if random_line[0][1] - random_line[1][1] <= 0:
                random_direction = (0,0,math.pi/2)
            else:
                random_direction = (0,0,-1.0 * math.pi/2)
        # else:
        #     random_direction = (0,0,0)
        # The z coordinate is always zero
        random_start_point[2] = -0
        random_goal_point = list(random_line[2])
        return (random_start_point, random_direction, random_goal_point)

def trainNetwork(client, car_controls):

    car_controls.throttle = 0.5
    car_controls.steering = 0
    client.reset()
    print('Environment initialized!')
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    # sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    sess = tf.InteractiveSession()
    with tf.name_scope("OnlineNetwork"):
        online_net = Deep_Q_Network(sess)
    with tf.name_scope("TargetNetwork"):
        target_net = Deep_Q_Network(sess)
    time.sleep(1)

    reward_var = tf.Variable(0., trainable=False)
    tf.summary.scalar('reward', reward_var)
    # define summary
    merged_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('./logs', sess.graph)
    # Initialize the buffer
    replay_experiences = deque()
    replay_experiences = store_transition(replay_experiences,'read')
    # get the first state
    observe_init = get_image(client,'DepthPlanner')
    state_pre = np.stack((observe_init, observe_init, observe_init, observe_init), axis=2)
    # saving and loading networks
    trainables = tf.trainable_variables()
    trainable_saver = tf.train.Saver(trainables)
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state("D:\\finalproject\\Obstacle-Avoidance-in-AirSim\\saved_networks")
    print('checkpoint:', checkpoint)
    if checkpoint and checkpoint.model_checkpoint_path:
        trainable_saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        if not os.path.exists("D:\\finalproject\\Obstacle-Avoidance-in-AirSim\\saved_networks\\new_model_lidar"):
            os.mkdir("D:\\finalproject\\Obstacle-Avoidance-in-AirSim\\saved_networks\\new_model_lidar")
            print('The file not exists, is created successfully')
        print("Could not find old network weights")

    #load road points and target point
    road_points = init_road_points()
    goal_point = [128.4, -1]
    # start training
    episode = 1
    epsilon = INITIAL_EPSILON
    print('Number of trainable variables:', len(trainables))
    targetOps = updateTargetGraph(trainables, TAU)
    inner_loop_time_start = time.time()
    while episode < MAX_EPISODE:
        reward_episode = 0.
        terminal = 0
        step = 1
        reset = False

        loop_start_time = time.time()
        car_position = client.getCarState().kinematics_estimated.position
        last_position = [car_position.x_val,car_position.y_val]

        car_controls.brake = 0
        client.setCarControls(car_controls)

        flag = open('flag.txt','w')
        flag.write('0')

        while not reset:

            image = get_image(client, 'Scene')
            observe = get_image(client,'DepthPlanner')
            observe = np.reshape(observe, (DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH, 1))
            state_current = np.append(image, observe, axis = 2)
            # state_current = np.append(observe, state_pre[:, :, :(NumBufferFrames - 1)], axis=2)
            current_position,distance,reward_current, terminal,reset = env_feedback(client,last_position,goal_point)
            # prevent vehicle driving out of the map
            car_state = client.getCarState()
            x_val = car_state.kinematics_estimated.position.x_val
            y_val = car_state.kinematics_estimated.position.y_val
            print(x_val,y_val)
            print('####################################')
            if -137<x_val<137 and -137<y_val<137:
                # store the experience
                if step-1 > 0:
                    replay_experiences.append((state_pre, action_current, reward_current, state_current, terminal))
                    if len(replay_experiences) > MEMORY_SIZE:
                        replay_experiences.popleft()
                state_pre = state_current
                # choose an action epsilon greedily
                actions = sess.run(online_net.readout, feed_dict={online_net.state: [state_current]})
                readout_t = actions[0]
                action_current = np.zeros([ACTION_NUMS])
                # fill the reply experience
                if len(replay_experiences) <= OBSERVE:
                    action_index = random.randrange(ACTION_NUMS)
                    print('episode=%05d,step=%05d,we are observing the env,the action is random......'%(episode,step))
                    action_current[action_index] = 1
                else:
                    if random.random() <= epsilon:
                        print("----------Random Action----------")
                        action_index = random.randrange(ACTION_NUMS)
                        action_current[action_index] = 1
                    else:
                        action_index = np.argmax(readout_t)
                        action_current[action_index] = 1
                # Control the agent
                side_num = int(ACTION_NUMS - 1) // 2
                steer = float((action_index - side_num) / side_num)

                car_controls = excute_action(client,car_controls,steer)
                car_controls, offroad = image_auxiliary(car_controls)
                car_controls = lidar_auxiliary(client,car_controls)
                client.setCarControls(car_controls)

                if offroad:
                    reward_current -= 10
                # overwirte flag.txt again if reset  
                if reset:
                    flag = open('flag.txt','w')
                    flag.write('0')

                inner_loop_time_end = time.time()
                inner_loop_time = inner_loop_time_end-inner_loop_time_start

                if inner_loop_time <= LOOPTIME:
                    time_to_sleep = LOOPTIME - inner_loop_time
                    time.sleep(time_to_sleep)
                else:
                    pass
                inner_loop_time_end = time.time()
                
                print_action(episode, step, car_controls, distance, steer, reward_current)
                print(',experience len=%05d'%len(replay_experiences),end='')
                print(',inner loop=%.4fs'%(inner_loop_time_end-inner_loop_time_start))

                inner_loop_time_start = time.time()
                last_position = current_position

                if episode > OBSERVE:
                    # # sample a minibatch to train on
                    minibatch = random.sample(replay_experiences, MINI_BATCH)
                    y_batch = []
                    # get the batch variables
                    state_pre_batch = [d[0] for d in minibatch]
                    actions_batch = [d[1] for d in minibatch]
                    rewards_batch = [d[2] for d in minibatch]
                    state_current_batch = [d[3] for d in minibatch]
                    Q1 = online_net.readout.eval(feed_dict={online_net.state: state_current_batch})
                    Q2 = target_net.readout.eval(feed_dict={target_net.state: state_current_batch})
                    for i in range(0, len(minibatch)):
                        terminal_batch = minibatch[i][4]
                        # if terminal, only equals reward
                        if terminal_batch:
                            y_batch.append(rewards_batch[i])
                        else:
                            y_batch.append(rewards_batch[i] + GAMMA * Q2[i, np.argmax(Q1[i])])

                    # Update the network with our target values.
                    online_net.train_step.run(feed_dict={online_net.y: y_batch,
                                                         online_net.actions: actions_batch,
                                                         online_net.state: state_pre_batch})
                    updateTarget(targetOps, sess)  # Set the target network to be equal to the primary network.

                reward_episode = reward_episode + reward_current
                step = step+1
                # scale down epsilon
                if epsilon > FINAL_EPSILON and episode > EPSILON_DECAY_START:
                    epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            else:
                reset = True
                reward_current -= 100
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                print('vehicle drives out of boundary')
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        # save progress every 20 episodes and write summaries
        if (episode % 5 == 0):
            trainable_saver.save(sess, "D:\\finalproject\\Obstacle-Avoidance-in-AirSim\\saved_networks\\Simply_maze",global_step=episode,write_state=True)
            print('######## The mode has been saved successfully after %d episodes ##########' % episode)
            #  write summaries
            summary_str = sess.run(merged_summary, feed_dict={reward_var: reward_episode})
            summary_writer.add_summary(summary_str, episode)
        if(len(replay_experiences)<2500):
            signal_back = store_transition(replay_experiences, 'store')
            if signal_back:
                print('######## The replay experiences has been saved successfully after %d episodes ##########' % episode)
            else:
                print('######## Warning: the replay experiences can not be saved after %d episodes ##########' % episode)

        loop_end_time = time.time()
        loop_time = loop_end_time - loop_start_time
        print("EPISODE", episode, "/ REWARD", reward_episode, "/ steps ", step, "/ LoopTime:", loop_time, '/ goal_point:', goal_point)
        episode = episode + 1
        # restart at a new point
        starting_points, starting_direction, goal_point = get_next_starting_point(client, road_points)
        position = airsim.Vector3r(starting_points[0], starting_points[1], starting_points[2])
        heading = airsim.utils.to_quaternion(starting_direction[0], starting_direction[1], starting_direction[2])
        pose = airsim.Pose(position, heading)
        print(starting_points, starting_direction, goal_point)
        client.simSetVehiclePose(pose, True)
        car_controls.brake = 1
        client.setCarControls(car_controls)
        time.sleep(2)


"""
test part
"""
def test_env_feedback(client,terminal_position):
 
    collision_info = client.simGetCollisionInfo()
    car_position = client.getCarState().kinematics_estimated.position
    distance = np.sqrt((car_position.x_val-terminal_position[0])**2+(car_position.y_val-terminal_position[1])**2)
    if collision_info.has_collided:
        reset = True
    else:
        reset = False

    if (distance <= 10):
        terminal = 1
        reset = True
    else:
        terminal = 0

    return terminal,reset,distance

def print_test_action(episode,t,car_controls,distance,steer):
    print('Episode:%05d,Step:%05d '%(episode,t),end='')
    if(steer<0):
        print('The car is turning left    , ',end='')
    elif(steer>0):
        print('The car is turning right   , ',end='')
    else:
        print('The car is going straightly, ',end='')
    print('throttle=%.1f,  steer=%.3f, distance=%.2f'%(car_controls.throttle,steer,distance),end='')

def testNetwork(client, car_controls, terminal_position):

    global nmap, road_points, path_in_list, path_in_environment

    car_controls.throttle = 0.5
    car_controls.steering = 0
    client.reset()
    print('Environment initialized!')
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    # sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    sess = tf.InteractiveSession()
    with tf.name_scope("TargetNetwork"):
        Q_net = Deep_Q_Network(sess)
    time.sleep(1)

    reward_var = tf.Variable(0., trainable=False)
    tf.summary.scalar('reward', reward_var)
    # define summary
    merged_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('./logs', sess.graph)
    # get the first state
    observe_init = get_image(client,'DepthPlanner')
    state_pre = np.stack((observe_init, observe_init, observe_init, observe_init), axis=2)
    # saving and loading networks
    trainables = tf.trainable_variables()
    trainable_saver = tf.train.Saver(trainables)
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state("saved_networks/new_model_lidar/")
    print('checkpoint:', checkpoint)
    if checkpoint and checkpoint.model_checkpoint_path:
        trainable_saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        if not os.path.exists("saved_networks/new_model_lidar"):
            os.mkdir("saved_networks/new_model_lidar")
            print('The file not exists, is created successfully')
        print("Could not find old network weights")
    # start training
    episode = 1
    print('Number of trainable variables:', len(trainables))
    inner_loop_time_start = time.time()
    steps_record = []

    step = 1
    finish = False
    while not finish:
        reset = False
        while not reset:
            # take the latest 4 frames as an input
            image = get_image(client, 'Scene')
            observe = get_image(client,'DepthPlanner')
            observe = np.reshape(observe, (DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH, 1))
            state_current = np.append(image, observe, axis = 2)
            # get feedback from environoment
            terminal,reset,distance = test_env_feedback(client,terminal_position)
            # store the experience
            state_pre = state_current
            # prediction
            actions = sess.run(Q_net.readout, feed_dict={Q_net.state: [state_current]})
            readout_t = actions[0]
            action_current = np.zeros([ACTION_NUMS])
            action_index = np.argmax(readout_t)
            action_current[action_index] = 1
            # Set steering angle
            side_num = int(ACTION_NUMS - 1) // 2
            steer = float((action_index - side_num) / side_num)
            inner_loop_time_end = time.time()
            inner_loop_time = inner_loop_time_end-inner_loop_time_start
            # calculate sleep time make sure each loop time == LOOPTIME 
            if inner_loop_time <= LOOPTIME:
                time_to_sleep = LOOPTIME - inner_loop_time
                time.sleep(time_to_sleep)
            else:
                pass

            inner_loop_time_end = time.time()
            car_controls = excute_action(client,car_controls,steer)
            car_controls, whatever = image_auxiliary(car_controls)
            client.setCarControls(car_controls)

            # turn control
            # finish = turn.execute_turns(client, nmap, road_points, path_in_list, path_in_environment)

            print_test_action(episode, step, car_controls, distance, steer)
            print(',inner loop=%.4fs\n'%(inner_loop_time_end-inner_loop_time_start),end='')
            inner_loop_time_start = time.time()

            step = step+1

        client.reset()

def main():

    global nmap, road_points, path_in_list, path_in_environment

    # finish = turn.execute_turns(client, nmap, road_points, path_in_list, path_in_environment)
    if len(sys.argv) == 2:
        client = airsim.CarClient()
        client.confirmConnection()
        print('Connect succcefully！')
        client.enableApiControl(True)
        car_controls = airsim.CarControls()

        client.simSetTimeOfDay(True,"2019-08-19 21:00:00",True,1,60, False)
        client.simSetCameraOrientation(2, airsim.to_quaternion(0, 0, -math.pi/4));
        client.simSetCameraOrientation(1, airsim.to_quaternion(0, 0, math.pi/4));

        start_point = [13,13]
        target_point =  [0,22]

        nmap, road_points, path_in_list, path_in_environment = turn.init_path_planning(start_point,target_point)

        if sys.argv[1] == 'test':
            print('start testing')
            testNetwork(client, car_controls , list(path_in_environment[-1]))

        elif sys.argv[1] == 'train':
            print('start training')
            trainNetwork(client, car_controls)

        else:
            print('unkown input argument, please run this file as follow!')
            print('python DQN_airsim.py train OR python DQN_airsim.py test')
    elif len(sys.argv) == 3:
        client = airsim.CarClient()
        client.confirmConnection()
        print('Connect succcefully！')
        if sys.argv[1] == 'show_map' and sys.argv[2] == 'train':
            print('display map')
            show_map.show_map(client, [])
        elif sys.argv[1] == 'show_map' and sys.argv[2] == 'test':
            print('display map')
            show_map.show_map(client, path_in_environment)
        else:
            print('unkown input argument, please run this file as follow!')
            print('python DQN_airsim.py show_map train OR python DQN_airsim.py show_map test')
    else:
        print('please include an argument train or test or show_map')

# init path parameters
nmap, road_points, path_in_list, path_in_environment = turn.init_path_planning([13,13],[0,22])

if __name__ == "__main__":
    main()