from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Activation
from keras import regularizers
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
# from collections import deque
import random
import gym
# from typing import List

import argparse



class DQN():
	def __init__(self, discount=0.99, batch_size = 64, max_episodes = 300):
		# TODO: Import new environment
        # self.env = gym.make('CartPole-v0')
		# self.env.wrappers.Monitor(env, directory="results/", force=True)

        # TODO: Get observation space of multirotor environment
		# self.input_size= self.env.observation_space.shape[0]
        # TODO: Get action space of multirotor environment        
		# self.output_size= self.env.action_space.n

		self.DISCOUNT_RATE=discount
		self.BATCH_SIZE = batch_size
		self.TARGET_UPDATE_FREQUENCY = 5
		self.MAX_EPISODES = max_episodes

		self.main_dqn = self.build()
		self.target_dqn = self.build()

		self.main_dqn.compile(optimizer = Adam(), loss ="mean_squared_error")

		self.target_dqn.set_weights(self.main_dqn.get_weights())

	def build(self, h_size = 16, lr = 0.001):
		state = Input(shape=(self.input_size,))
		dense1 = Dense(h_size, activation = "relu")(state)
		action = Dense(self.output_size, kernel_regularizer=regularizers.l2(0.01))(dense1)
		model = Model(state, action)
		return model


	def train(self):
		buffer = []
		last_100_game_reward = []

		for episode in range(self.MAX_EPISODES):
			e = e = 1. / ((episode / 10) + 1)
			done = False
			step_count = 0
			state =  self.env.reset()

			while not done:
				state = np.reshape(state, (1,self.input_size))
				if np.random.rand() < e:
					action = self.env.action_space.sample()
				else:
					action = np.argmax(self.main_dqn.predict(state))
					# print("predict", self.main_dqn.predict(state))

				next_state, reward, done, info = self.env.step(action)

				if done:
					reward = -1
				# print(action)
				buffer.append((state, action, reward, next_state, done))

				if len(buffer) > self.BATCH_SIZE:
					minibatch = random.sample(buffer, self.BATCH_SIZE)
					
					states = np.vstack([x[0] for x in minibatch])
					actions = np.array([x[1] for x in minibatch])
					rewards = np.array([x[2] for x in minibatch])
					next_states = np.vstack([x[3] for x in minibatch])
					done_array = np.array([x[4] for x in minibatch])

					# print(actions, actions.shape)

					Q_target = rewards + self.DISCOUNT_RATE*np.max(self.target_dqn.predict(next_states), axis=1) * ~done_array

					y = self.main_dqn.predict(states)
					y[np.arange(len(states)), actions] = Q_target
					# print(y,y.shape)
					# print(states.shape, actions.shape)
					self.main_dqn.train_on_batch(states, y)

				if step_count % self.TARGET_UPDATE_FREQUENCY == 0:
					self.target_dqn.set_weights(self.main_dqn.get_weights())

				state = next_state
				step_count += 1
			print("Episode: {}  steps: {}".format(episode, step_count))

	def play(self):

        # TODO: Initialize obersvation by getting reset of envrionment
		# observation = self.env.reset()
		reward_sum = 0
		while True:
			# TODO: Get multirotor envrionment render
            # self.env.render()

			s = np.reshape(observation, (1,self.input_size))
			Qs = self.main_dqn.predict(s)
			a = np.argmax(Qs)

            # TODO: Get step function of multirotor environment
			# observation, reward, done, info = self.env.step(a)
			reward_sum += reward
			if done:
				print("total score: {}".format(reward_sum))
                # TODO: Initialize obersvation by getting reset of envrionment                
				# observation = self.env.reset()
		        reward_sum = 0 


def transform_input(responses):
    img1d = np.array(responses[0].image_data_float, dtype=np.float)
    img1d = 255/np.maximum(np.ones(img1d.size), img1d)
    img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

    from PIL import Image
    image = Image.fromarray(img2d)
    im_final = np.array(image.resize((84, 84)).convert('L')) 

    return im_final

def interpret_action(action):
    scaling_factor = 0.25
    if action == 0:
        quad_offset = (0, 0, 0)
    elif action == 1:
        quad_offset = (scaling_factor, 0, 0)
    elif action == 2:
        quad_offset = (0, scaling_factor, 0)
    elif action == 3:
        quad_offset = (0, 0, scaling_factor)
    elif action == 4:
        quad_offset = (-scaling_factor, 0, 0)    
    elif action == 5:
        quad_offset = (0, -scaling_factor, 0)
    elif action == 6:
        quad_offset = (0, 0, -scaling_factor)
    
    return quad_offset

def compute_reward(quad_state, quad_vel, collision_info):
    thresh_dist = 7
    beta = 1

    z = -10
    pts = [np.array([-.55265, -31.9786, -19.0225]), np.array([48.59735, -63.3286, -60.07256]), np.array([193.5974, -55.0786, -46.32256]), np.array([369.2474, 35.32137, -62.5725]), np.array([541.3474, 143.6714, -32.07256])]

    quad_pt = np.array(list((quad_state.x_val, quad_state.y_val, quad_state.z_val)))

    if collision_info.has_collided:
        reward = -100
    else:    
        dist = 10000000
        for i in range(0, len(pts)-1):
            dist = min(dist, np.linalg.norm(np.cross((quad_pt - pts[i]), (quad_pt - pts[i+1])))/np.linalg.norm(pts[i]-pts[i+1]))

        #print(dist)
        if dist > thresh_dist:
            reward = -10
        else:
            reward_dist = (math.exp(-beta*dist) - 0.5) 
            reward_speed = (np.linalg.norm([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val]) - 0.5)
            reward = reward_dist + reward_speed

    return reward

def isDone(reward):
    done = 0
    if  reward <= -10:
        done = 1
    return done



initX = -.55265
initY = -31.9786
initZ = -19.0225

# connect to the AirSim simulator 
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

client.takeoffAsync().join()
client.moveToPositionAsync(initX, initY, initZ, 5).join()
client.moveByVelocityAsync(1, -0.67, -0.8, 5).join()
time.sleep(0.5)

# Make RL agent
NumBufferFrames = 4
SizeRows = 84
SizeCols = 84
NumActions = 7
agent = DeepQAgent((NumBufferFrames, SizeRows, SizeCols), NumActions, monitor=True)

# Train
epoch = 100
current_step = 0
max_steps = epoch * 250000

responses = client.simGetImages([airsim.ImageRequest(3, airsim.ImageType.DepthPerspective, True, False)])
current_state = transform_input(responses)

while True:
    action = agent.act(current_state)
    quad_offset = interpret_action(action)
    quad_vel = client.getMultirotorState().kinematics_estimated.linear_velocity
    client.moveByVelocityAsync(quad_vel.x_val+quad_offset[0], quad_vel.y_val+quad_offset[1], quad_vel.z_val+quad_offset[2], 5).join()
    time.sleep(0.5)
 
    quad_state = client.getMultirotorState().kinematics_estimated.position
    quad_vel = client.getMultirotorState().kinematics_estimated.linear_velocity
    collision_info = client.simGetCollisionInfo()
    reward = compute_reward(quad_state, quad_vel, collision_info)
    done = isDone(reward)
    print('Action, Reward, Done:', action, reward, done)

    agent.observe(current_state, action, reward, done)
    agent.train()

    if done:
        client.moveToPositionAsync(initX, initY, initZ, 5).join()
        client.moveByVelocityAsync(1, -0.67, -0.8, 5).join()
        time.sleep(0.5)
        current_step +=1

    responses = client.simGetImages([airsim.ImageRequest(3, airsim.ImageType.DepthPerspective, True, False)])
    current_state = transform_input(responses)