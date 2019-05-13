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
		self.env = gym.make('CartPole-v0')
		# self.env.wrappers.Monitor(env, directory="results/", force=True)

		self.input_size= self.env.observation_space.shape[0]
		self.output_size= self.env.action_space.n

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

		observation = self.env.reset()
		reward_sum = 0
		while True:
			self.env.render()

			s = np.reshape(observation, (1,self.input_size))
			Qs = self.main_dqn.predict(s)
			a = np.argmax(Qs)

			observation, reward, done, info = self.env.step(a)
			reward_sum += reward
			if done:
				print("total score: {}".format(reward_sum))
				observation = self.env.reset()
		        reward_sum = 0 

if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--discount", required=False)
	ap.add_argument("-b", "--batch", required=False)
	ap.add_argument("-ep", "--max", required=False)

	args = vars(ap.parse_args())

	dqn = DQN(float(args["discount"]), int(args["batch"]), int(args["max"]))
	dqn.train()


	dqn.play()
	