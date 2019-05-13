import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


env = gym.make('CartPole-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position+1)%self.capacity

    def sample(self, batch_size):
        """Select a random batch of transitions for training"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Return length of replay memory"""
        return len(self.memory)


class Network(object):
    def __init__(self, h, w, outputs):
        
        self.convNet = self.convNet(h, w, outputs)

    def convNet(h, w, outputs):
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)        

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class DQN(object):
    def __init__(self, batch_size=64, discount_rate=0.99, max_episodes=300):
        self.env = gym.make('CartPole-v0')
        
        self.obs = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

        self.BATCH_SIZE = batch_size
        self.DISCOUNT_RATE = discount_rate
        self.MAX_EPISODES = max_episodes

        self.TARGET_UPDAT_FREQUENCY = 5
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200

        self.main_Qnet = self.Network(obs, n_actions)
        self.target_Qnet = self.Network(obs, n_actions)


    def plot_durations():
        plt.figure(2)
        plt.clf()
        durations_t = torch.tensor(episode_durations, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())


    def train(self):
        buffer = ReplayMemory()

        for episode in range(self.MAX_EPISODES):
            e = 1/.((episode/10)+1)
            done = False
            step_count = 0
            state = self.env.reset()

            while not done:
                state = np.reshape(state, (1,self.obs))
                if np.random.rand() < e:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.main_Qnet.predict(state))
                
                next_state, reward, done, info = self.env.step(action)

                if done:
                    reward = -1

                buffer.push((state, action, reward, next_state))

                if len(buffer) > self.BATCH_SIZE:
                    minibatch = random.sample(buffer, self.BATCH_SIZE)

				    states = np.vstack([x[0] for x in minibatch])
					actions = np.array([x[1] for x in minibatch])
					rewards = np.array([x[2] for x in minibatch])
					next_states = np.vstack([x[3] for x in minibatch])

                    Q_target = rewards + self.DISCOUNT_RATE+np.max(self.target_Qnet.forward(next_states), axis=1)

                    y = self.main_Qnet(states)
                    y[np.arrange(len(states)), actions] = Q_target
                    # self.main_Qnet.train_on

                if step_count % self.TARGET_UPDAT_FREQUENCY == 0:
                    # self.target_Qnet

                state = next_state
                step_count +=1
            print("Episode: {}  steps: {}".format(episode, step_count))

    

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
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        Q_value = main_Qnet(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        next_Q_values = (next_state_values * DISCOUNT_RATE) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

    num_episodes = 50
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        env.reset()
        last_screen = get_screen()
        current_screen = get_screen()
        state = current_screen - last_screen

        for t in count():
            # Select and perform an action
            action = select_action(state)
            _, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)

            # Observe new state
            last_screen = current_screen
            current_screen = get_screen()
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            optimize_model()
            if done:
                episode_durations.append(t + 1)
                plot_durations()
                break

        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
    ap.add_argument("-b", "--batch", required=False)
	ap.add_argument("-d", "--discount", required=False)
	ap.add_argument("-ep", "--max", required=False)

	args = vars(ap.parse_args())

	dqn = DQN(int(args["batch"]), float(args["discount"]), int(args["max"]))
	dqn.train()

    print('Complete')
    env.render()
    env.close()
    plt.ioff()
    plt.show()   
