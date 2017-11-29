import random
import os

import torch

from torch import nn
from torch import optim
from torch import autograd

_HIDDEN_LAYER_NEURONS_COUNT = 30  # this is something to experiment with

"""
	Also called temperature, decreses low softmax values and increases high values.
	This basically speeds up learning by increasing chance of high values to be picked.
"""
_SCALE_FACTOR = 100

class _DrivingModule(nn.Module):
	"""
		Single module neural network implementation.

		It has only one hidden layer.
	"""

	def __init__(self, input_size, output_size):
		nn.Module.__init__(self)

		self.__input_to_hidden_connections = nn.Linear(
			input_size,
			_HIDDEN_LAYER_NEURONS_COUNT,
		)

		self.__hidden_to_output_connections = nn.Linear(
			_HIDDEN_LAYER_NEURONS_COUNT,
			output_size,
		)

	def forward(self, state):
		"""
			Forward propagation implementation.
		"""

		return self.__hidden_to_output_connections(
			nn.functional.relu(  # rectifier lower to upper
				self.__input_to_hidden_connections(state),
			),
		)


class _ShortTermMemory:
	"""
		Experience replay implementation.
	"""

	def __init__(self, capacity):
		self.__capacity = capacity
		self.__data = []

	def remember(self, event):
		self.__data.append(event)
		if len(self.__data) > self.__capacity:
			del self.__data[0]

	def recall(self, batch_size):
		# uniform distribution is probably better than random samples
		# what zip does: ((1,2), (3,4), (5,6)) => ((1,3,5), (2,4,6))
		samples = zip(*random.sample(self.__data, batch_size))
		return map(lambda x: autograd.Variable(torch.cat(x, 0)), samples)

	@property
	def data(self):
		return self.__data


class Brain:
	"""
	Deep Q Learning brain implemenation.
	"""

	def __init__(self, input_size, output_size, discount_factor):
		self.__gamma = discount_factor
		self.__reward_window = []
		self.__module = _DrivingModule(input_size, output_size)
		self.__memory = _ShortTermMemory(100000)
		# below is one of many variants of gradient descent tools
		self.__optimizer = optim.Adam(self.__module.parameters(), lr = 0.001)
		"""
			>>> torch.Tensor(5)

			 0.0000e+00
			 0.0000e+00
			 6.2625e+22
			 4.7428e+30
			 3.7843e-39
			[torch.FloatTensor of size 5x1]

			>>> y.unsqueeze(0)

			 0.0000e+00  0.0000e+00  6.2625e+22  4.7428e+30  3.7843e-39
			[torch.FloatTensor of size 1x5]
		"""
		self.__last_state = torch.Tensor(input_size).unsqueeze(0)
		self.__last_action = 0
		self.__last_reward = 0

	def select_action(self, state):
		probs = nn.functional.softmax(
			self.__module(autograd.Variable(state, volatile = True)) * _SCALE_FACTOR,
			dim=1
		)
		action = probs.multinomial()
		return action.data[0,0]

	def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
		outputs = self.__module(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
		next_outputs = self.__module(batch_next_state).detach().max(1)[0]  # action = 1, state = 0
		target = self.__gamma*next_outputs + batch_reward
		td_loss = nn.functional.smooth_l1_loss(outputs, target)
		self.__optimizer.zero_grad()  # re-init optimizer
		td_loss.backward(retain_graph=True)  # backpropagate
		self.__optimizer.step()  # update the weights

	def update(self, reward, new_signal):
		new_state = torch.Tensor(new_signal).float().unsqueeze(0)
		self.__memory.remember(
			(
				self.__last_state,
				new_state,
				torch.LongTensor([int(self.__last_action)]),
				torch.Tensor([self.__last_reward])
			)
		)
		action = self.select_action(new_state)
		if len(self.__memory.data) > 100:
			batch_state, batch_next_state, batch_action, batch_reward = self.__memory.recall(100)
			self.learn(batch_state, batch_next_state, batch_reward, batch_action)
		self.__last_action = action
		self.__last_state = new_state
		self.__last_reward = reward
		self.__reward_window.append(reward)

		if len(self.__reward_window) > 1000:
			del self.__reward_window[0]
		return action

	def score(self):
		return sum(self.__reward_window)/(len(self.__reward_window)+1.0)

	def save(self):
		torch.save({
			'state_dict': self.__module.state_dict(),
			'optimizer' : self.__optimizer.state_dict(),
		}, 'last_brain.pth')

	def load(self):
		if os.path.isfile('last_brain.pth'):
			print("=> loading checkpoint... ")
			checkpoint = torch.load('last_brain.pth')
			self.__module.load_state_dict(checkpoint['state_dict'])
			self.__optimizer.load_state_dict(checkpoint['optimizer'])
			print("done !")
		else:
			print("no checkpoint found...")
