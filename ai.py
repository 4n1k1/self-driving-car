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
_SCALE_FACTOR = 300
_LEARNING_PERIOD = 1
_LEARNING_BATCH_SIZE = 1000  # uniform selection from memory
_MEMORY_SIZE = 10000
_PLOTTING_INTERVAL = 1000

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
			nn.functional.relu(  # rectifier linear units
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
		# what zip does: (1,2), (3,4), (5,6) => (1,3,5), (2,4,6)
		samples = zip(*self.__data[0::_MEMORY_SIZE/_LEARNING_BATCH_SIZE])
		return map(lambda x: autograd.Variable(torch.cat(x, 0)), samples)

	@property
	def size(self):
		return len(self.__data)


class Brain:
	"""
	Deep Q Learning brain implemenation.
	"""
	_DISCOUNT_FACTOR = 0.9

	def __init__(self, input_size, output_size):
		self.__gamma = self._DISCOUNT_FACTOR
		self.__last_rewards = []  # is used for plotting
		self.__module = _DrivingModule(input_size, output_size)
		self.__memory = _ShortTermMemory(_MEMORY_SIZE)
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
		self.__current_tick_num = 0

	def __select_action(self, state):
		return nn.functional.softmax(
			self.__module(autograd.Variable(state, volatile = True)) * _SCALE_FACTOR, dim=1
		).multinomial().data[0,0]

	def __learn(self, batch_state, batch_next_state, batch_reward, batch_action):
		outputs = self.__module(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
		next_outputs = self.__module(batch_next_state).detach().max(1)[0]  # action = 1, state = 0
		target = self.__gamma*next_outputs + batch_reward
		td_loss = nn.functional.smooth_l1_loss(outputs, target)
		self.__optimizer.zero_grad()  # re-init optimizer
		td_loss.backward(retain_graph=True)  # backpropagate
		self.__optimizer.step()  # update the weights

	def update(self, reward, car_state):
		self.__current_tick_num += 1

		new_state = torch.Tensor(car_state).float().unsqueeze(0)
		self.__memory.remember(
			(
				self.__last_state,
				new_state,
				torch.Tensor([self.__last_reward]),
				torch.LongTensor([int(self.__last_action)]),
			)
		)
		action = self.__select_action(new_state)

		if self.__current_tick_num >= _LEARNING_PERIOD:
			self.__learn(*self.__memory.recall(
				self.__memory.size if self.__memory.size < _LEARNING_BATCH_SIZE else _LEARNING_BATCH_SIZE
			))
			self.__current_tick_num = 0

		self.__last_action = action
		self.__last_state = new_state
		self.__last_reward = reward

		self.__last_rewards.append(reward)

		if len(self.__last_rewards) > _PLOTTING_INTERVAL:
			del self.__last_rewards[0]

		return action

	@property
	def score(self):
		return sum(self.__last_rewards)/(len(self.__last_rewards) + 1)

	def save(self, file_name):
		torch.save({
			'state_dict': self.__module.state_dict(),
			'optimizer' : self.__optimizer.state_dict(),
		}, file_name)

	def load(self, file_name):
		if os.path.isfile(file_name):
			checkpoint = torch.load(file_name)
			self.__module.load_state_dict(checkpoint['state_dict'])
			self.__optimizer.load_state_dict(checkpoint['optimizer'])
