import os

import torch

from torch import nn
from torch import optim

_HIDDEN_LAYER_NEURONS_COUNT = 30  # this is something to experiment with

"""
	Also called temperature, decreases low softmax values and increases high values.
	This speeds up learning by increasing chance of high values to be picked but
	reduces exploration.
"""
_SCALE_FACTOR = 300
_LEARNING_PERIOD = 1
_MEMORY_SIZE = 1000  #
_PLOTTING_INTERVAL = 1000
_SELECTION_STEP = 10  # learn on every 10th memory stamp


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

	def recall(self):
		# what zip does: (1,2), (3,4), (5,6) => (1,3,5), (2,4,6)
		samples = zip(*self.__data[0::_SELECTION_STEP])
		return map(lambda x: torch.cat(x, 0), samples)

	@property
	def size(self):
		return len(self.__data)


class Brain:
	"""
	Deep Q Learning brain implementation.
	"""
	def __init__(self, input_size, output_size):
		self.__gamma = 0.9
		self.__last_rewards = []  # is used for plotting
		self.__module = _DrivingModule(input_size, output_size)
		self.__memory = _ShortTermMemory(_MEMORY_SIZE)
		# below is one of many variants of gradient descent tools
		self.__optimizer = optim.Adam(self.__module.parameters(), lr=0.001)
		"""
			>>> y = torch.Tensor(5)
			>>> y
			tensor([-2.1135e-03,  3.0726e-41,  4.7428e+30,  3.7843e-39,  4.4842e-44])

			>>> y.unsqueeze(0)
			tensor([[-1.9258e-03,  3.0726e-41, -2.2202e-03,  3.0726e-41,  4.4842e-44]])
		"""
		self.__last_state = torch.Tensor(input_size).unsqueeze(0)
		self.__last_action = 0
		self.__last_reward = 0
		self.__current_tick_num = 0

	def __select_action(self, state):
		with torch.no_grad():
			action_probabilities = nn.functional.softmax(
				self.__module(state) * _SCALE_FACTOR, dim=1
			)[0]

			best_action = action_probabilities.max()

			return torch.LongTensor(
				[list(action_probabilities).index(best_action)]
			)

	def __learn(self, batch_state, batch_next_state, batch_reward, batch_action):
		"""
			>>> batch_state
			tensor([[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
			        [ 0.0000,  0.0000,  0.0000, -0.1406,  0.1406],
			        [ 0.0000,  0.0000,  0.0000,  0.7420, -0.7420]])
			>>> self.__module(batch_state)
				tensor([[-0.5203, -0.3913, -0.0808],
				        [-0.4317, -0.3452,  0.0710],
				        [-1.0347, -0.6176, -0.9143]], grad_fn=<ThAddmmBackward>)
			>>> batch_action.unsqueeze(1)
				tensor([[0],
				        [2],
				        [2]])
			>>> self.__module(batch_state).gather(1, batch_action.unsqueeze(1))
				tensor([[-0.5203],
				        [ 0.0710],
				        [-0.9143]], grad_fn=<GatherBackward>)
			>>> self.__module(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
				tensor([-0.5203,  0.0710, -0.9143], grad_fn=<GatherBackward>)
		"""
		outputs = self.__module(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
		"""
			>>> self.__module(batch_next_state)
			tensor([[ 0.1743, -0.0896, -0.0638],
				[ 0.0692, -0.2744, -0.2676],
				[ 0.0692, -0.2742, -0.2674]], grad_fn=<ThAddmmBackward>)
			>>> self.__module(batch_next_state).detach()
			tensor([[ 0.1743, -0.0896, -0.0638],
				[ 0.0692, -0.2744, -0.2676],
				[ 0.0692, -0.2742, -0.2674]])
			>>> self.__module(batch_next_state).detach().max(1)
			(tensor([0.1743, 0.0692, 0.0692]), tensor([0, 0, 0]))
			>>> self.__module(batch_next_state).detach().max(1)[0]
			tensor([0.1743, 0.0692, 0.0692])
		"""
		next_outputs = self.__module(batch_next_state).detach().max(1)[0]

		target = self.__gamma * next_outputs + batch_reward

		td_loss = nn.functional.smooth_l1_loss(outputs, target)

		self.__optimizer.zero_grad()  # re-init optimizer
		td_loss.backward(retain_graph=True)  # backpropagate
		self.__optimizer.step()  # update the weights

	def update(self, reward, car_state):
		self.__current_tick_num += 1

		"""
			>>> y = torch.Tensor(5)
			>>> y
			tensor([-2.1135e-03,  3.0726e-41,  4.7428e+30,  3.7843e-39,  4.4842e-44])

			>>> y.unsqueeze(0)
			tensor([[-1.9258e-03,  3.0726e-41, -2.2202e-03,  3.0726e-41,  4.4842e-44]])
		"""
		new_state = torch.Tensor(car_state).unsqueeze(0)

		action = self.__select_action(new_state)

		self.__memory.remember(
			(
				self.__last_state,
				new_state,
				torch.Tensor([self.__last_reward]),
				torch.LongTensor([self.__last_action]),
			)
		)

		if self.__current_tick_num >= _LEARNING_PERIOD:
			self.__learn(*self.__memory.recall())
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
