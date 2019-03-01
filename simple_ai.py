import numpy
from math import exp, tan, pow
from random import random
from collections import namedtuple


def sigmoid(weighted_input):
	return 1.0 / (1.0 + exp(-weighted_input))


def hyper_tang(weighted_input):
	return (exp(weighted_input) - exp(-weighted_input)) / (exp(weighted_input) + exp(-weighted_input))


DERIVATIVES = {
	sigmoid: lambda output: output * (1.0 - output),
	hyper_tang: lambda output: 1 - pow(tan(output), 2),
}


class NeuralNetwork(object):
	def __init__(self, structure, is_reinforcement=True, capacity=1000):
		self._discount_factor = 0.9
		self._layers = []
		self._is_reinforcement = is_reinforcement
		self._bias_neuron = BiasNeuron()
		self._max_capacity = capacity
		self._memory = namedtuple(
			"RLMemory", [
				"t0_state",
				"t1_state",
				"t0_to_t1_action",
				"t0_to_t1_reward",
			]
		)([], [], [], [])

		for idx, neurons_count in enumerate(structure):
			if idx == 0:
				layer = [StateNeuron() for i in range(neurons_count)]
			elif idx == len(structure) - 1:
				layer = [PredictionNeuron(sigmoid, 10.0) for i in range(neurons_count)]
			else:
				layer = [HiddenNeuron(i, sigmoid, 10.0) for i in range(neurons_count)]

			self._layers.append(layer)

		for idx, layer in enumerate(self._layers):
			for neuron in layer:
				if idx == 0:
					neuron.connect([], self._layers[idx + 1])
				elif idx == len(self._layers) - 1:
					neuron.connect(self._layers[idx - 1] + [self._bias_neuron], [])
				else:
					neuron.connect(self._layers[idx - 1] + [self._bias_neuron], self._layers[idx + 1])

		for layer in self._layers[1:]:
			self._bias_neuron.connect([], layer)

	def remember(self, t0_state, t1_state, t0_to_t1_action, t0_to_t1_reward):
		if len(self._memory.t0_state) == self._max_capacity:
			del(self._memory.t0_state[0])
			del(self._memory.t1_state[0])
			del(self._memory.t0_to_t1_action[0])
			del(self._memory.t0_to_t1_reward[0])

		self._memory.t0_state.append(t0_state)
		self._memory.t1_state.append(t1_state)
		self._memory.t0_to_t1_action.append(t0_to_t1_action)
		self._memory.t0_to_t1_reward.append(t0_to_t1_reward)

	def _get_training_data(self):
		result = [[], [], []]

		for i in range(len(self._memory.t0_state)):
			t0_predicted_output = self.predict(self._memory.t0_state[i])
			t0_to_t1_action = self._memory.t0_to_t1_action[i]
			t0_picked_output_value = t0_predicted_output[t0_to_t1_action]
			t1_predicted_output = self.predict(self._memory.t1_state[i])
			t1_best_output_value = max(t1_predicted_output)

			result[0].append(t0_picked_output_value)
			result[1].append(t1_best_output_value)
			result[2].append(self._memory.t0_to_t1_reward[i])

		return result

	def learn(self, last_state, last_action, reward):
		for idx, value in enumerate(last_state):
			self._layers[0][idx].output = value

		#
		# State forward propagation.
		#
		for layer in self._layers[1:]:
			for neuron in layer:
				neuron.calculate_output()

		#
		# Error backward propagation.
		#
		for idx, layer in enumerate(reversed(self._layers[1:])):
			if idx == 0:
				layer[last_action].calculate_error(reward)
			else:
				for neuron in layer:
					neuron.calculate_error()

		#
		# Weights update.
		#
		for layer in self._layers[1:]:
			for neuron in layer:
				neuron.update_weights()

		if False:
			self.write_visual_file()

	def predict(self, state):
		for idx, value in enumerate(state):
			self._layers[0][idx].output = value

		for layer in self._layers[1:]:
			for neuron in layer:
				neuron.calculate_output()

		outputs = []

		for neuron in self._layers[-1]:
			outputs.append(neuron.output)

		probabilities = self._softmax(outputs)

		return numpy.random.choice(
			list(range(len(self._layers[-1]))),  # list of actions [0, 1, 2]
			1,  # pick one best given
			p=probabilities  # probabilities
		)[0]

	def _softmax(self, values, enforce_factor=1.0):
		values = [value * enforce_factor for value in values]

		return numpy.exp(values) / numpy.sum(numpy.exp(values), axis=0)

	def write_visual_file(self):
		self._visual_file = open("network.visual", "w")

		self._visual_file.seek(0)
		self._visual_file.write("          ".join([str(neuron.output) for neuron in self._layers[0]]) + "\n")
		self._visual_file.write("|\n")

		for layer_idx, layer in enumerate(self._layers[1:], 1):
			self._visual_file.write("          ".join(["========="] * len(self._layers[layer_idx])) + "\n")

			for neuron_idx, neuron_0 in enumerate(self._layers[layer_idx - 1]):
				self._visual_file.write("          ".join(["{: f}".format(neuron_1.weights[neuron_idx]) for neuron_1 in self._layers[layer_idx]]) + "\n")

			self._visual_file.write("          ".join(["---------"] * len(self._layers[layer_idx])) + "\n")
			self._visual_file.write("          ".join(["{: f}".format(neuron_1.output) for neuron_1 in self._layers[layer_idx]]) + "\n")
			self._visual_file.write("          ".join(["========="] * len(self._layers[layer_idx])) + "\n")
			self._visual_file.write("|\n")

		self._visual_file.close()


class Neuron(object):
	def __init__(self):
		self._output = 0.0

		self._output_neurons = []
		self._input_neurons = []

	def connect(self, input_neurons, output_neurons):
		self._input_neurons = input_neurons
		self._output_neurons = output_neurons

	@property
	def output(self):
		return self._output


class StateNeuron(Neuron):
	def __init__(self):
		super(StateNeuron, self).__init__()

	@property
	def output(self):
		return super(StateNeuron, self).output

	@output.setter
	def output(self, new_output):
		self._output = new_output


class BiasNeuron(StateNeuron):
	def __init__(self):
		super(BiasNeuron, self).__init__()

		self._output = 1.0


class NeuronCore(Neuron):
	def __init__(self, activation_function, learning_rate):
		super(NeuronCore, self).__init__()

		self._weights = []
		self._error = 0.0

		self._activation_function = activation_function
		self._learning_rate = learning_rate

	@property
	def error(self):
		return self._error

	@property
	def weights(self):
		return self._weights

	def connect(self, input_neurons, output_neurons):
		super(NeuronCore, self).connect(input_neurons, output_neurons)

		for i in range(len(input_neurons)):
			#
			# These are considered to be the best initialization values.
			#
			self._weights.append(2 * random() - 1)

	def calculate_output(self):
		weighted_input = 0.0

		for idx, neuron in enumerate(self._input_neurons):
			weighted_input += neuron.output * self._weights[idx]

		self._output = self._activation_function(weighted_input)

	def update_weights(self):
		new_weights = []

		for idx, weight in enumerate(self._weights):
			weight_delta = self._learning_rate * self._input_neurons[idx].output * self._error

			new_weights.append(self._weights[idx] + weight_delta)

		self._weights = new_weights


class PredictionNeuron(NeuronCore):
	def __init__(self, activation_function, learning_rate):
		super(PredictionNeuron, self).__init__(activation_function, learning_rate)

		self.expected = 0.0

	def calculate_error(self, reward):
		self._error = reward * DERIVATIVES[self._activation_function](self._output)


class HiddenNeuron(NeuronCore):
	def __init__(self, idx_in_layer, activation_function, learning_rate):
		super(HiddenNeuron, self).__init__(activation_function, learning_rate)

		self._idx = idx_in_layer

	def calculate_error(self):
		self._error = 0.0

		for neuron in self._output_neurons:
			self._error += neuron.error * neuron.weights[self._idx]

		self._error *= DERIVATIVES[self._activation_function](self._output)


class Brain(object):
	def __init__(self, input_size, output_size):
		self._last_state = [0.0] * input_size
		self._last_reward = 0.0
		self._last_action = 0

		self._rewards = []
		self._rewards_capacity = 1000

		self._nn = NeuralNetwork([input_size, 30, output_size], is_reinforcement=True)

	def update(self, reward, car_state):
		action = self._nn.predict(car_state)

		self._nn.remember(
			self._last_state,
			car_state,
			self._last_reward,
			self._last_action,
		)

		self._nn.learn(self._last_state, self._last_action, reward)

		self._last_state = car_state
		self._last_reward = reward
		self._last_action = action

		if len(self._rewards) > self._rewards_capacity:
			del(self._rewards[0])

		self._rewards.append(reward)

		return action

	@property
	def score(self):
		return sum(self._rewards)/(len(self._rewards) + 1)

	def save(self, file_name):
		pass

	def load(self, file_name):
		pass
