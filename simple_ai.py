#!/usr/bin/env python

import numpy

from math import exp, tan
from random import uniform
from time import time, sleep


def sigmoid(weighted_input):
    return 1.0 / (1.0 + exp(-weighted_input))


def hyper_tang(weighted_input):
    return (exp(weighted_input) - exp(-weighted_input)) / (exp(weighted_input) + exp(-weighted_input))


DERIVATIVES = {
    sigmoid: lambda output: output * (1.0 - output),
    hyper_tang: lambda output: 1 - pow(tan(output), 2),
}


class NeuralNetwork:
    def __init__(self, structure, args):
        self._layers = []
        self._scale_factor = args.scale_factor
        self._visual_file = open("network.visual", "w")

        for idx, neurons_count in enumerate(structure):
            if idx == 0:
                layer = [StateNeuron() for _ in range(neurons_count)]
            elif idx == len(structure) - 1:
                layer = [PredictionNeuron(sigmoid) for _ in range(neurons_count)]
            else:
                layer = [HiddenNeuron(i, sigmoid) for i in range(neurons_count)]

            self._layers.append(layer)

        for idx, layer in enumerate(self._layers):
            for neuron in layer:
                if idx == 0:
                    neuron.connect([], self._layers[idx + 1])
                elif idx == len(self._layers) - 1:
                    neuron.connect(self._layers[idx - 1], [])
                else:
                    neuron.connect(self._layers[idx - 1], self._layers[idx + 1])

    @staticmethod
    def _softmax(values, enforce_factor=1.0):
        values = [value * enforce_factor for value in values]

        return numpy.exp(values) / numpy.sum(numpy.exp(values), axis=0)

    def learn(self, state, reward):
        for idx, value in enumerate(state):
            self._layers[0][idx].output = value

        action_probabilities = self.predict(state)

        for idx, value in enumerate(solution):
            self._layers[-1][idx].expected = solution[idx]

        #
        # State forward propagation.
        #
        for idx, neuron in enumerate(self._layers[-1]):
            neuron.calculate_output()

        #
        # Error backward propagation.
        #
        for layer in reversed(self._layers[1:]):
            for neuron in layer:
                neuron.calculate_error()

        #
        # Weights update.
        #
        for layer in self._layers[1:]:
            for neuron in layer:
                neuron.update_weights()

        return [neuron.output for neuron in self._layers[-1]]

    def predict(self, state):
        for idx, value in enumerate(state):
            self._layers[0][idx].output = value

        output = [neuron.calculate_output() for neuron in self._layers[-1]]

        action_probabilities = self._softmax(
            output * self._scale_factor, dim=1
        )[0]

        best_action = action_probabilities.max()

        return torch.LongTensor(
            [list(action_probabilities).index(best_action)]
        )

    def write_visual_file(self):
        self._visual_file.seek(0)
        self._visual_file.write("          ".join([str(neuron.output) for neuron in self._layers[0]]) + "\n")
        self._visual_file.write("|\n")

        for layer_idx, _ in enumerate(self._layers[1:], 1):
            self._visual_file.write("          ".join(["========="] * len(self._layers[layer_idx])) + "\n")

            for neuron_idx, _ in enumerate(self._layers[layer_idx - 1]):
                self._visual_file.write("          ".join(
                    ["{: f}".format(neuron_1.weights[neuron_idx]) for neuron_1 in self._layers[layer_idx]]
                ) + "\n")

            self._visual_file.write("          ".join(["---------"] * len(self._layers[layer_idx])) + "\n")
            self._visual_file.write("          ".join(
                ["{: f}".format(neuron_1.output) for neuron_1 in self._layers[layer_idx]]
            ) + "\n")
            self._visual_file.write("          ".join(["========="] * len(self._layers[layer_idx])) + "\n")
            self._visual_file.write("|\n")


class Neuron:
    def __init__(self):
        self._output = 0.0
        self._bias = uniform(-1.0, 1.0)
        self._output_neurons = None
        self._input_neurons = None

    def connect(self, input_neurons, output_neurons):
        self._input_neurons = input_neurons
        self._output_neurons = output_neurons

    @property
    def output(self):
        return self._output


class StateNeuron(Neuron):
    def calculate_output(self):
        return self._output

    @property
    def output(self):
        return super(StateNeuron, self).output

    @output.setter
    def output(self, new_output):
        self._output = new_output


class NeuronCore(Neuron):
    def __init__(self, activation_function):
        super(NeuronCore, self).__init__()

        self._weights = []
        self._error = 0.0

        self._activation_function = activation_function

    @property
    def error(self):
        return self._error

    @property
    def weights(self):
        return self._weights

    def connect(self, input_neurons, output_neurons):
        super(NeuronCore, self).connect(input_neurons, output_neurons)

        for _ in range(len(input_neurons)):
            #
            # These are considered to be the best initialization values.
            #
            self._weights.append(uniform(-1.0, 1.0))

    def calculate_output(self):
        weighted_input = 0.0

        for idx, neuron in enumerate(self._input_neurons):
            weighted_input += neuron.calculate_output() * self._weights[idx]

        self._output = self._activation_function(weighted_input + self._bias)

        return self._output

    def update_weights(self):
        new_weights = []

        for idx, _ in enumerate(self._weights):
            weight_delta = self._input_neurons[idx].output * self._error

            new_weights.append(self._weights[idx] + weight_delta)

        self._weights = new_weights


class PredictionNeuron(NeuronCore):
    def __init__(self, activation_function):
        super(PredictionNeuron, self).__init__(activation_function)

        self.expected = 0.0

    def calculate_error(self):
        self._error = (self.expected - self._output) * DERIVATIVES[self._activation_function](self._output)


class HiddenNeuron(NeuronCore):
    def __init__(self, idx_in_layer, activation_function):
        super(HiddenNeuron, self).__init__(activation_function)

        self._idx = idx_in_layer

    def calculate_error(self):
        self._error = 0.0

        for neuron in self._output_neurons:
            self._error += neuron.error * neuron.weights[self._idx]

        self._error *= DERIVATIVES[self._activation_function](self._output)


class Brain:
    def __init__(self, inputs_count, outputs_count, args):
        network_structure = [inputs_count, 30, outputs_count]

        self._network = NeuralNetwork(network_structure, args)

    def update(self, reward, state):
        self._network.learn(state, reward)

        return self._network.predict(state)

    @property
    def score(self):
        return self._network.score()
