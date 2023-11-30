import random
import numpy as np


class NeuralNetwork(object):
    def __init__(self, nodes_input, nodes_hidden, nodes_output):
        self.nodes_input = nodes_input
        self.nodes_hidden = nodes_hidden
        self.nodes_output = nodes_output

        self.activations_input = [1.0] * self.nodes_input
        self.activations_hidden = [1.0] * self.nodes_hidden
        self.activations_output = [1.0] * self.nodes_output


        self.weights_input = [[0.0] * self.nodes_hidden for _ in range(self.nodes_input)]
        self.weights_output = [[0.0] * self.nodes_output for _ in range(self.nodes_hidden)]
        self.randomize_matrix(self.weights_input, -0.1, 0.1)
        self.randomize_matrix(self.weights_output, -2.0, 2.0)


        # self.activation_fun = tanh
        # self.activation_fun_deriv = tanh_derivate
      

    def sum_loss(self, data):
        loss = 0.0
        for item in data:
            inputs = item[0]
            targets = item[1]
            self.feed_forward(inputs)
            loss += self.calculate_loss(targets)
        inverr = 1.0 / loss
        return inverr

    def randomize_matrix(self, matrix, a, b):
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                matrix[i][j] = random.uniform(a, b)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivate(self, x):
        return 1.0 - np.tanh(x) * np.tanh(x)

    def calculate_loss(self, targets):
        loss = 0.0
        for k in range(len(targets)):
            loss += 0.5 * (targets[k] - self.activations_output[k]) ** 2
        return loss

    def feed_forward(self, inputs):
        if len(inputs) != self.nodes_input:
            print('incorrect number of inputs')

        for i in range(self.nodes_input):
            self.activations_input[i] = inputs[i]

        for j in range(self.nodes_hidden):
            self.activations_hidden[j] = self.tanh(
                sum([self.activations_input[i] * self.weights_input[i][j] for i in range(self.nodes_input)]))
        for k in range(self.nodes_output):
            self.activations_output[k] = self.tanh(
                sum([self.activations_hidden[j] * self.weights_output[j][k] for j in range(self.nodes_hidden)]))
        return self.activations_output

    def assign_weights(self, weights, I):
        io = 0
        for i in range(self.nodes_input):
            for j in range(self.nodes_hidden):
                self.weights_input[i][j] = weights[I][io][i][j]
        io = 1
        for j in range(self.nodes_hidden):
            for k in range(self.nodes_output):
                self.weights_output[j][k] = weights[I][io][j][k]

    def test_weights(self, weights, I):
        same = []
        io = 0
        for i in range(self.nodes_input):
            for j in range(self.nodes_hidden):
                if self.weights_input[i][j] != weights[I][io][i][j]:
                    same.append(('I', i, j, round(self.weights_input[i][j], 2), round(weights[I][io][i][j], 2),
                                 round(self.weights_input[i][j] - weights[I][io][i][j], 2)))

        io = 1
        for j in range(self.nodes_hidden):
            for k in range(self.nodes_output):
                if self.weights_output[j][k] != weights[I][io][j][k]:
                    same.append((('O', j, k), round(self.weights_output[j][k], 2), round(weights[I][io][j][k], 2),
                                 round(self.weights_output[j][k] - weights[I][io][j][k], 2)))
        if same:
            print(same)

    def test(self, data):
        results, targets = [], []
        for d in data:
            inputs = d[0]
            rounded = [round(i) for i in self.feed_forward(inputs)]
            if rounded == d[1]:
                result = '√ Classification Prediction is Correct '
            else:
                result = '× Classification Prediction is Wrong'
            print('{0} {1} {2} {3} {4} {5} {6}'.format(
                'Inputs:', d[0], '-->', str(self.feed_forward(inputs)).rjust(65), 'target classification', d[1],
                result))
            results += self.feed_forward(inputs)
            targets += d[1]
        return results, targets
