from django.shortcuts import render
import pandas as pd
import numpy as np
from io import StringIO
from django.views import View
from operator import itemgetter
import random
import matplotlib.pyplot as plt
import numpy as np
import time
from .neural_network import NeuralNetwork

class home_view(View):
    def get(self, request):
        context = {}
        return render(request, 'adminpage.html', context)

class content_view(View):
    def __init__(self):
        self.graphical_error_scale = 300
        self.max_iterations = 100
        self.pop_size = 10
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.nodes_input, self.nodes_hidden, self.nodes_output = 4, 6, 1
        self.x_train, self.x_test, self.y_train, self.y_test = self.read_data()
        self.iris_train_data, self.iris_test_data = self.pre_processing(self.x_train, self.x_test, self.y_train, self.y_test)

    def week(self, i):
        switcher={
              'Iris-setosa': 0,
              'Iris-versicolor': 1,
              'Iris-virginica': 2
            }
        return switcher.get(i)

    def read_data(self):
        IRIS_TRAIN_URL = 'iris/iris_training.csv'
        IRIS_TEST_URL = 'iris/iris_test.csv'

        names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'species']
        train = pd.read_csv(IRIS_TRAIN_URL, names=names, skiprows=1)
        test = pd.read_csv(IRIS_TEST_URL, names=names, skiprows=1)
        # data_test = pd.read_csv(IRIS_URL, names=names)

        # a = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        # data_test = data_test.replace('Iris-setosa', -1)
        # data_test = data_test.replace('Iris-versicolor', 0)
        # data_test = data_test.replace('Iris-virginica', 1)
        # setosa = data_test[0:50]
        # versicolor = data_test[50:100]
        # virginica = data_test[100:150]


        x_train = train.drop('species', axis=1)
        x_test = test.drop('species', axis=1)

        y_train = train.species
        y_test = test.species

        for i in range(0, 3):
            y_train = y_train.replace(i, i - 1)
            y_test = y_test.replace(i, i - 1)

        return x_train, x_test, y_train, y_test

    def pre_processing(self, x_train, x_test, y_train, y_test):
        x_train_list = np.array(x_train).tolist()
        x_test_list = np.array(x_test).tolist()

        y_train_one_list = np.array(y_train).tolist()
        y_train_list = []
        for i in y_train_one_list:
            y_train_list.append([i])

        y_test_one_list = np.array(y_test).tolist()
        y_test_list = []
        for i in y_test_one_list:
            y_test_list.append([i])

        iris_train_data = []
        for item in list(zip(x_train_list, y_train_list)):
            iris_train_data.append(list(item))

        iris_test_data = []
        for item in list(zip(x_test_list, y_test_list)):
            iris_test_data.append(list(item))

        return iris_train_data, iris_test_data

    def calculate_fit(self, loss):
        total, fitnesses = sum(loss), []
        for i in range(len(loss)):
            fitnesses.append(loss[i] / total)
        return fitnesses

    def pair_pop(self, iris_data, pop):
        weights, loss = [], []

        for individual_obj in pop:
            weights.append([individual_obj.weights_input, individual_obj.weights_output])
            loss.append(individual_obj.sum_loss(data=iris_data))

        fitnesses = self.calculate_fit(loss)
        for i in range(int(self.pop_size)):
            print(str(i).zfill(2), '1/sum(MSEs)', str(loss[i]).rjust(15), str(
                int(loss[i] * self.graphical_error_scale) * '-').rjust(20), 'fitness'.rjust(12), str(fitnesses[i]).rjust(
                17), str(int(fitnesses[i] * 1000) * '-').rjust(20))
        del pop

        return zip(weights, loss, fitnesses)

    def roulette(self, fitness_scores):
        cumalative_fitness = 0.0
        r = random.random()
        for i in range(len(fitness_scores)):
            cumalative_fitness += fitness_scores[i]
            if cumalative_fitness > r:
                return i

    def iterate_pop(self, ranked_pop):
        ranked_weights = [item[0] for item in ranked_pop]
        fitness_scores = [item[-1] for item in ranked_pop]
        new_pop_weight = [eval(repr(x)) for x in ranked_weights[:int(self.pop_size * 0.15)]]

        while len(new_pop_weight) <= self.pop_size:
            ch1, ch2 = [], []
            index1 = self.roulette(fitness_scores)
            index2 = self.roulette(fitness_scores)
            while index1 == index2:
                index2 = self.roulette(fitness_scores)
            ch1.extend(eval(repr(ranked_weights[index1])))
            ch2.extend(eval(repr(ranked_weights[index2])))
            if random.random() < self.crossover_rate:
                ch1, ch2 = self.crossover(ch1, ch2)
            self.mutate(ch1)
            self.mutate(ch2)
            new_pop_weight.append(ch1)
            new_pop_weight.append(ch2)
        return new_pop_weight

    def crossover(self, m1, m2):
        # ni*nh+nh*no  = total weights
        r = random.randint(0, (self.nodes_input * self.nodes_hidden) + (self.nodes_hidden * self.nodes_output))
        output1 = [[[0.0] * self.nodes_hidden] * self.nodes_input, [[0.0] * self.nodes_output] * self.nodes_hidden]
        output2 = [[[0.0] * self.nodes_hidden] * self.nodes_input, [[0.0] * self.nodes_output] * self.nodes_hidden]
        for i in range(len(m1)):
            for j in range(len(m1[i])):
                for k in range(len(m1[i][j])):
                    if r >= 0:
                        output1[i][j][k] = m1[i][j][k]
                        output2[i][j][k] = m2[i][j][k]
                    elif r < 0:
                        output1[i][j][k] = m2[i][j][k]
                        output2[i][j][k] = m1[i][j][k]
                    r -= 1
        return output1, output2

    def mutate(self, m):
        for i in range(len(m)):
            for j in range(len(m[i])):
                for k in range(len(m[i][j])):
                    if random.random() < self.mutation_rate:
                        m[i][j][k] = random.uniform(-2.0, 2.0)

    def rank_pop(self, new_pop_weight, pop):
        loss, copy = [], []
        pop = [NeuralNetwork(self.nodes_input, self.nodes_hidden, self.nodes_output) for _ in range(self.pop_size)]
        for i in range(self.pop_size):
            copy.append(new_pop_weight[i])

        for i in range(self.pop_size):
            pop[i].assign_weights(new_pop_weight, i)
            pop[i].test_weights(new_pop_weight, i)

        for i in range(self.pop_size):
            pop[i].test_weights(new_pop_weight, i)

        
        paired_pop = self.pair_pop(self.iris_train_data, pop)

        ranked_pop = sorted(paired_pop, key=itemgetter(-1), reverse=True)
        loss = [eval(repr(x[1])) for x in ranked_pop]
        return ranked_pop, eval(repr(ranked_pop[0][1])), float(sum(loss)) / float(len(loss))

    def randomize_matrix(self, matrix, a, b):
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                matrix[i][j] = random.uniform(a, b)

    def get(self, request):
        context = {}

        if 'graphical_error_scale' in request.GET:
            self.graphical_error_scale = int(request.GET.get('graphical_error_scale'))
            context['graphical_error_scale'] = self.graphical_error_scale
        if 'max_iterations' in request.GET:
            self.max_iterations = int(request.GET.get('max_iterations'))
            context['max_iterations'] = self.max_iterations
        if 'pop_size' in request.GET:
            self.pop_size = int(request.GET.get('pop_size'))
            context['pop_size'] = self.pop_size
        if 'mutation_rate' in request.GET:
            self.mutation_rate = float(request.GET.get('mutation_rate'))
            context['mutation_rate'] = self.mutation_rate
        if 'crossover_rate' in request.GET:
            self.crossover_rate = float(request.GET.get('crossover_rate'))
            context['crossover_rate'] = self.crossover_rate
        if 'nodes_input' in request.GET:
            self.nodes_input = int(request.GET.get('nodes_input'))
            context['nodes_input'] = self.nodes_input
        if 'nodes_hidden' in request.GET:
            self.nodes_hidden = int(request.GET.get('nodes_hidden'))
            context['nodes_hidden'] = self.nodes_hidden
        if 'nodes_output' in request.GET:
            self.nodes_output = int(request.GET.get('nodes_output'))
            context['nodes_output'] = self.nodes_output

        pop = [NeuralNetwork(self.nodes_input, self.nodes_hidden, self.nodes_output) for i in range(self.pop_size)] 

        paired_pop = self.pair_pop(self.iris_train_data, pop)

        ranked_pop = sorted(paired_pop, key=itemgetter(-1), reverse=True)  

        iters = 0
        tops, avgs = [], []

        while iters != self.max_iterations:

            print('Iteration'.rjust(150), iters)

            new_pop_weight = self.iterate_pop(ranked_pop)
            ranked_pop, toperr, avgerr = self.rank_pop(new_pop_weight, pop)

            tops.append(toperr)
            avgs.append(avgerr)
            iters += 1



        tester = NeuralNetwork(self.nodes_input, self.nodes_hidden, self.nodes_output)
        fittestWeights = [x[0] for x in ranked_pop]
        tester.assign_weights(fittestWeights, 0)
        results, targets = tester.test(self.iris_test_data)
        x = np.arange(0, 150)
        title2 = 'Test after ' + str(iters) + ' iterations'

        fig1 = plt.figure()

        plt.title(title2)
        plt.ylabel('Node output')
        plt.xlabel('Instances')
        plt.plot(results, 'xr', linewidth=0.5)
        plt.plot(targets, 's', color='black', linewidth=3)

        imgdata1 = StringIO()
        fig1.savefig(imgdata1, format='svg')
        data1 = imgdata1.getvalue()

        context['image1'] = data1

        fig2 = plt.figure(2)
        plt.annotate(text='Target Values', xy=(110, 0), color='black', family='sans-serif', size='small')
        plt.annotate(text='Test Values', xy=(110, 0.5), color='red', family='sans-serif', size='small', weight='bold')
        plt.subplot(121)
        plt.title('Top individual error evolution')
        plt.ylabel('Inverse error')
        plt.xlabel('Iterations')
        plt.plot(tops, '-g', linewidth=1)
        plt.subplot(122)
        plt.plot(avgs, '-g', linewidth=1)
        plt.title('Population average error evolution')
        plt.ylabel('Inverse error')
        plt.xlabel('Iterations')

        imgdata2 = StringIO()
        fig2.savefig(imgdata2, format='svg')
        data2 = imgdata2.getvalue()

        context['image2'] = data2

        plt.close()

        # print('max_iterations', max_iterations, 'pop_size', pop_size, 'pop_size*0.15', int(
        #     pop_size * 0.15), 'mutation_rate', mutation_rate, 'crossover_rate', crossover_rate,
        #       'nodes_input, nodes_hidden, nodes_output', nodes_input, nodes_hidden, nodes_output)
        return render(request, 'adminpage2.html', context)
