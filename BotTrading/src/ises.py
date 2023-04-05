import numpy as np
import pandas as pd
import time 

from data.make_dataset import base_csv


## Variáveis de Controle
rates_file = 'E:\\Projetos\\BotTrading-Quantil\\BotTrading\\src\\6A1.csv'
window_size = 10

## Definições
class Deep_Evolution_Strategy:
    inputs = None
    def __init__(self, weights, reward_function, population_size, sigma, learning_rate):
        self.weights = weights
        self.reward_function = reward_function
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate

    def _get_weight_from_population(self, weights, population):
        weights_population = []
        for index, i in enumerate(population):
            jittered = self.sigma * i
            weights_population.append(weights[index] + jittered)
        return weights_population

    def get_weights(self):
        return self.weights

    def train(self, epoch = 100, print_every = 1):
        lasttime = time.time()
        for i in range(epoch):
            population = []
            rewards = np.zeros(self.population_size)
            for k in range(self.population_size):
                x = []
                for w in self.weights:
                    x.append(np.random.randn(*w.shape))
                population.append(x)
            for k in range(self.population_size):
                weights_population = self._get_weight_from_population(
                    self.weights, population[k]
                )
                rewards[k] = self.reward_function(weights_population)
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-7)
            for index, w in enumerate(self.weights):
                A = np.array([p[index] for p in population])
                self.weights[index] = (
                    w
                    + self.learning_rate
                    / (self.population_size * self.sigma)
                    * np.dot(A.T, rewards).T
                )
            if (i + 1) % print_every == 0:
                print(
                    'iteracao %d. recompensa: %f'
                    % (i + 1, self.reward_function(self.weights))
                )
        print('Tempo de Treinamento:', time.time() - lasttime, 'segundos')        

class Model:
    def __init__(self, input_size, layer_size, output_size):
        self.weights = [np.random.randn(input_size, layer_size),
            np.random.randn(layer_size, output_size),
            np.random.randn(1, layer_size)]

    def predict(self, inputs):
        feed = np.dot(inputs, self.weights[0]) + self.weights[-1]
        decision = np.dot(feed, self.weights[1])
        return decision

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights
        
class Agent:

    POPULATION_SIZE = 15
    SIGMA = 0.1
    LEARNING_RATE = 0.03

    def __init__(self, model, data, price, initial_money):
        self.model = model
        #self.window_size = window_size
        #self.half_window = window_size // 2
        self.data = data
        self.price = price
        self.initial_money = initial_money
        self.es = Deep_Evolution_Strategy(
            self.model.get_weights(),
            self.get_reward,
            self.POPULATION_SIZE,
            self.SIGMA,
            self.LEARNING_RATE,
        )

    def act(self, sequence):
        decision = self.model.predict(np.array(sequence))
        return np.argmax(decision[0])
    
    def get_reward(self, weights):
        initial_money = self.initial_money
        starting_money = initial_money
        self.model.weights = weights
        inventory = []
        position = False
        
        for t in range(0, self.data.shape[0]):
            action = self.act(self.data[t])
                        
            if action == 1 and starting_money >= self.price[t] and position == False:
                position = True
                inventory.append(self.price[t])
                starting_money = starting_money - self.price[t]
                                                
            elif action == 2 and len(inventory) and position:
                position = False
                bought_price = inventory.pop(0)
                starting_money = starting_money + self.price[t]
                        
        d = starting_money - initial_money
        d = d / np.abs(self.data[1:, -1]).sum()
        return d

    def fit(self, iterations, checkpoint):
        self.es.train(iterations, print_every = checkpoint)

## Leitura de Dados
df = pd.read_csv(rates_file)

## Definição de Simulação
df['retorno'] = df.close.diff()
df.dropna(inplace=True)
simulation_data = np.lib.stride_tricks.sliding_window_view(df.retorno.values, window_size)

## Treinamento
model = Model(input_size=simulation_data.shape[1], layer_size=2, output_size=3)
agent = Agent(model, simulation_data, df.close[-simulation_data.shape[0]:].values, 124.5)
agent.fit(iterations=100, checkpoint=1)

