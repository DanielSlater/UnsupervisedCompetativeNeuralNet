from random import uniform

def default_weight_init_func():
    return uniform(-0.5, 0.5)

class UnsupervisedNN(object):
    def __init__(self, size_of_inputs,
                 number_of_clusters,
                 learning_rate,
                 conscience_rate,
                 weight_init_func=default_weight_init_func):
        self.learning_rate = learning_rate
        self.conscience_rate = conscience_rate
        self.__hidden_nodes = [0.0]*number_of_clusters
        self.__conscience = [0.0] * number_of_clusters
        self.__connections = [[weight_init_func() for j in range(number_of_clusters)] for i in range(size_of_inputs)]

    def __feed_forward(self, inputs):
        for h in range(len(self.__hidden_nodes)):
            activation = self.__conscience[h]
            for i in range(len(self.__connections)):
                activation += inputs[i]*self.__connections[i][h]
            self.__hidden_nodes[h] = activation

    def __get_winner(self):
        return self.__hidden_nodes.index(max(self.__hidden_nodes))

    def get_cluster(self, inputs):
        self.__feed_forward(inputs)
        return self.__get_winner()

    def train_many(self, data_set):
        for inputs in data_set:
            self.train(inputs)

    def train(self, inputs):
        winner = self.get_cluster(inputs)
        for i in range(len(self.__connections)):
            weight = self.__connections[i][winner]
            self.__connections[i][winner] = weight + self.learning_rate*(inputs[i]-weight)

        self.__conscience[winner] -= self.conscience_rate
        return winner