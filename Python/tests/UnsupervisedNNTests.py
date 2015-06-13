from unittest import TestCase
from lib.UnsupervisedNN import UnsupervisedNN

class UnsupervisedNNTests(TestCase):
    MAX_TRIES = 100
    LEARNING_RATE = 0.2
    BIAS_LEARNING_RATE = 0.01

    def test_toy_data_set(self):
        data_set = [
            [1.0, 0.9, 0.8],
            [0.0, 0.1, 0.2],
            [0.2, 0.1, 0.1],
            [0.9, 0.9, 0.8]
        ]

        expected_cluster = [
            [0, 3],
            [1, 2]
        ]

        self.assertTrue(self.train_on_data_set(data_set, expected_cluster))

    def test_only_a_single_value_matters(self):
        data_set = [
            [1.0, 0.9, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.9, 0.0],
            [0.9, 1.0, 1.0]
        ]
        expected_cluster = [
            [0, 3],
            [1, 2]
        ]

        self.assertTrue(self.train_on_data_set(data_set, expected_cluster))

    def test_three_classes(self):
        data_set = [
            [0.2, 0.9],
            [0.2, 0.8],
            [0.9, 0.9],
            [0.8, 0.9],
            [0.5, 0.1],
            [0.3, 0.0]
        ]
        expected_cluster = [
            [0, 1],
            [2, 3],
            [4, 5]
        ]

        self.assertTrue(self.train_on_data_set(data_set, expected_cluster))

    def train_on_data_set(self, data_set, expected_clusters):
        net = UnsupervisedNN(len(data_set[0]), len(expected_clusters), self.LEARNING_RATE, self.BIAS_LEARNING_RATE)
        for i in range(self.MAX_TRIES):
            net.train_many(data_set)

            if self.validate_clusters(data_set, expected_clusters, net):
                return True

        for item in data_set:
            cluster = net.get_cluster(item)
            print "{0} = {1}" % (item, cluster)

        return False

    def validate_clusters(self, data_set, expected_clusters, neural_net):
        for expectedCluster in expected_clusters:
            cluster_number = -1

            for i in expectedCluster:
                current_cluster = neural_net.get_cluster(data_set[i])
                if cluster_number == -1:
                    cluster_number = current_cluster
                elif cluster_number != current_cluster:
                    return False

        return True