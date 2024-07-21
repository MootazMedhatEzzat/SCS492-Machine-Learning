import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k=3):
        self.k = k
        self.x_train = None
        self.y_train = None

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

    @staticmethod
    def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def calculate_distances(self, x):
        distances = []
        for x_train in self.x_train:
            distance = self.euclidean_distance(x, x_train)
            distances.append(distance)
        return distances

    def get_k_nearest_labels(self, distances):
        # sort distances in ascending order and return the indices of the first k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]                  # the first k smallest distances
        k_nearest_labels = []
        for k_nearest_label in k_indices:
            k_nearest_labels.append(self.y_train[k_nearest_label])
        return k_nearest_labels

    @staticmethod
    def get_most_common_label(k_nearest_labels):
        # count the occurrences of each unique class label among the k nearest neighbors,
        # returning a list of tuples where each tuple represents a class label and the number of times it appears
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]  # if there's a tie, pick the one that comes first in the Train file

    def predict(self, x_test):
        predictions = []
        for x in x_test:
            # calculate the distances between x and all features examples in the training set
            distances = self.calculate_distances(x)
            # get the class labels of the k nearest neighbor training samples
            k_nearest_labels = self.get_k_nearest_labels(distances)
            #
            predicted_label = self.get_most_common_label(k_nearest_labels)
            # append the predicted label to the predictions list
            predictions.append(predicted_label)
        return predictions
