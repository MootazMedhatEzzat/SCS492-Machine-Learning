import numpy as np
from sklearn import model_selection
from sklearn import tree
from sklearn import metrics


class Experiment:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.results = []

    def run_experiment(self, train_test_split_ratios=[25], experiments_number=5, fixed=False):
        for split_ratio in train_test_split_ratios:
            if fixed:
                for experiment_number in range(experiments_number):
                    x_train, x_test, y_train, y_test = model_selection.train_test_split(
                        self.x, self.y, train_size=split_ratio/100, random_state=experiment_number+1
                    )

                    model = tree.DecisionTreeClassifier(criterion="entropy")
                    model.fit(x_train, y_train)

                    y_predicted = model.predict(x_test)

                    accuracy = metrics.accuracy_score(y_test, y_predicted)
                    tree_depth = model.tree_.max_depth
                    tree_size = model.tree_.node_count
                    tree_leaf_nodes = model.tree_.n_leaves

                    self.results.append([accuracy, tree_depth, tree_size, tree_leaf_nodes])
            else:
                accuracies = []
                tree_sizes = []

                for experiment_number in range(experiments_number):
                    x_train, x_test, y_train, y_test = model_selection.train_test_split(
                        self.x, self.y, train_size=split_ratio/100, random_state=experiment_number+1
                    )

                    model = tree.DecisionTreeClassifier(criterion="entropy")
                    model.fit(x_train, y_train)

                    y_predicted = model.predict(x_test)

                    accuracy = metrics.accuracy_score(y_test, y_predicted)
                    accuracies.append(accuracy)

                    tree_size = model.tree_.node_count
                    tree_sizes.append(tree_size)

                mean_accuracy = np.mean(accuracies)
                max_accuracy = np.max(accuracies)
                min_accuracy = np.min(accuracies)

                mean_tree_size = np.mean(tree_sizes)
                max_tree_size = np.max(tree_sizes)
                min_tree_size = np.min(tree_sizes)

                self.results.append([mean_accuracy, max_accuracy, min_accuracy, mean_tree_size, max_tree_size,
                                     min_tree_size])

        return self.results
