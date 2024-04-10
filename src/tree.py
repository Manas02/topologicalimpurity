"""Implementation of Decision Trees with Topological Impurity"""

import numpy as np
import pandas as pd
import networkx as nx
from loguru import logger
from molecularnetwork import MolecularNetwork
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


class TopologicalDecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree_ = None

    def fit(self, X, y, adj_matrix):
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        self.tree_ = self._build_tree(X, y, adj_matrix, depth=0)
        return self

    def _build_tree(self, X, y, adj_matrix, depth):
        if depth == self.max_depth or len(np.unique(y)) == 1 or len(y) < self.min_samples_split:
            return self._create_leaf_node(y)

        best_split_feature, best_split_value = self._find_best_split(X, y, adj_matrix)
        if best_split_feature is None:
            return self._create_leaf_node(y)

        left_indices = X[:, best_split_feature] <= best_split_value
        right_indices = ~left_indices

        left_tree = self._build_tree(X[left_indices], y[left_indices], adj_matrix[left_indices][:, left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], adj_matrix[right_indices][:, right_indices], depth + 1)

        return {'split_feature': best_split_feature, 'split_value': best_split_value, 'left': left_tree, 'right': right_tree}

    def _create_leaf_node(self, y):
        unique_classes, counts = np.unique(y, return_counts=True)
        majority_class = unique_classes[np.argmax(counts)]
        return {'leaf': True, 'class': majority_class}

    def _topological_impurity(self, y, adj_matrix):
        """Calculate Topological Impurity"""
        total_samples = len(y)
        class_counts = np.bincount(y)
        
        # Count edges between different classes using vectorized operation
        edges_between_classes = np.sum(np.triu(np.logical_and(adj_matrix.toarray(), (y[:, None] != y[None, :]))))
        
        # Compute the product of class proportions
        class_proportions_product = np.prod(class_counts / total_samples)
        
        # Compute the topological impurity
        topological_impurity = class_proportions_product * (1 + edges_between_classes / np.sum(adj_matrix))

        return topological_impurity

    def _find_best_split(self, X, y, adj_matrix):
        best_impurity = np.inf
        best_split_feature = None
        best_split_value = None

        for feature in range(X.shape[1]):
            unique_values = np.unique(X[:, feature])
            for split_value in unique_values:
                left_indices = X[:, feature] <= split_value
                right_indices = ~left_indices

                left_impurity = self._topological_impurity(y[left_indices], adj_matrix[left_indices][:, left_indices])
                right_impurity = self._topological_impurity(y[right_indices], adj_matrix[right_indices][:, right_indices])

                child_impurity = left_indices.sum() / len(X) * left_impurity + right_indices.sum() / len(X) * right_impurity
                if child_impurity < best_impurity:
                    best_impurity = child_impurity
                    best_split_feature = feature
                    best_split_value = split_value

        return best_split_feature, best_split_value

    def predict(self, X):
        predictions = np.zeros(len(X), dtype=int)
        for i, sample in enumerate(X):
            predictions[i] = self._predict_sample(sample, self.tree_)
        return predictions

    def _predict_sample(self, sample, node):
        if node.get('leaf'):
            return node['class']
        
        if sample[node['split_feature']] <= node['split_value']:
            return self._predict_sample(sample, node['left'])
        else:
            return self._predict_sample(sample, node['right'])


if __name__ == "__main__":
    df = pd.read_csv('../data/CHEMBL1267247.csv')

    # train
    train_df = df[df["split"] == "train"]
    train_smiles_list = train_df["canonical_smiles"].values
    train_classes = train_df["active"].values
    network = MolecularNetwork(descriptor="morgan2", sim_metric="tanimoto", sim_threshold=0.5)
    graph = network.create_graph(train_smiles_list, train_classes)
    X_train = np.array([graph.nodes[i]['fp'] for i in graph.nodes])
    y_train = np.array([int(graph.nodes[i]['categorical_label']) for i in graph.nodes])
    A_train = nx.adjacency_matrix(graph, weight=None)

    # test
    test_df = df[df["split"] == "test"]
    test_smiles_list = test_df["canonical_smiles"].values
    test_classes = test_df["active"].values
    test_network = MolecularNetwork(descriptor="morgan2", sim_metric="tanimoto", sim_threshold=0.5)
    test_graph = test_network.create_graph(test_smiles_list, test_classes)
    X_test = np.array([test_graph.nodes[i]['fp'] for i in test_graph.nodes])
    y_test = np.array([int(test_graph.nodes[i]['categorical_label']) for i in test_graph.nodes])
    A_test = nx.adjacency_matrix(test_graph, weight=None)

    # Train TopologicalDecisionTreeClassifier
    topo_clf = TopologicalDecisionTreeClassifier()
    topo_clf.fit(X_train, y_train, A_train)

    # Train DecisionTreeClassifier
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # Make predictions
    topo_pred = topo_clf.predict(X_test)
    pred = clf.predict(X_test)

    # Generate classification report
    print("Topological Decision Tree Classifier:")
    print(classification_report(y_test, topo_pred))

    print("\n\nDecision Tree Classifier:")
    print(classification_report(y_test, pred))