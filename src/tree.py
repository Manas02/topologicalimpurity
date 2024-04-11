"""Implementation of Decision Trees with Topological Impurity"""

import numpy as np

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
        print(f"{left_indices.sum() = }, {right_indices.sum() = }")
        left_tree = self._build_tree(X[left_indices], y[left_indices], adj_matrix[left_indices][:, left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], adj_matrix[right_indices][:, right_indices], depth + 1)

        return {'split_feature': best_split_feature, 'split_value': best_split_value, 'left': left_tree, 'right': right_tree}

    def _create_leaf_node(self, y):
        if y.any():
            unique_classes, counts = np.unique(y, return_counts=True)
            majority_class = unique_classes[np.argmax(counts)]
            return {'leaf': True, 'class': majority_class}
        else:
            return {'leaf': True, 'class': 0} # FIXME ??

    def _topological_impurity(self, y, adj_matrix):
        """Calculate Topological Impurity"""
        total_samples = len(y)
        class_counts = np.bincount(y)
        
        # Count edges between different classes using vectorized operation
        edges_between_classes = np.sum(np.triu(np.logical_and(adj_matrix.toarray(), (y[:, None] != y[None, :]))))
        
        # Compute the product of class proportions
        class_proportions_product = np.prod(class_counts / total_samples)
        
        total_edges = np.sum(adj_matrix)
        # Avoid ZeroDivisionError
        if total_edges == 0:
            return class_proportions_product

        # Compute the topological impurity
        topological_impurity = class_proportions_product * (1 + edges_between_classes / total_edges)
        return topological_impurity

    def _find_best_split(self, X, y, adj_matrix):
        best_impurity_reduction = -np.inf
        best_split_feature = None
        best_split_value = None
        parent_impurity = self._topological_impurity(y, adj_matrix)

        for feature in range(X.shape[1]):
            unique_values = np.unique(X[:, feature])
            for split_value in unique_values:
                left_indices = X[:, feature] <= split_value
                right_indices = ~left_indices

                left_impurity = self._topological_impurity(y[left_indices], adj_matrix[left_indices][:, left_indices])
                right_impurity = self._topological_impurity(y[right_indices], adj_matrix[right_indices][:, right_indices])

                child_impurity = left_indices.sum() / len(X) * left_impurity + right_indices.sum() / len(X) * right_impurity
                impurity_reduction = parent_impurity - child_impurity # max this
                
                if impurity_reduction > best_impurity_reduction:
                    best_impurity_reduction = impurity_reduction
                    best_split_feature = feature
                    best_split_value = split_value

        return best_split_feature, best_split_value

    def predict(self, X):
        predictions = np.zeros(len(X), dtype=int)
        for i, sample in enumerate(X):
            predictions[i] = self._predict_sample(sample, self.tree_)
        return predictions

    def predict_proba(self, X):
        proba = np.zeros((len(X), self.n_classes_))
        for i, sample in enumerate(X):
            proba[i] = self._predict_proba_sample(sample, self.tree_)
        return proba

    def _predict_sample(self, sample, node):
        if node.get('leaf'):
            return node['class']
        
        if sample[node['split_feature']] <= node['split_value']:
            return self._predict_sample(sample, node['left'])
        else:
            return self._predict_sample(sample, node['right'])

    def _predict_proba_sample(self, sample, node):
        if node.get('leaf'):
            class_proba = np.zeros(self.n_classes_)
            class_proba[node['class']] = 1.0
            return class_proba
        
        if sample[node['split_feature']] <= node['split_value']:
            return self._predict_proba_sample(sample, node['left'])
        else:
            return self._predict_proba_sample(sample, node['right'])