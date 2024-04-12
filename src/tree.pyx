import numpy as np
cimport numpy as np

cdef class TopologicalDecisionTreeClassifier:
    cdef int max_depth, min_samples_split, min_samples_leaf
    cdef float min_impurity_reduction
    cdef object tree_
    cdef np.ndarray classes_
    cdef int n_classes_

    def __init__(self, int max_depth=-1, int min_samples_split=2, int min_samples_leaf=1, float min_impurity_reduction=0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_reduction = min_impurity_reduction 
        self.tree_ = None

    cpdef fit(self, np.ndarray X, np.ndarray y, np.ndarray adj_matrix):
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        assert self.n_classes_ >= 2, f"At least 2 classes should be present in categorical label (y), only {self.n_classes_} found"
        
        self.tree_ = self._build_tree(X, y, adj_matrix, 0)
        return self

    cdef _build_tree(self, np.ndarray X, np.ndarray y, np.ndarray adj_matrix, int depth):
        if (depth == self.max_depth or 
            len(np.unique(y)) == 1 or 
            len(y) <= self.min_samples_split):
            return self._create_leaf_node(y)

        best_split_feature, best_split_value = self._find_best_split(X, y, adj_matrix)
        if best_split_feature is None:
            return self._create_leaf_node(y)

        left_indices = X[:, best_split_feature] <= best_split_value
        right_indices = ~left_indices

        left_tree = self._build_tree(X[left_indices], y[left_indices], adj_matrix[left_indices][:, left_indices], depth + 1)
        
        right_tree = self._build_tree(X[right_indices], y[right_indices], adj_matrix[right_indices][:, right_indices], depth + 1)

        return {'split_feature': best_split_feature, 'split_value': best_split_value, 'left': left_tree, 'right': right_tree}

    cdef _create_leaf_node(self, np.ndarray y):
        assert y.shape[0] > 0, "y is empty, how to create a leaf node with no data?"
        unique_classes, counts = np.unique(y, return_counts=True)
        majority_class = unique_classes[np.argmax(counts)]
        return {'leaf': True, 'class': majority_class}

    cdef _topological_impurity(self, np.ndarray y, np.ndarray adj_matrix):
        """Calculate Topological Impurity"""
        total_samples = len(y)
        class_counts = np.bincount(y)
        # Count edges between different classes using vectorized operation
        edges_between_classes = np.sum(np.triu(np.logical_and(adj_matrix, (y[:, None] != y[None, :]))))
        # Compute the product of class proportions
        class_proportions_product = np.prod(class_counts / total_samples)

        total_edges = np.sum(adj_matrix)
        
        # Avoid ZeroDivisionError
        if total_edges == 0:
            return class_proportions_product

        # Compute the topological impurity
        topological_impurity = class_proportions_product * (1 + edges_between_classes / total_edges)
        return topological_impurity

    cdef _find_best_split(self, np.ndarray X, np.ndarray y, np.ndarray adj_matrix):
        best_impurity_reduction = -np.inf
        best_split_feature = None
        best_split_value = None
        parent_impurity = self._topological_impurity(y, adj_matrix)

        for feature in range(X.shape[1]):
            unique_values = np.unique(X[:, feature])
            # trivial optimization: at least 2 unique values required to split data
            if len(unique_values) < 2:
                continue

            for split_value in unique_values:
                left_indices = X[:, feature] <= split_value
                right_indices = ~left_indices

                if (left_indices.sum() < self.min_samples_leaf or 
                    right_indices.sum() < self.min_samples_leaf):
                    continue

                left_impurity = self._topological_impurity(y[left_indices], adj_matrix[left_indices][:, left_indices])
                right_impurity = self._topological_impurity(y[right_indices], adj_matrix[right_indices][:, right_indices])

                child_impurity = left_indices.sum() / len(y) * left_impurity + right_indices.sum() / len(y) * right_impurity
                impurity_reduction = parent_impurity - child_impurity  # max this

                if (impurity_reduction > best_impurity_reduction and 
                    impurity_reduction > self.min_impurity_reduction):  # at least > 0 impurity reduction
                    best_impurity_reduction = impurity_reduction
                    best_split_feature = feature
                    best_split_value = split_value

        return best_split_feature, best_split_value

    cpdef predict(self, np.ndarray X):
        predictions = np.zeros(len(X), dtype=np.int_)
        for i, sample in enumerate(X):
            predictions[i] = self._predict_sample(sample, self.tree_)
        return predictions

    cdef _predict_sample(self, np.ndarray sample, object node):
        if node.get('leaf'):
            return node['class']

        if sample[node['split_feature']] <= node['split_value']:
            return self._predict_sample(sample, node['left'])
        else:
            return self._predict_sample(sample, node['right'])
