#cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, cdivision=True
from typing import Any, Optional, Union
import numpy as np
cimport numpy as np


cdef class TopologicalDecisionTreeClassifier:
    cdef int max_depth
    cdef int min_samples_split
    cdef int min_samples_leaf
    cdef float min_impurity_reduction
    cdef float mol_net_threshold
    cdef dict tree_
    cdef np.ndarray classes_
    cdef int n_classes_

    def __init__(self, max_depth: int = -1, 
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1, 
                 min_impurity_reduction: float = 0,
                 mol_net_threshold:float = 0.0):
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_reduction = min_impurity_reduction
        self.mol_net_threshold = mol_net_threshold
        self.tree_ = None

    cpdef fit(self, np.ndarray X, np.ndarray y, np.ndarray  adj_matrix):
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        adj_matrix = (adj_matrix > self.mol_net_threshold).astype(int)
        assert self.n_classes_ >= 2, f"At least 2 classes should be present in categorical label (y), only {self.n_classes_} found"
        self.tree_ = self._build_tree(X, y, adj_matrix, depth=0)
        return self.tree_
    
    cdef dict _build_tree(self, np.ndarray X, np.ndarray y, np.ndarray  adj_matrix, int depth):
        best_split_feature: Union[int, None] = None 
        best_split_value: Union[float, None] = None
        topo_impurity = self._topological_impurity(y, adj_matrix)
        p_act = (y == 1).sum()/len(y)

        if (depth == self.max_depth or 
            len(np.unique(y)) == 1 or 
            len(y) <= self.min_samples_split):
            leaf_node = self._create_leaf_node(y)
            leaf_node['topological_impurity'] = topo_impurity
            leaf_node['P_active'] = p_act
            return leaf_node

        best_split_feature, best_split_value = self._find_best_split(X, y, adj_matrix)
        if best_split_feature is None:
            leaf_node = self._create_leaf_node(y)
            leaf_node['topological_impurity'] = topo_impurity
            leaf_node['P_active'] = p_act
            return leaf_node

        left_indices = X[:, best_split_feature] <= best_split_value
        right_indices = ~left_indices

        left_tree = self._build_tree(X[left_indices], y[left_indices], adj_matrix[left_indices][:, left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], adj_matrix[right_indices][:, right_indices], depth + 1)

        return {'split_feature': best_split_feature, 
                'split_value': best_split_value, 
                'left': left_tree, 
                'right': right_tree,
                'topological_impurity': topo_impurity,
                'P_active': p_act}
    

    cdef dict _create_leaf_node(self, np.ndarray y):
        assert y.shape[0] > 0, "y is empty, how to create a leaf node with no data?"
        unique_classes, counts = np.unique(y, return_counts=True)
        majority_class = unique_classes[np.argmax(counts)]
        return {"leaf": True, "class": majority_class}

    cpdef np.float64_t _topological_impurity(self, np.ndarray y, np.ndarray adj_matrix):
        total_samples = len(y)
        class_counts = np.bincount(y, minlength=self.n_classes_)

        # Count edges between different classes using vectorized operation 
        edges_between_classes = np.sum(adj_matrix * (y[:, None] != y[None, :])) / 2
        # Compute the product of class proportions
        class_proportions_product = np.prod(class_counts / total_samples)
        # Total number of edges in the graph
        total_edges = adj_matrix.sum() / 2
        
        # Avoid ZeroDivisionError
        if total_edges == 0:
            return class_proportions_product

        # Compute the topological impurity
        topological_impurity = class_proportions_product * (1 + edges_between_classes / total_edges)
        return topological_impurity

    cpdef _find_best_split(self, np.ndarray X, np.ndarray y, np.ndarray adj_matrix):
        best_impurity_reduction = float("-inf")
        best_split_feature: int|None = None 
        best_split_value: float|None = None
        parent_impurity = self._topological_impurity(y, adj_matrix)

        for feature in range(X.shape[1]):
            unique_values = np.unique(X[:, feature])
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
                impurity_reduction = parent_impurity - child_impurity # max this

                if (impurity_reduction > best_impurity_reduction and 
                    impurity_reduction > self.min_impurity_reduction):
                
                    best_impurity_reduction = impurity_reduction
                    best_split_feature = feature
                    best_split_value = split_value

        return best_split_feature, best_split_value

    cpdef np.ndarray predict(self, np.ndarray X):
        predictions = np.zeros(len(X), dtype=int)
        for i, sample in enumerate(X):
            predictions[i] = self._predict_sample(sample, self.tree_)
        return predictions

    cdef int _predict_sample(self, np.ndarray sample, dict node):
        if node is None:
            raise ValueError("Expected a non-None dictionary node")
        
        if node.get("leaf"):
            return node["class"]

        if sample[node["split_feature"]] <= node["split_value"]:
            return self._predict_sample(sample, node["left"])
        else:
            return self._predict_sample(sample, node["right"])
