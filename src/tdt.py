from typing import Any, Optional, Union
import numpy as np
from loguru import logger
from scipy.sparse import csr_matrix # type: ignore


class TopologicalDecisionTreeClassifier:
    def __init__(self, max_depth: int = -1, 
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1, 
                 min_impurity_reduction: float = 0):
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_reduction = min_impurity_reduction 
        self.tree_: Optional[dict[str, Any]] = None

        
    
    def fit(self, X: np.ndarray, y: np.ndarray, adj_matrix: csr_matrix) -> 'TopologicalDecisionTreeClassifier':
        self.classes_ = np.unique(y)
        logger.debug(f'Unique classes : {self.classes_}')
        self.n_classes_ = len(self.classes_)
        assert self.n_classes_ >= 2, f"At least 2 classes should be present in categorical label (y), only {self.n_classes_} found"
        logger.debug(f'Total number of unique classes : {self.n_classes_}')
        self.tree_ = self._build_tree(X, y, adj_matrix, depth=0)
        return self
    

    def _build_tree(self, X: np.ndarray, y: np.ndarray, adj_matrix: csr_matrix, depth: int) -> dict[str, Any]:
        best_split_feature: Union[int,None] = None 
        best_split_value:Union[float,None] = None
        topo_impurity = self._topological_impurity(y, adj_matrix)

        if (depth == self.max_depth or 
            len(np.unique(y)) == 1 or 
            len(y) <= self.min_samples_split):
            leaf_node = self._create_leaf_node(y)
            leaf_node['impurity'] = topo_impurity
            logger.debug(f"Creating Leaf Node with y ({y.shape})")
            return leaf_node

        best_split_feature, best_split_value = self._find_best_split(X, y, adj_matrix)
        if best_split_feature is None:
            logger.warning(f"{best_split_feature = } for {y.shape = } & {adj_matrix.shape = }")
            leaf_node = self._create_leaf_node(y)
            # Save impurity in leaf node
            leaf_node['impurity'] = topo_impurity
            return leaf_node

        left_indices = X[:, best_split_feature] <= best_split_value
        right_indices = ~left_indices

        logger.debug(f"Building left tree with {left_indices.sum()} samples")
        left_tree = self._build_tree(X[left_indices], y[left_indices], adj_matrix[left_indices][:, left_indices], depth + 1)
        
        logger.debug(f"Building right tree with {right_indices.sum()} samples")
        right_tree = self._build_tree(X[right_indices], y[right_indices], adj_matrix[right_indices][:, right_indices], depth + 1)

        return {'split_feature': best_split_feature, 
                'split_value': best_split_value, 
                'left': left_tree, 
                'right': right_tree,
                'topological impurity': topo_impurity}
    

    def _create_leaf_node(self, y: np.ndarray) -> dict[str, Any]:
        assert y.shape[0] > 0, "y is empty, how to create a leaf node with no data ?"
        unique_classes, counts = np.unique(y, return_counts=True)
        majority_class = unique_classes[np.argmax(counts)]
        return {'leaf': True, 'class': majority_class}


    def _topological_impurity(self, y: np.ndarray, adj_matrix: csr_matrix) -> np.float64:
        """Calculate Topological Impurity"""
        total_samples = len(y)
        class_counts = np.bincount(y)
        logger.debug(f"Topological Impurity = ∑ P(class_i) * (1 + |E_class_ij|/|E|)")

        # Count edges between different classes using vectorized operation 
        edges_between_classes = np.sum(adj_matrix.multiply(y[:, None] != y[None, :]).data) / 2

        # Compute the product of class proportions
        class_proportions_product = np.prod(class_counts / total_samples)
        
        total_edges = adj_matrix.sum()/2
        
        # Avoid ZeroDivisionError
        if total_edges == 0:
            return class_proportions_product

        # Compute the topological impurity
        topological_impurity = class_proportions_product * (1 + edges_between_classes / total_edges)
        logger.debug(f"Topological Impurity = {class_proportions_product} * (1 + {edges_between_classes}/{total_edges})\n= {topological_impurity}")
        return topological_impurity


    def _find_best_split(self, X: np.ndarray, y: np.ndarray, adj_matrix: csr_matrix) -> tuple[Optional[int], Optional[float]]:
        best_impurity_reduction = float('-inf')
        best_split_feature: Optional[int] = None  # Add this annotation
        best_split_value: Optional[float] = None
        parent_impurity = self._topological_impurity(y, adj_matrix)

        for feature in range(X.shape[1]):
            unique_values = np.unique(X[:, feature])
            # trivial optim at least 2 unique_values required to split data
            if len(unique_values) < 2:
                logger.debug(f"Skipping | Feature {feature} has {unique_values = }")
                continue

            for split_value in unique_values:
                left_indices = X[:, feature] <= split_value
                right_indices = ~left_indices

                if (left_indices.sum() < self.min_samples_leaf or 
                    right_indices.sum() < self.min_samples_leaf):
                    logger.info(f"Skipping | {left_indices.sum() =} | {right_indices.sum() =}")
                    continue

                left_impurity = self._topological_impurity(y[left_indices], adj_matrix[left_indices][:, left_indices])
                right_impurity = self._topological_impurity(y[right_indices], adj_matrix[right_indices][:, right_indices])

                child_impurity = left_indices.sum() / len(y) * left_impurity + right_indices.sum() / len(y) * right_impurity
                impurity_reduction = parent_impurity - child_impurity # max this

                if (impurity_reduction > best_impurity_reduction and 
                    impurity_reduction > self.min_impurity_reduction): # at least > 0 impurity reduction
                    best_impurity_reduction = impurity_reduction
                    best_split_feature = feature
                    best_split_value = split_value

        logger.info(f"{best_impurity_reduction = }")
        return best_split_feature, best_split_value


    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.zeros(len(X), dtype=int)
        for i, sample in enumerate(X):
            predictions[i] = self._predict_sample(sample, self.tree_)
        return predictions


    def _predict_sample(self, sample: np.ndarray, node: Optional[dict[str, Any]]) -> int:
        if node is None:
            # Handle the case where node is None
            # This can be an error case, so raise an exception
            raise ValueError("Expected a non-None dictionary node")
        
        if node.get('leaf'):
            return node['class']

        if sample[node['split_feature']] <= node['split_value']:
            return self._predict_sample(sample, node['left'])
        else:
            return self._predict_sample(sample, node['right'])
