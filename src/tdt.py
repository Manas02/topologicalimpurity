import numpy as np
from loguru import logger


class TopologicalDecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_impurity_reduction=0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_reduction = min_impurity_reduction 
        self.tree_ = None

    
    def fit(self, X, y, adj_matrix):
        self.classes_ = np.unique(y)
        logger.debug(f'Unique classes : {self.classes_}')
        self.n_classes_ = len(self.classes_)
        assert self.n_classes_ >= 2, f"At least 2 classes should be present in categorical label (y), only {self.n_classes} found"
        logger.debug(f'Total number of unique classes : {self.n_classes_}')
        
        self.tree_ = self._build_tree(X, y, adj_matrix, depth=0)
        return self
    

    def _build_tree(self, X, y, adj_matrix, depth):
        if (depth == self.max_depth or 
            len(np.unique(y)) == 1 or 
            len(y) <= self.min_samples_split):
            logger.debug(f"Creating Leaf Node with y ({y.shape})")
            return self._create_leaf_node(y)

        best_split_feature, best_split_value = self._find_best_split(X, y, adj_matrix)
        if best_split_feature is None:
            logger.info(f"{best_split_feature = } for {y.shape = } & {adj_matrix.shape = }")
            return self._create_leaf_node(y)

        left_indices = X[:, best_split_feature] <= best_split_value
        right_indices = ~left_indices

        logger.debug(f"Building left tree with {left_indices.sum()} samples")
        left_tree = self._build_tree(X[left_indices], y[left_indices], adj_matrix[left_indices][:, left_indices], depth + 1)
        
        logger.debug(f"Building right tree with {right_indices.sum()} samples")
        right_tree = self._build_tree(X[right_indices], y[right_indices], adj_matrix[right_indices][:, right_indices], depth + 1)

        return {'split_feature': best_split_feature, 'split_value': best_split_value, 'left': left_tree, 'right': right_tree}
    

    def _create_leaf_node(self, y):
        assert y.any(), "y is empty, how to create a leaf node with no data ?"
        unique_classes, counts = np.unique(y, return_counts=True)
        majority_class = unique_classes[np.argmax(counts)]
        return {'leaf': True, 'class': majority_class}


    def _topological_impurity(self, y, adj_matrix):
        """Calculate Topological Impurity"""
        total_samples = len(y)
        class_counts = np.bincount(y)
        logger.debug(f"Topological Impurity = ∑ P(class_i) * (1 + |E_class_ij|/|E|)")
        # Count edges between different classes using vectorized operation
        edges_between_classes = np.sum(np.triu(np.logical_and(adj_matrix.toarray(), (y[:, None] != y[None, :]))))
        # Compute the product of class proportions
        class_proportions_product = np.prod(class_counts / total_samples)
        
        total_edges = np.sum(adj_matrix)
        logger.debug(f"Topological Impurity = {class_proportions_product} * (1 + {edges_between_classes}/{total_edges})")
        
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
