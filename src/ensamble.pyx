# cython: language_level=3, boundscheck=False, wraparound=False
# cython: cdivision=True
# cython: profile=False
# cython: auto_cpdef=True

import numpy as np
cimport numpy as np

from cython.parallel import prange
from collections import Counter

# Replace `CustomDecisionTree` with your custom decision tree class
from topotree import TopologicalDecisionTreeClassifier

# Define a function to count classes using a NumPy array
cdef int count_classes(np.ndarray[np.int64_t, ndim=1] predictions):
    """
    Count occurrences of each class in the predictions and find the majority class.

    Args:
        predictions: Array of predictions.

    Returns:
        Majority class.
    """
    cdef int i
    cdef int max_count = 0
    cdef int max_class = -1
    cdef np.ndarray[np.int64_t, ndim=1] counts = np.zeros(256, dtype=np.int64)

    for i in range(predictions.shape[0]):
        counts[predictions[i]] += 1

    for i in range(counts.shape[0]):
        if counts[i] > max_count:
            max_count = counts[i]
            max_class = i

    return max_class

# Define the RandomForest class
cdef class RandomForest:
    def __init__(self, tree_class=TopologicalDecisionTreeClassifier, int n_trees=100, int max_depth=None, int random_state=None):
        """
        Initialize the Random Forest.

        Args:
            tree_class: The custom decision tree class you want to use in the Random Forest.
            n_trees: Number of trees in the Random Forest.
            max_depth: Maximum depth of each tree in the Random Forest.
            random_state: Seed for random number generator.
        """
        self.tree_class = tree_class
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []

        if self.random_state is not None:
            np.random.seed(self.random_state)

    def fit(self, np.ndarray[np.float64_t, ndim=2] X, np.ndarray[np.int64_t, ndim=1] y, np.ndarray[np.int64_t, ndim=2] adj_matrix):
        """
        Fit the Random Forest to the training data.

        Args:
            X: The training features.
            y: The training labels.
            adj_matrix: The training adjecency matrix
        """
        cdef int n_samples = X.shape[0]

        # Parallelize the tree fitting process
        for i in prange(self.n_trees, nogil=True):
            # Create a bootstrap sample
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            adj_matrix_bootstrap = adj_matrix[bootstrap_indices][:, bootstrap_indices]

            # Create and fit a decision tree on the bootstrap sample
            tree = self.tree_class(max_depth=self.max_depth)
            tree.fit(X_bootstrap, y_bootstrap, adj_matrix_bootstrap)
            self.trees.append(tree)

    def predict(self, np.ndarray[np.float64_t, ndim=2] X):
        """
        Predict labels for new data using the Random Forest.

        Args:
            X: The test features.

        Returns:
            An array of predicted labels.
        """
        cdef int n_samples = X.shape[0]
        cdef int n_trees = len(self.trees)
        
        # Array to hold predictions from each tree
        tree_predictions = np.zeros((n_samples, n_trees), dtype=np.int64)

        # Parallelize predictions from each tree
        for i in prange(n_trees, nogil=True):
            tree_predictions[:, i] = self.trees[i].predict(X)

        # Array for final predictions
        final_predictions = np.zeros(n_samples, dtype=np.int64)

        # Parallelize the aggregation of predictions
        for j in prange(n_samples, nogil=True):
            final_predictions[j] = count_classes(tree_predictions[j])

        return final_predictions
