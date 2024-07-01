import numpy as np
from pqdm.threads import pqdm

from topotree import TopologicalDecisionTreeClassifier


class TopologicalRandomForest:
    def __init__(self, n_trees=100, n_jobs=6, max_depth=None, max_features=None,random_state=None):
        """
        Initialize the Random Forest.

        Args:
            tree_class: The custom decision tree class you want to use in the Random Forest.
            n_trees: Number of trees in the Random Forest.
            n_jobs: 6
            max_depth: Maximum depth of each tree in the Random Forest.
            random_state: Seed for random number generator.
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.random_state = random_state
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.trees = []

        # Set the random seed if provided
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
    def fit(self, X, y, adj_matrix):
        """
        Fit the Random Forest to the training data.

        Args:
            X: The training features.
            y: The training labels.
            adj_matrix: The adjacency matrix.
        """
        n_samples, n_features = X.shape

        # Define a function to train a single decision tree
        def train_tree(_):
            # Create a bootstrap sample
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            adj_matrix_bootstrap = adj_matrix[bootstrap_indices][:, bootstrap_indices]
            molnet_threshold = np.random.choice([0.4,0.5,0.6,0.7,0.8,0.9,0.95])

            # Randomly select a subset of features for the tree
            if self.max_features is not None and self.max_features < n_features:
                selected_features = np.random.choice(n_features, self.max_features, replace=False)
                X_bootstrap = X_bootstrap[:, selected_features]
            else:
                selected_features = None

            # Create and fit a decision tree on the bootstrap sample
            tree = TopologicalDecisionTreeClassifier(max_depth=self.max_depth, mol_net_threshold=molnet_threshold)
            tree.fit(X_bootstrap, y_bootstrap, adj_matrix_bootstrap)

            # Return the trained tree
            return tree, selected_features

        # Parallelize training each decision tree on a bootstrap sample using pqdm
        trained_trees = pqdm(range(self.n_trees), train_tree, n_jobs=self.n_jobs, desc="Training Trees", argument_type='index')

        # Append the trained trees and their selected features to the list of trees
        for tree, selected_features in trained_trees:
            self.trees.append((tree, selected_features))



    def predict(self, X):
        """
        Predict labels for new data using the Random Forest.

        Args:
            X: The test features.

        Returns:
            An array of predicted labels.
        """
        n_samples = X.shape[0]
        n_trees = len(self.trees)
        
        # Array to hold predictions from each tree
        tree_predictions = np.zeros((n_samples, n_trees), dtype=np.int64)

        # Predict with each tree and aggregate the results
        for i in range(n_trees):
            tree, selected_features = self.trees[i]
            if selected_features is not None:
                X_sub = X[:, selected_features]
            else:
                X_sub = X
            tree_predictions[:, i] = tree.predict(X_sub)
		
        # Aggregate predictions to find the majority class
        final_predictions = np.array([self.count_classes(pred) for pred in tree_predictions])
        return final_predictions
  

    def count_classes(self, predictions):
        """
        Count occurrences of each class in the predictions and find the majority class.

        Args:
            predictions: Array of predictions.

        Returns:
            Majority class.
        """
        # Use np.bincount to count occurrences of each class
        counts = np.bincount(predictions, minlength=256)
        
        # Find the index with the maximum count, which represents the majority class
        majority_class = np.argmax(counts)
        
        return majority_class
