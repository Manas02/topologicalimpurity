import os
import sys

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from loguru import logger
from molecularnetwork import MolecularNetwork
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from joblib import dump
from tqdm import trange
from topotree import TopologicalDecisionTreeClassifier
from sklearn.utils import resample


logger.remove()  # Remove any previous configurations
logger.add(sys.stdout, level="CRITICAL")  # Add stdout with INFO level


def preprocess_data(df, sim_threshold, molnet_fp, model_fp):
    # train
    train_df = df[df["split"] == "train"]
    train_smiles_list = train_df["canonical_smiles"].values
    train_classes = train_df["active"].values
    network = MolecularNetwork(descriptor=molnet_fp, sim_metric="tanimoto", sim_threshold=sim_threshold, node_descriptor=model_fp)
    graph = network.create_graph(train_smiles_list, train_classes)
    X_train = np.array([graph.nodes[i]['fp'] for i in graph.nodes])
    y_train = np.array([int(graph.nodes[i]['categorical_label']) for i in graph.nodes])
    A_train = nx.adjacency_matrix(graph, weight='similarity').toarray()

    # test
    test_df = df[df["split"] == "test"]
    test_smiles_list = test_df["canonical_smiles"].values
    test_classes = test_df["active"].values
    test_network = MolecularNetwork(descriptor=molnet_fp, sim_metric="tanimoto", sim_threshold=sim_threshold, node_descriptor=model_fp)
    test_graph = test_network.create_graph(test_smiles_list, test_classes)
    X_test = np.array([test_graph.nodes[i]['fp'] for i in test_graph.nodes])
    y_test = np.array([int(test_graph.nodes[i]['categorical_label']) for i in test_graph.nodes])
    
    return (X_train, y_train, A_train, X_test, y_test)


class TopologicalAdaBoost:
    def __init__(self, n_estimators=50, gamma=1.0, method='additive'):
        self.n_estimators = n_estimators
        self.gamma = gamma
        self.method = method
        self.alphas = []
        self.models = []
        self.train_errors = []  # Ensure train_errors is initialized

    def fit(self, X, y, A_train):
        n_samples, n_features = X.shape
        w = np.ones(n_samples) / n_samples  # Initialize weights uniformly
        
        for _ in trange(self.n_estimators):
            # Resample the training set according to the weights
            # X_resampled, y_resampled = resample(X, y)
            
            # Fit the decision stump on the resampled data
            stump = DecisionTreeClassifier(max_depth=3, class_weight='balanced')  # Add class weighting
            stump.fit(X, y, sample_weight=w)
            # stump.fit(X_resampled, y_resampled, sample_weight=w)
            
            pred = stump.predict(X)
            
            error = np.sum(w * (pred != y)) / np.sum(w)
            alpha = np.log((1 - error) / error) / 2
            
            self.models.append(stump)
            self.alphas.append(alpha)
            
            w = self.update_weights(w, y, pred, alpha, A_train)
            w /= np.sum(w)  # Normalize weights
            
            train_error = 1 - accuracy_score(y, self.predict(X))
            self.train_errors.append(train_error)
        
    def update_weights(self, w, y, pred, alpha, A_train):
        activity_cliffs = self.identify_activity_cliffs(y, A_train)
        
        if self.method == 'additive':
            return w * np.exp(alpha * (pred != y) + self.gamma * activity_cliffs * (pred != y))
        
        elif self.method == 'multiplicative':
            return w * np.exp(alpha * (pred != y)) * (1 + self.gamma * activity_cliffs * (pred != y))
        
        elif self.method == 'exponential':
            return w * np.exp(alpha * (pred != y) * (1 + self.gamma * activity_cliffs))
        
        else:
            raise ValueError("Invalid method. Choose from 'additive', 'multiplicative', or 'exponential'.")
    
    def identify_activity_cliffs(self, y, A_train):
        n_samples = len(y)
        activity_cliffs = np.zeros(n_samples)
        
        for i in range(n_samples):
            neighbors = np.where(A_train[i] > 0)[0]
            num_cliffs = sum(y[i] != y[neighbor] for neighbor in neighbors)
            activity_cliffs[i] = num_cliffs
        return activity_cliffs / max(activity_cliffs)
    
    def predict(self, X):
        model_preds = np.array([alpha * model.predict(X) for model, alpha in zip(self.models, self.alphas)])
        pred = (np.sum(model_preds, axis=0) >= 0.5).astype(int)
        return pred

if __name__ == '__main__':
    # Load dataset
    file_name = "../data/simpd/CHEMBL1267245.csv"
    df = pd.read_csv(file_name)
    # Extract features and labels
    sim_threshold = 0.7
    molnet_fp, model_fp = "morgan2", "morgan2"
    (X_train, y_train, A_train, X_test, y_test) = preprocess_data(df, sim_threshold, molnet_fp, model_fp)
    logger.info(f'{file_name[:-4]}|{model_fp=}|{molnet_fp=}|{sim_threshold=}')

    
    methods = ['additive', 'multiplicative', 'exponential']
    test_accuracies = {}
    train_errors_dict = {}  # Dictionary to store training errors for each method
    
    for method in methods:
        model = TopologicalAdaBoost(n_estimators=50, gamma=1.0, method=method)
        model.fit(X_train, y_train, A_train)
        predictions = model.predict(X_test)
        
        accuracy = roc_auc_score(y_test, predictions)
        test_accuracies[method] = accuracy
        train_errors_dict[method] = model.train_errors  # Store training errors
        print(f"Accuracy with {method} method: {accuracy:.2f}")

    # Plot test accuracy
    plt.figure(figsize=(10, 6))
    plt.bar(test_accuracies.keys(), test_accuracies.values())
    plt.title('Test Accuracy for Different Methods')
    plt.xlabel('Method')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid()
    plt.show()