import os
import sys

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from loguru import logger
from molecularnetwork import MolecularNetwork
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from rdkit import RDLogger           
RDLogger.DisableLog('rdApp.*')                             

from topotree import TopologicalRandomForest


logger.remove()  # Remove any previous configurations
logger.add(sys.stdout, level="INFO")  # Add stdout with INFO level


def calculate_percent_activity_cliffs(adj_matrix, y):
    """
    Calculate the percentage of activity cliffs in the given adjacency matrix.

    Parameters:
    adj_matrix (np.ndarray): Adjacency matrix representing the molecular network.
    y (np.ndarray): Array of class labels for the nodes.

    Returns:
    float: Percentage of activity cliffs.
    """
    n_samples = len(y)
    assert adj_matrix.shape == (n_samples, n_samples), "Adjacency matrix shape does not match the number of samples."

    # Count edges between different classes
    different_class_edges = np.sum(adj_matrix * (y[:, None] != y[None, :])) / 2

    # Total number of edges in the adjacency matrix
    total_edges = adj_matrix.sum() / 2

    # Avoid division by zero
    if total_edges == 0:
        return 0.0

    percent_activity_cliffs = (different_class_edges / total_edges) * 100
    return percent_activity_cliffs


# Define a function to evaluate the models and store metrics
def evaluate_models(dataset_name, molnet_fp, 
                    model_fp, X_train, y_train, 
                    A_train, X_test, y_test, A_test):
    
    topo_clf = TopologicalRandomForest(max_depth=25, max_features=200, n_trees=100, random_state=69420)
    topo_clf.fit(X_train, y_train, A_train)

    # Train DecisionTreeClassifier
    clf = RandomForestClassifier(random_state=69420)
    clf.fit(X_train, y_train)

    percent_activity_cliff_train = calculate_percent_activity_cliffs(A_train, y_train)
    percent_activity_cliff_test = calculate_percent_activity_cliffs(A_test, y_test)

    # Make predictions
    topo_pred = topo_clf.predict(X_test)
    pred = clf.predict(X_test)

    # Calculate evaluation metrics
    metrics = {
        "Dataset": dataset_name,
        "Percent_Activity_Cliff_train":percent_activity_cliff_train,
        "Percent_Activity_Cliff_test":percent_activity_cliff_test,
        "Molecular_network_Fingeprint": molnet_fp,
        "Model_Fingeprint": model_fp,
        "Topological_Random_Forest_Accuracy": accuracy_score(y_test, topo_pred),
        "Topological_Random_Forest_Precision": precision_score(y_test, topo_pred),
        "Topological_Random_Forest_Recall": recall_score(y_test, topo_pred),
        "Topological_Random_Forest_Balanced_Accuracy": balanced_accuracy_score(y_test, topo_pred),
        "Topological_Random_Forest_F1_Score": f1_score(y_test, topo_pred),
        "Topological_Random_Forest_AUC_ROC": roc_auc_score(y_test, topo_pred),
        "Random_Forest_Accuracy": accuracy_score(y_test, pred),
        "Random_Forest_Precision": precision_score(y_test, pred),
        "Random_Forest_Recall": recall_score(y_test, pred),
        "Random_Forest_Balanced_Accuracy": balanced_accuracy_score(y_test, pred),
        "Random_Forest_F1_Score": f1_score(y_test, pred),
        "Random_Forest_AUC_ROC": roc_auc_score(y_test, pred)
    }

    return metrics

# Path to the folder containing datasets
data_folder = '../data/tdc_data/admet_group'

# Initialize an empty list to store metrics for all datasets
all_metrics = []

# Iterate over each dataset file in the data folder
file_name  = 'dili'
# Load dataset
train_df = pd.read_csv(os.path.join(data_folder, file_name, "train_val.csv"))
print(f"{train_df.shape = }")
test_df = pd.read_csv(os.path.join(data_folder, file_name, "test.csv"))
# Extract features and labels
train_smiles_list = train_df["Drug"].values
train_classes = train_df["Y"].values
test_smiles_list = test_df["Drug"].values
test_classes = test_df["Y"].values
for model_fp in ["rdkit", "maccs", "morgan2", "morgan3"]:
    for molnet_fp in ["rdkit", "maccs", "morgan2", "morgan3"]:
        logger.info(f'DATA PREP...{file_name}|{model_fp=}|{molnet_fp=}')
        network = MolecularNetwork(descriptor=molnet_fp, sim_metric="tanimoto", sim_threshold=0.4, node_descriptor=model_fp)
        graph = network.create_graph(train_smiles_list, train_classes)
        X_train = np.array([graph.nodes[i]['fp'] for i in graph.nodes])
        y_train = np.array([int(float(graph.nodes[i]['categorical_label'])) for i in graph.nodes])
        A_train = nx.adjacency_matrix(graph, weight='similarity').toarray()

        # test
        test_network = MolecularNetwork(descriptor=molnet_fp, sim_metric="tanimoto", sim_threshold=0.4, node_descriptor=model_fp)
        test_graph = test_network.create_graph(test_smiles_list, test_classes)
        X_test = np.array([test_graph.nodes[i]['fp'] for i in test_graph.nodes])
        y_test = np.array([int(float(test_graph.nodes[i]['categorical_label'])) for i in test_graph.nodes])
        A_test = nx.adjacency_matrix(test_graph, weight='similarity').toarray()
        logger.critical(f'DATA DONE...{file_name}|{model_fp=}|{molnet_fp=}')
        
        # Evaluate models and store metrics
        dataset_name = os.path.splitext(file_name)[0]

        # Evaluate models    
        logger.critical(f'STARTING EVAL...{file_name}|{model_fp=}|{molnet_fp=}')
        metrics = evaluate_models(dataset_name, molnet_fp, model_fp, X_train, y_train, A_train, X_test, y_test, A_test)
        logger.critical(f'EVAL DONE...{file_name}|{model_fp=}|{molnet_fp=}')
        # Append metrics to the list
        all_metrics.append(metrics)

        # Convert the list of dictionaries to a DataFrame
        metrics_df = pd.DataFrame(all_metrics)

        # Save the metrics to a CSV file
        metrics_df.to_csv('benchmark_dili_tdc_trf_result.csv', index=False)
