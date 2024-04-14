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


# Define a function to evaluate the models and store metrics
def evaluate_models(dataset_name, molnet_fp, 
                    model_fp, X_train, y_train, 
                    A_train, X_test, y_test):
    
    topo_clf = TopologicalRandomForest(max_depth=25, n_trees=100, random_state=69420)
    topo_clf.fit(X_train, y_train, A_train)

    # Train DecisionTreeClassifier
    clf = RandomForestClassifier(random_state=69420)
    clf.fit(X_train, y_train)

    # Make predictions
    topo_pred = topo_clf.predict(X_test)
    pred = clf.predict(X_test)

    # Calculate evaluation metrics
    metrics = {
        "Dataset": dataset_name,
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
for model_fp in ["maccs",  "morgan2"]:
    for molnet_fp in ["maccs", "morgan2"]:
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
        logger.critical(f'DATA DONE...{file_name}|{model_fp=}|{molnet_fp=}')
        
        # Evaluate models and store metrics
        dataset_name = os.path.splitext(file_name)[0]

        # Evaluate models    
        logger.critical(f'STARTING EVAL...{file_name}|{model_fp=}|{molnet_fp=}')
        metrics = evaluate_models(dataset_name, molnet_fp, model_fp, X_train, y_train, A_train, X_test, y_test)
        logger.critical(f'EVAL DONE...{file_name}|{model_fp=}|{molnet_fp=}')
        # Append metrics to the list
        all_metrics.append(metrics)

        # Convert the list of dictionaries to a DataFrame
        metrics_df = pd.DataFrame(all_metrics)

        # Save the metrics to a CSV file
        metrics_df.to_csv('tdc_adme_results_tree_one_rf.csv', index=False)
