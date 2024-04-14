import os
import sys

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from loguru import logger
from molecularnetwork import MolecularNetwork
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from topotree import TopologicalDecisionTreeClassifier


logger.remove()  # Remove any previous configurations
logger.add(sys.stdout, level="INFO")  # Add stdout with INFO level


# Define a function to evaluate the models and store metrics
def evaluate_models(dataset_name, sim_threshold, molnet_fp, 
                    model_fp, X_train, y_train, 
                    A_train, X_test, y_test):
    
    topo_clf = TopologicalDecisionTreeClassifier()
    topo_clf.fit(X_train, y_train, A_train)

    # Train DecisionTreeClassifier
    clf = DecisionTreeClassifier(random_state=69420)
    clf.fit(X_train, y_train)

    # Make predictions
    topo_pred = topo_clf.predict(X_test)
    pred = clf.predict(X_test)



    # Calculate evaluation metrics
    metrics = {
        "Dataset": dataset_name,
        "Sim_threshold": sim_threshold,
        "Molecular_network_Fingeprint": molnet_fp,
        "Model_Fingeprint": model_fp,
        "Topological_Decision_Tree_Accuracy": accuracy_score(y_test, topo_pred),
        "Decision_Tree_Accuracy": accuracy_score(y_test, pred),
        "Topological_Decision_Tree_Precision": precision_score(y_test, topo_pred),
        "Decision_Tree_Precision": precision_score(y_test, pred),
        "Topological_Decision_Tree_Recall": recall_score(y_test, topo_pred),
        "Decision_Tree_Recall": recall_score(y_test, pred),
        "Topological_Decision_Tree_Balanced_Accuracy": balanced_accuracy_score(y_test, topo_pred),
        "Decision_Tree_Balanced_Accuracy": balanced_accuracy_score(y_test, pred),
        "Topological_Decision_Tree_F1_Score": f1_score(y_test, topo_pred),
        "Decision_Tree_F1_Score": f1_score(y_test, pred),
        "Topological_Decision_Tree_AUC_ROC": roc_auc_score(y_test, topo_pred),
        "Decision_Tree_AUC_ROC": roc_auc_score(y_test, pred)
    }

    return metrics

def preprocess_data(df, sim_threshold, molnet_fp, model_fp):
    # train
    train_df = df[df["split"] == "train"]
    train_smiles_list = train_df["canonical_smiles"].values
    train_classes = train_df["active"].values
    network = MolecularNetwork(descriptor=molnet_fp, sim_metric="tanimoto", sim_threshold=sim_threshold, node_descriptor=model_fp)
    graph = network.create_graph(train_smiles_list, train_classes)
    X_train = np.array([graph.nodes[i]['fp'] for i in graph.nodes])
    logger.critical(X_train.shape)
    y_train = np.array([int(graph.nodes[i]['categorical_label']) for i in graph.nodes])
    A_train = nx.adjacency_matrix(graph, weight=None).toarray()

    # test
    test_df = df[df["split"] == "test"]
    test_smiles_list = test_df["canonical_smiles"].values
    test_classes = test_df["active"].values
    test_network = MolecularNetwork(descriptor=molnet_fp, sim_metric="tanimoto", sim_threshold=sim_threshold, node_descriptor=model_fp)
    test_graph = test_network.create_graph(test_smiles_list, test_classes)
    X_test = np.array([test_graph.nodes[i]['fp'] for i in test_graph.nodes])
    y_test = np.array([int(test_graph.nodes[i]['categorical_label']) for i in test_graph.nodes])
    
    return (X_train, y_train, A_train, X_test, y_test)


# Path to the folder containing datasets
data_folder = '../data/'

# Initialize an empty list to store metrics for all datasets
all_metrics = []

# Iterate over each dataset file in the data folder
for file_name in tqdm(os.listdir(data_folder)[3:], leave=False, desc="Dataset"):
    if file_name.endswith('.csv'):
        # Load dataset
        df = pd.read_csv(os.path.join(data_folder, file_name))
        # Extract features and labels
        for sim_threshold in tqdm([0.5, 0.7, 0.9, 0.95], leave=False, desc="Similarity Threshold"):
            for model_fp in tqdm(["maccs",  "morgan2"], leave=False, desc="Model Fingerprint"):
                for molnet_fp in tqdm(["maccs", "morgan2"], leave=False, desc="Molecularnet Fingerprint"):
                    logger.info(f'{file_name[:-4]}|{model_fp=}|{molnet_fp=}|{sim_threshold=}')
                    (X_train, y_train, A_train, X_test, y_test) = preprocess_data(df, sim_threshold, molnet_fp, model_fp)

                    # Evaluate models and store metrics
                    dataset_name = os.path.splitext(file_name)[0]

                    # Evaluate models    
                    metrics = evaluate_models(dataset_name, sim_threshold, molnet_fp, model_fp, X_train, y_train, A_train, X_test, y_test)

                    # Append metrics to the list
                    all_metrics.append(metrics)

                    # Convert the list of dictionaries to a DataFrame
                    metrics_df = pd.DataFrame(all_metrics)

                    # Save the metrics to a CSV file
                    metrics_df.to_csv('results_tree.csv', index=False)
