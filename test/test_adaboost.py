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

from topotree import TopologicalDecisionTreeClassifier


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



""" HELPER FUNCTION: GET ERROR RATE ========================================="""
def get_error_rate(pred, Y):
    score = balanced_accuracy_score(Y, pred)
    return 1 - score

""" HELPER FUNCTION: PRINT ERROR RATE ======================================="""
def print_error_rate(err):
    print('Error rate: Training: %.4f - Test: %.4f' % err)

""" HELPER FUNCTION: GENERIC CLASSIFIER ====================================="""
def generic_clf(y_train, X_train, y_test, X_test, A_train, clf):
    if isinstance(clf, TopologicalDecisionTreeClassifier):
        clf.fit(X_train,y_train,A_train)
    else:
        clf.fit(X_train,y_train)
    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)
    return get_error_rate(pred_train, y_train), \
           get_error_rate(pred_test, y_test)
    
""" ADABOOST IMPLEMENTATION ================================================="""
def adaboost_clf(y_train, X_train, y_test, X_test, M, clf):
    n_train, n_test = len(X_train), len(X_test)
    # Initialize weights
    w = np.ones(n_train) / n_train
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]
    
    for i in range(M):
        # Fit a classifier with the specific weights
        clf.fit(X_train, y_train, sample_weight = w)
        pred_train_i = clf.predict(X_train)
        pred_test_i = clf.predict(X_test)
        # Indicator function
        miss = [int(x) for x in (pred_train_i != y_train)]
        # Equivalent with 1/-1 to update weights
        miss2 = [x if x==1 else -1 for x in miss]
        # Error
        err_m = np.dot(w,miss) / sum(w)
        # Alpha
        alpha_m = 0.5 * np.log( (1 - err_m) / float(err_m))
        # New weights
        w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))
        # Add to prediction
        pred_train = [sum(x) for x in zip(pred_train, 
                                          [x * alpha_m for x in pred_train_i])]
        pred_test = [sum(x) for x in zip(pred_test, 
                                         [x * alpha_m for x in pred_test_i])]
    
    pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)
    # Return error rate in train and test set
    return get_error_rate(pred_train, y_train), \
           get_error_rate(pred_test, y_test)

""" PLOT FUNCTION ==========================================================="""
def plot_error_rate(er_train, er_test, title):
    df_error = pd.DataFrame([er_train, er_test]).T
    df_error.columns = ['Training', 'Test']
    plot1 = df_error.plot(linewidth = 3, figsize = (8,6),
            color = ['lightblue', 'darkblue'], grid = True)
    plot1.set_xlabel('Number of iterations', fontsize = 12)
    plot1.set_xticklabels(range(0,450,50))
    plot1.set_ylabel('Error rate', fontsize = 12)
    plot1.set_title(f'{title} | Error rate vs number of iterations', fontsize = 16)
    plt.axhline(y=er_test[0], linewidth=1, color = 'red', ls = 'dashed')


if __name__ == '__main__':
    # Load dataset
    file_name = "../data/simpd/CHEMBL3888295.csv"
    df = pd.read_csv(file_name)
    # Extract features and labels
    sim_threshold = 0.8
    molnet_fp, model_fp = "morgan2", "morgan2"
    (X_train, y_train, A_train, X_test, y_test) = preprocess_data(df, sim_threshold, molnet_fp, model_fp)
    logger.info(f'{file_name[:-4]}|{model_fp=}|{molnet_fp=}|{sim_threshold=}')

    # Fit a simple decision tree first
    clf_tree = DecisionTreeClassifier(max_depth = 1, random_state = 1)
    er_dt_tree = generic_clf(y_train, X_train, y_test, X_test, None, clf_tree)
    topo_clf = TopologicalDecisionTreeClassifier(max_depth=1, mol_net_threshold=sim_threshold)
    er_tdt_tree = generic_clf(y_train, X_train, y_test, X_test, A_train, topo_clf)
    
    # Fit Adaboost classifier using a decision tree as base estimator
    # Test with different number of iterations
    er_dt_train, er_dt_test = [er_dt_tree[0]], [er_dt_tree[1]]
    er_tdt_train, er_tdt_test = [er_tdt_tree[0]], [er_tdt_tree[1]]

    for i in range(10, 210, 10):    
        er_dt = adaboost_clf(y_train, X_train, y_test, X_test, i, clf_tree)
        er_dt_train.append(er_dt[0])
        er_dt_test.append(er_dt[1])

        er_tdt = adaboost_clf(y_train, X_train, y_test, X_test, i, clf_tree)
        er_tdt_train.append(er_tdt[0])
        er_tdt_test.append(er_tdt[1])

    # Compare error rate vs number of iterations
    plot_error_rate(er_dt_train, er_dt_test, "decision tree")
    plt.savefig('adaboost_dt.pdf')
    plt.close()
    plot_error_rate(er_tdt_train, er_tdt_test, "topological decision tree")
    plt.savefig('adaboost_tdt.pdf')
    plt.close()

