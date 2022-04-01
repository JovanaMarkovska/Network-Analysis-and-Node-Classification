import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
from gem.evaluation import evaluate_graph_reconstruction as gr
from gem.evaluation import visualize_embedding as viz
from sklearn.metrics import accuracy_score, f1_score
from gem.embedding.sdne import SDNE
import numpy as np
from graph_embeddings import save_embeddings
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def calculate_metrics(test_targets, predictions):
    """Calculation of accuracy score, F1 micro and F1 macro"""
    print(f'\tAccuracy score: {accuracy_score(test_targets, predictions)}')
    print(f'\tF1-micro: {f1_score(test_targets, predictions, average="micro")}')
    print(f'\tF1-macro: {f1_score(test_targets, predictions, average="macro")}')

def save_embeddings(file_path, embs, nodes):
    """Save node embeddings
    :param file_path: path to the output file
    :type file_path: str
    :param embs: matrix containing the embedding vectors
    :type embs: numpy.array
    :param nodes: list of node names
    :type nodes: list(int)
    :return: None
    """
    with open(file_path, 'w') as f:
        f.write(f'{embs.shape[0]} {embs.shape[1]}\n')
        for node, emb in zip(nodes, embs):
            f.write(f'{node} {" ".join(map(str, emb.tolist()))}\n')


def read_embeddings(file_path):
    """ Load node embeddings
    :param file_path: path to the embedding file
    :type file_path: str
    :return: dictionary containing the node names as keys
    and the embeddings vectors as values
    :rtype: dict(int, numpy.array)
    """
    with open(file_path, 'r') as f:
        f.readline()
        embs = {}
        line = f.readline().strip()
        while line != '':
            parts = line.split()
            embs[int(parts[0])] = np.array(list(map(float, parts[1:])))
            line = f.readline().strip()
    return embs

def read_graph():
    # Load the graph from edgelist
    edgelist = pd.read_table('../cora/cora.cites',
                             header=None, names=['source', 'target'])
    edgelist['label'] = 'cites'
    graph = nx.from_pandas_edgelist(edgelist, edge_attr='label')
    nx.set_node_attributes(graph, 'paper', 'label')

    # Load the features and subject for the nodes
    feature_names = ['w_{}'.format(ii) for ii in range(1433)]
    column_names = feature_names + ['subject']
    node_data = pd.read_table('../cora/cora.content',
                              header=None, names=column_names)

    return graph, node_data, feature_names


if __name__ == '__main__':
    graph, node_features, feature_names = read_graph()
    nodes = graph.nodes()

    # SDNE
    sdne = SDNE(d=50, beta=5, alpha=1, nu1=0.000001, nu2=0.000001, K=2,
                 n_units=[100, 50], n_iter=50, xeta=0.01, n_batch=500)
    embeddings_sdne, t = sdne.learn_embedding(graph=graph, edge_f=None, is_weighted=False, no_python=True)
    save_embeddings(file_path='../data/cora_sdne_embeddings.emb', embs=embeddings_sdne, nodes=nodes)


    embeddings_df = pd.DataFrame(embeddings_sdne, index=nodes)

    subject_df = pd.DataFrame(node_features['subject'])

    encoder = OrdinalEncoder()
    encodings = encoder.fit_transform(subject_df)
    subject_df = pd.DataFrame(encodings, columns=['subject'], index=nodes)

    train_x, test_x, train_y, test_y = train_test_split(embeddings_df, subject_df, test_size=0.1, stratify=subject_df)

    classifier = RandomForestClassifier(n_estimators=100)
    classifier.fit(train_x, train_y.values.ravel())

    pred_y = classifier.predict(test_x)
    calculate_metrics(test_y,pred_y)




