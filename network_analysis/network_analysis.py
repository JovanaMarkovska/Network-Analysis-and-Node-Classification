from collections import Counter
import networkx as nx
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from networkx import diameter

def read_graph():
    # Load the graph from edgelist
    edgelist = pd.read_table('../cora.cites',
                             header=None, names=['source', 'target'])
    edgelist['label'] = 'cites'
    graph = nx.from_pandas_edgelist(edgelist, edge_attr='label')
    nx.set_node_attributes(graph, 'paper', 'label')

    # Load the features and subject for the nodes
    feature_names = ['w_{}'.format(ii) for ii in range(1433)]#1433 unique words(features)
    column_names = feature_names + ['subject']
    node_data = pd.read_table('../cora.content',
                              header=None, names=column_names)

    return graph, node_data, feature_names

def plot_degree_distribution(node_degrees):
    node_degrees = [d[1] for d in node_degrees]
    counts = Counter(node_degrees)
    degrees = list(counts.keys())
    values = list(counts.values())
    sns.barplot(degrees, values)
    plt.show()


def plot_graph(g):
    plt.figure(num=None, figsize=(20, 20), dpi=80)
    plt.axis('off')
    fig = plt.figure(1)

    pos = nx.spring_layout(g)
    nx.draw_networkx_nodes(g, pos, node_size=10)
    nx.draw_networkx_edges(g, pos)

    plt.show()


# Finding the largest connected component (maximal number of nodes)
def max_nodes_component(component_map):
    max_value=0
    max_key=0
    for key in component_map.keys():
        if(max_value<component_map[key]):
            max_value = component_map[key]
            max_key = key

    print(f'Largest component is {max_key} with {max_value} nodes.')

# Mapping the connected components with their number of nodes
def nodes_in_component(components):
    counter = 0
    component_map = {}
    for component in components:
        counter = counter + 1
        print(f'Number of nodes in the connected component {counter}: {len(component)}')
        component_map[counter] = len(component)

    return component_map


if __name__ == '__main__':
    # CORA graph
    graph, node_features, feature_names = read_graph()
    plot_graph(graph)

    cora_nodes = graph.nodes()

    num_cora_nodes = graph.number_of_nodes()
    print(f'Number of nodes in cora dataset {num_cora_nodes}')

    cora_edges = graph.edges()

    num_cora_edges = graph.number_of_edges()
    print(f'Number of edges in cora dataset {num_cora_edges}')

    degree_cora = graph.degree(cora_nodes)
    print(f'Node degrees: {degree_cora}')

    plot_degree_distribution(degree_cora)

    print(f'Number of connected components: {nx.number_connected_components(graph)}')

    components = nx.connected_components(graph)

    # Counting the nodes in each connected component
    component_map = nodes_in_component(components)

    # Finding the largest connected component
    max_nodes_component(component_map)

    # Diameter of the graph - You cannot compute diameter for either 1) a weakly-connected directed graph or
    # 2) a disconnected graph, but you might use the maximal shortest path like so:
    diameter_cora = max([max(j.values()) for (i, j) in nx.shortest_path_length(graph)])
    print(f'The diameter of the cora graph is: {diameter_cora}')

    print(f'Average clustering coefficient of cora: {nx.average_clustering(graph)}')

    # ERDOS RENYI MODEL (Random Graph)
    graph_er = nx.erdos_renyi_graph(num_cora_nodes, 0.2)

    plot_graph(graph_er)

    plot_degree_distribution(graph_er.degree(graph_er.nodes()))

    print(f'Number of connected components of Erdos Renyi graph: {nx.number_connected_components(graph_er)}')

    print(f'The diameter of the Erdos Renyi graph is: {diameter(graph_er)}')

    print(f'Average clustering coefficient of Erdos Renyi graph: {nx.average_clustering(graph_er)}')



    # WATTS STROGATZ MODEL (Small World)
    graph_sm = nx.watts_strogatz_graph(num_cora_nodes, 3, 0.2)

    plot_graph(graph_sm)

    plot_degree_distribution(graph_sm.degree(graph_sm.nodes()))

    print(f'Number of connected components of Watts Strogatz graph: {nx.number_connected_components(graph_sm)}')

    # Diameter of the graph - You cannot compute diameter for either
    # 1) a weakly-connected directed graph or
    # 2) a disconnected graph, but you might use the maximal shortest path like so:
    diameter_sm = max([max(j.values()) for (i, j) in nx.shortest_path_length(graph_sm)])
    print(f'The diameter of the Watts Strogatz graph is: {diameter_sm}')

    print(f'Average clustering coefficient of Watts Strogatz graph: {nx.average_clustering(graph_sm)}')





