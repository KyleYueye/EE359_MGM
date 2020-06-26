import numpy as np


def mgm_online_cluster(X, K, num_graph, num_node, num_cluster):
    """
    :param K: affinity matrix, (num_graph, num_graph, num_node^2, num_node^2)
    :param num_graph: number of graph, int
    :param num_node: number of node, int
    :param num_cluster: number of cluster, int
    :return: an numpy array where cluster[i] tells the cluster g_i belong to, (num_graph, 1)
    """
    cluster_list = []
    for i in range(num_graph):
        cluster_list.append(np.ones((1, 1)) * i % num_cluster)
    cluster = np.concatenate(cluster_list, axis=0)
    return cluster, X
