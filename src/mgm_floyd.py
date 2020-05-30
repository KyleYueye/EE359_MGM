import numpy as np


def afnty_scr(X, K, max_afnty):
    """
    compute the affinity score
    :param X: (n, n)
    :param K: (n*n, n*n)
    :return: affinity_score (b, 1, 1)
    """
    n, _ = X.shape
    vecx = X.transpose().reshape(-1, 1)
    vecxT = vecx.transpose()
    affinity_score = np.matmul(np.matmul(vecxT, K), vecx)
    return affinity_score[0][0] / max_afnty


def consistency_pair(X):
    """
    compute consistency pair
    :param X: matching result permutation matrix (m, m, n, n)
    :return: consistency pair (m, m)
    """
    m, _, n, _ = X.shape
    con_pair = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            cnt = 0.0
            X_ij = X[i, j]
            for k in range(m):
                # X_ikj = X[i, k] * X[k, j]
                cnt += np.sum(np.abs(X_ij - X[i, k] * X[k, j]))
            con_pair[i, j] = 1 - cnt / (2 * m * n)
    return con_pair


def mgm_floyd(X, K, num_graph, num_node):
    """
    :param K: affinity matrix, (num_graph, num_graph, num_node^2, num_node^2)
    :param num_graph: number of graph, int
    :param num_node: number of node, int
    :return: matching results, (num_graph, num_graph, num_node, num_node)
    """
    afnty = np.zeros((num_graph, num_graph))
    for i in range(num_graph):
        for j in range(num_graph):
            afnty[i, j] = afnty_scr(X[i, j], K[i, j], 1.0)
    max_afnty = np.max(afnty)

    c = 0.8
    for k in range(num_graph):
        consistency = consistency_pair(X)
        for i in range(num_graph):
            for j in range(num_graph):
                Xo = X[i, j]
                Xu = np.matmul(X[i, k], X[k, j])
                so = c * np.sqrt(consistency[i, j]) + (1 - c) * afnty_scr(Xo, K[i, j], max_afnty)
                su = c * np.sqrt(consistency[i, k] * consistency[k, j]) + (1 - c) * afnty_scr(Xu, K[i, j], max_afnty)
                if su > so:
                    X[i, j] = Xu
    return X
