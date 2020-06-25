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
    # m, _, n, _ = X.shape
    # store_x = [i[0] for i in store]
    # store_y = [i[1] for i in store]
    # for i in range(m):
    #     for j in range(m):
    #         cnt = 0.0
    #         X_ij = X[i, j]
    #         if i in store_x or j in store_y:
    #             pass
    #         else:
    #             continue
    #         for k in range(m):
    #             # X_ikj = X[i, k] * X[k, j]
    #             if ((i, k) not in store) and ((k, j) not in store):
    #                 continue
    #             cnt += np.sum(np.abs(X_ij - X[i, k] * X[k, j]))
    #         con_pair[i, j] = 1 - cnt / (2 * m * n)
    # return con_pair
    m, _, n, _ = X.shape
    
    X_i = X.reshape(m, 1, m, n, n)
    X_j = X.transpose(1, 0, 2, 3).reshape(1, m, m, n, n)
    X_k = np.expand_dims(X, 2).repeat(m, axis=2)
    
    X_ikj = np.matmul(X_i, X_j)
    X_sum = np.abs(X_k - X_ikj)
    res = 1-np.sum(X_sum, axis=(2,3,4)) / (2 * m * n)
    
    return res


def part_consistency(con_pair, X, store):
    """
    compute part of consistency pair to improve performance.
    :param con_pair: consistency pair (m, m)
    :param X: matching result permutation matrix (m, m, n, n)
    :param store: list of pairs which are modified in X
    :return: consistency pair (m, m)
    """
    m, _, n, _ = X.shape
    store_x = [i[0] for i in store]
    store_y = [i[1] for i in store]
    for i in range(m):
        for j in range(m):
            cnt = 0.0
            X_ij = X[i, j]
            if i in store_x or j in store_y:
                pass
            else:
                continue
            for k in range(m):
                # X_ikj = X[i, k] * X[k, j]
                if ((i, k) not in store) and ((k, j) not in store):
                    continue
                cnt += np.sum(np.abs(X_ij - X[i, k] * X[k, j]))
            con_pair[i, j] = 1 - cnt / (2 * m * n)
    return con_pair


def mgm_spfa(K, X, num_graph, num_node):
    """
    :param K: affinity matrix, (num_graph, num_graph, num_node^2, num_node^2)
    :param X: matching results, X[:-1, :-1] is the matching results obtained by last iteration of MGM-SPFA,
              X[num_graph,:] and X[:,num_graph] is obtained via two-graph matching solver(RRWM), We suppose the last
              graph is the new coming graph. (num_graph, num_graph, num_node, num_node)
    :param num_graph: number of graph, int
    :param num_node: number of node, int
    :return: X, matching results, match graph_m to {graph_1, ... , graph_m-1)
    """
    # afnty = np.zeros((num_graph, num_graph))
    # for i in range(num_graph):
    #     for j in range(num_graph):
    #         afnty[i, j] = afnty_scr(X[i, j], K[i, j], 1.0)
    
    X = X.transpose(0, 1, 3, 2)
    pro_X = X.reshape(num_graph, num_graph, -1, 1)
    pro_X_t = X.reshape(num_graph, num_graph, 1, -1)
    afnty = np.matmul(np.matmul(pro_X_t, K), pro_X).reshape(num_graph, num_graph)
    max_afnty = np.max(afnty)

    queue = [x for x in range(num_graph - 1)]
    consistency = consistency_pair(X)
    cnt = 0
    c = 0.9
    while len(queue) != 0:
        store = []
        cnt += 1
        G_x = queue[0]
        queue.remove(G_x)
        for i in range(num_graph - 1):
            if i == G_x:
                continue
            X_org = X[num_graph - 1, i]
            X_opt = np.matmul(X[num_graph - 1, G_x], X[G_x, i])
            K_org = K[num_graph - 1, i]
            S_org = c * np.sqrt(consistency[num_graph - 1, i]) + (1 - c) * afnty_scr(X_org, K_org, max_afnty)
            S_opt = c * np.sqrt(consistency[num_graph - 1, G_x] * consistency[G_x, i]) + (1 - c) * afnty_scr(X_opt,
                                                                                                             K_org,
                                                                                                             max_afnty)
            if S_org < S_opt:
                X[num_graph - 1, i] = X_opt
                store.append((num_graph - 1, i))
                X[i, num_graph - 1] = X_opt.transpose()
                store.append((i, num_graph - 1))
        if cnt % 2 == 0:
            # consistency = consistency_pair(X)
            consistency = part_consistency(consistency, X, store)
        if cnt > num_graph * num_graph:
            queue = []

    consistency = consistency_pair(X)

    for i in range(num_graph):
        for j in range(num_graph):
            X_org = X[i, j]
            X_opt = np.matmul(X[i, num_graph - 1], X[num_graph - 1, j])
            S_org = c * np.sqrt(consistency[i, j]) + (1 - c) * afnty_scr(X_org, K[i, j], max_afnty)
            S_opt = c * np.sqrt(consistency[i, num_graph - 1] * consistency[num_graph - 1, j]) + (1 - c) * afnty_scr(
                X_opt, K[i, j], max_afnty)
            if S_org < S_opt:
                X[i, j] = X_opt
    return X
