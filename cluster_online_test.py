import time
import os
import torch
import pickle
import numpy as np
from statistics import mean
from src.mgm_online_cluster import mgm_online_cluster
from src.mgm_floyd import mgm_floyd
from src.rrwm import RRWM
from utils.cluster_data_prepare import DataGenerator
from utils.evaluation import evaluate_cluster_accuracy
from utils.hungarian import hungarian
from utils.cfg import cfg as CFG

# set dataset and class for offline multi-graph matching test
dataset_name = "WILLOW-ObjectClass"
# you can remove some classes type during the debug process
class_name = ["Car", "Motorbike", "Face", "Winebottle", "Duck"]

# set parameters for offline multi-graph matching test
test_iter = 1  # test iteration for each class, please set it less than 5 or new data will be generated

# number of graphs, inliers and outliers only affect the generated data (when test_iter is larger than 5),
# these parameters will not be used when the data is loaded from TestPrepare.
test_num_graph = 24  # number of graphs in each test
test_num_cluster = 2  # number of clusters
test_num_graph_cluster = 12  # number of graphs in each dataset(cluster)
test_num_inlier = 10  # number of inliers in each graph
test_num_outlier = 2  # number of outliers in each graph

assert test_num_graph == test_num_cluster * test_num_graph_cluster

rrwm = RRWM()
cfg = CFG()

print("Test begin: test online multi-graph matching on {}".format(dataset_name))
time_list, cp_list, ri_list, ca_list = [[] for _ in range(8)], [[] for _ in range(8)], \
                                       [[] for _ in range(8)], [[] for _ in range(8)]
for i_iter in range(test_iter):
    # prepare affinity matrix data for graph matching

    # set the path for loading data
    test_data_folder = "cluster_data_" + str(test_num_cluster) + "*" + str(test_num_graph_cluster)
    # test_data_folder = "cluster_data_{}*{}".format(test_num_cluster, test_num_graph_cluster)
    test_data_folder_path = "data" + "/" + "TestPrepare" + "/" + test_data_folder
    print(test_data_folder_path)
    if not os.path.exists(test_data_folder_path):
        os.mkdir(test_data_folder_path)
    test_data_path = "data" + "/" + "TestPrepare" + "/" + test_data_folder + "/" + "test_data_" + str(i_iter)
    if os.path.exists(test_data_path):
        # load data from "/TestPrepare/{test_data_folder_path}/test_data_{i_iter}"
        with open(test_data_path, "rb") as f:
            data = pickle.load(f)
    else:
        # if nothing can be loaded, generate new data and save it
        data = DataGenerator(
            data_path=None,
            num_graphs=test_num_graph,
            num_inlier=test_num_inlier,
            num_outlier=test_num_outlier,
            num_cluster=test_num_cluster,
            num_graphs_cluster=test_num_graph_cluster,
        )
        for i_class in range(test_num_cluster):
            class_path = "data" + "/" + dataset_name + "/" + class_name[i_class]
            data.add_data(
                data_path=class_path,
                num_inlier=test_num_inlier,
                num_outlier=test_num_outlier,
                num_graphs=test_num_graph_cluster
            )
        data.preprocess()
        with open(test_data_path, "wb") as f:
            pickle.dump(data, f)

    # pairwise matching: RRWM

    # set the path for loading pairwise matching results
    init_mat_path = "data" + "/" + "TestPrepare" + "/" + test_data_folder + "/" + "init_mat_" + str(i_iter)
    if os.path.exists(init_mat_path):
        # load pairwise matching results from "/TestPrepare/{ClassType}/init_mat_{i_iter}"
        with open(init_mat_path, "rb") as f:
            X = pickle.load(f)
    else:
        # if nothing can be loaded, generate the initial matching results and save them
        m, n = data.num_graphs, data.num_nodes
        Kt = torch.tensor(data.K).reshape(-1, n * n, n * n).cuda()
        ns_src = torch.ones(m * m).int().cuda() * n
        ns_tgt = torch.ones(m * m).int().cuda() * n
        X_continue = rrwm(Kt, n, ns_src, ns_tgt).reshape(m * m, n, n).transpose(1, 2).contiguous()
        X = hungarian(X_continue, ns_src, ns_tgt).reshape(m, m, n, n).cpu().detach().numpy()
        with open(init_mat_path, "wb") as f:
            pickle.dump(X, f)

    init_mat = X
    X = mgm_floyd(init_mat[:15, :15], data.K[:15, :15], 15, data.num_nodes)

    for n_graph in range(16, 24):
        tic = time.time()
        X_new = init_mat[:n_graph, :n_graph, :, :]
        X_new[:-1, :-1, :, :] = X
        cluster, X = mgm_online_cluster(X_new, data.K[:n_graph, :n_graph, :, :], n_graph, data.num_nodes,
                                        data.num_cluster)
        cp, ri, ca = evaluate_cluster_accuracy(cluster, data.gt[:n_graph], data.num_cluster, n_graph)
        toc = time.time()

        time_list[n_graph - 16].append(toc - tic)
        cp_list[n_graph - 16].append(cp)
        ri_list[n_graph - 16].append(ri)
        ca_list[n_graph - 16].append(ca)

        print("Performance on iter{} ngraph{}".format(i_iter, n_graph))
        print("Clustering Purity: {:.4f}, Rand Index: {:.4f}, Clustering Accuracy: {:.4f}, time: {:.4f}".
              format(cp, ri, ca, toc - tic))

print("")
print("Overall Performance")
for i in range(8):
    avg_time = mean(time_list[i])
    avg_cp = mean(cp_list[i])
    avg_ri = mean(ri_list[i])
    avg_ca = mean(ca_list[i])
    print(
        "Performance num graph {}, Clustering Purity: {:.4f}, Rand Index: {:.4f}, Clustering Accuracy: {:.4f}, time: {:.4f}".
        format(i + 16, avg_cp, avg_ri, avg_ca, avg_time))
