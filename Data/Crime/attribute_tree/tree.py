import os
import sys
import copy
import torch
import time
import pickle
import numpy as np
import networkx as nx
from codingTree import PartitionTree
from torch_geometric.utils import to_undirected
from tqdm import tqdm
from crime_max1SE_new import get_weight, knn_maxE1, get_adj_matrix, add_knn
from joblib import Parallel, delayed

device = "cuda:0"
PWD = os.path.dirname(os.path.realpath(__file__))

knn_edge_index = []


def trans_to_adj(graph):
    graph.remove_edges_from(nx.selfloop_edges(graph))
    nodes = range(len(graph.nodes))
    return nx.to_numpy_array(graph, nodelist=nodes)


def trans_to_tree(adj, k=2):
    undirected_adj = np.array(adj)
    y = PartitionTree(adj_matrix=undirected_adj)
    x = y.build_coding_tree(k)
    return y.tree_node


def update_depth(tree):
    wait_update = [k for k, v in tree.items() if v.children is None]
    while wait_update:
        for nid in wait_update:
            node = tree[nid]
            if node.children is None:
                node.child_h = 0
            else:
                node.child_h = tree[list(node.children)[0]].child_h + 1
        wait_update = set([tree[nid].parent for nid in wait_update if tree[nid].parent])


def update_node(tree):
    update_depth(tree)
    d_id = [(v.child_h, v.ID) for k, v in tree.items()]
    d_id.sort()
    new_tree = {}
    for k, v in tree.items():
        n = copy.deepcopy(v)
        n.ID = d_id.index((n.child_h, n.ID))
        if n.parent is not None:
            n.parent = d_id.index((n.child_h + 1, n.parent))
        if n.children is not None:
            n.children = [d_id.index((n.child_h - 1, c)) for c in n.children]
        n = n.__dict__
        n["depth"] = n["child_h"]
        new_tree[n["ID"]] = n
    return new_tree


def trans_graph_tree(G, tree_depth):
    adj_mat = trans_to_adj(G)
    tree = trans_to_tree(adj_mat, tree_depth)
    return update_node(tree)


def get_layer_graph(tree, graph, tree_depth):
    layer_graph = [graph]
    for l in range(1, tree_depth):
        partition = {
            frozenset([tree[c].get("graphID", c) for c in n["children"]]): i
            for i, n in tree.items()
            if n["depth"] == l
        }
        lg = nx.quotient_graph(layer_graph[-1], partition.keys(), relabel=False)
        lg = nx.relabel_nodes(lg, partition)
        layer_graph.append(lg)
    return layer_graph


def extract_layer_data(T, G, tree_depth):
    node_size = [0] * (tree_depth + 1)
    layer_idx = [0]
    for layer in range(tree_depth + 1):
        layer_nodes = [i for i, n in T.items() if n["depth"] == layer]
        layer_idx.append(layer_idx[-1] + len(layer_nodes))
        node_size[layer] = len(layer_nodes)

    interLayerEdges = [[] for i in range(tree_depth + 1)]
    for i, n in T.items():
        if n["depth"] == 0:
            continue
        n_idx = n["ID"] - layer_idx[n["depth"]]
        c_base = layer_idx[n["depth"] - 1]
        interLayerEdges[n["depth"]].extend([(n_idx, c - c_base) for c in n["children"]])
    interLayer_edgeMat = [torch.LongTensor(es).T for es in interLayerEdges]

    layer_graphs = get_layer_graph(T, G, tree_depth)
    layer_edgeMat = []
    for l in range(tree_depth):
        g = layer_graphs[l]
        nmap = {n: n - layer_idx[l] for n in g.nodes}
        g = nx.relabel_nodes(g, nmap)
        edges = [[n1, n2] for n1, n2 in g.edges]
        edges.extend([[n2, n1] for n1, n2 in edges])
        edge_mat = torch.LongTensor(edges).T
        layer_edgeMat.append(edge_mat)

    return {
        "node_size": node_size,
        "interLayer_edgeMat": interLayer_edgeMat,
        "layer_edgeMat": layer_edgeMat,
    }


def sub_load_attribute_coding_tree(add_knn_edge_index, x, tree_depth):
    G = nx.Graph()
    G.add_nodes_from(range(x.size(0)))
    G.add_edges_from(add_knn_edge_index.cpu().numpy().T)

    T = trans_graph_tree(G, tree_depth)
    layer_data = extract_layer_data(T, G, tree_depth)

    return layer_data


def sub_knn_edge_index(edge_index, x):
    edge_index = to_undirected(edge_index)
    k = knn_maxE1(x, device, edge_index.transpose(0, 1))
    add_knn_edge_index = add_knn(k, x, device, edge_index.transpose(0, 1))
    add_knn_edge_index = add_knn_edge_index.transpose(0, 1)

    return add_knn_edge_index


def load_attribute_coding_tree(dataname, tree_depth=2):
    data_path = PWD + "/mid"
    save_path = PWD + "/trees"
    if os.path.exists(save_path + "/%s_%s.pickle" % (dataname, tree_depth)):
        print("%s_%s.pickle" % (dataname, tree_depth) + " existed!")
        return
    graphs_attribute_feature = torch.load(data_path + "/attribute_nodes_feature.pt").to(
        device
    )
    graphs_edge_index = torch.load(data_path + "/edge_index.pt").to(device)
    print(f"attribute_feature shape: {graphs_attribute_feature.shape}")
    print(f"edge_index shape: {graphs_edge_index.shape}")
    knn_graphs_edge_index_path = data_path + "/knn_graphs_edge_index.pickle"
    knn_graphs_edge_index = []
    if os.path.exists(knn_graphs_edge_index_path):
        with open(knn_graphs_edge_index_path, "rb") as fp:
            knn_graphs_edge_index = pickle.load(fp)
    else:
        print("generating knn_graphs_edge_index.pickle")
        knn_graphs_edge_index = Parallel(n_jobs=8)(
            delayed(sub_knn_edge_index)(
                graphs_edge_index[i], graphs_attribute_feature[i]
            )
            for i in tqdm(range(graphs_edge_index.shape[0]))
        )
        print("knn_graphs_edge_index: " + str(len(knn_graphs_edge_index)))
        with open(knn_graphs_edge_index_path, "wb") as fp2:
            pickle.dump(knn_graphs_edge_index, fp2)
        print("generating knn_graphs_edge_index.pickle finished")

    graphs_layer_data = []
    print(dataname, tree_depth)
    for i in tqdm(range(graphs_edge_index.shape[0])):
        graphs_layer_data.append(
            sub_load_attribute_coding_tree(
                knn_graphs_edge_index[i], graphs_attribute_feature[i], tree_depth
            )
        )

    print("graphs_layer_data: " + str(len(graphs_layer_data)))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print("tree_depth-", tree_depth, "len(data)-", len(graphs_layer_data))
    with open(save_path + "/%s_%s.pickle" % (dataname, tree_depth), "wb") as fp:
        pickle.dump(graphs_layer_data, fp)
    print("/%s_%s.pickle" % (dataname, tree_depth) + "created!")
