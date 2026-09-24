import torch
import dgl


def add_knn(k, node_embed, device, edge_index=None):
    if device == torch.device("cpu"):
        knn_g = dgl.knn_graph(node_embed, k, algorithm="bruteforce", dist="cosine")
    else:
        knn_g = dgl.knn_graph(
            node_embed, k, algorithm="bruteforce-sharemem", dist="cosine"
        )
    knn_g = dgl.add_reverse_edges(knn_g)
    knn_edge_index = knn_g.edges()
    knn_edge_index = torch.cat(
        (knn_edge_index[0].reshape(1, -1), knn_edge_index[1].reshape(1, -1)), dim=0
    )
    knn_edge_index = knn_edge_index.t()
    if edge_index is not None:
        edge_index_2 = torch.concat((edge_index, knn_edge_index), dim=0)
    else:
        edge_index_2 = knn_edge_index
    edge_index_2 = torch.unique(edge_index_2, dim=0)

    return edge_index_2


def calc_e1_sparse(node_num, edge_index, weight):
    mask = edge_index[:, 0] != edge_index[:, 1]
    edge_index = edge_index[mask]
    weight = weight[mask]

    degree = torch.zeros(node_num, device=edge_index.device)
    degree = degree.scatter_add(0, edge_index[:, 0], weight)
    degree = degree.scatter_add(0, edge_index[:, 1], weight)

    vol = degree.sum()
    idx = degree.nonzero().reshape(-1)
    g = degree[idx]
    return -((g / vol) * torch.log2(g / vol)).sum()


def calc_e1(adj: torch.Tensor):
    adj = adj - torch.diag_embed(torch.diag(adj))
    degree = adj.sum(dim=1)
    vol = adj.sum()
    idx = degree.nonzero().reshape(-1)
    g = degree[idx]
    return -((g / vol) * torch.log2(g / vol)).sum()


def get_adj_matrix(node_num, edge_index, weight) -> torch.Tensor:
    adj_matrix = torch.zeros((node_num, node_num))
    adj_matrix[edge_index.t()[0], edge_index.t()[1]] = weight
    adj_matrix = adj_matrix - torch.diag_embed(torch.diag(adj_matrix))
    return adj_matrix


def get_weight_fast(node_embedding, edge_index):
    x_i = node_embedding[edge_index[:, 0]]
    x_j = node_embedding[edge_index[:, 1]]

    x_i_mean = x_i.mean(dim=1, keepdim=True)
    x_j_mean = x_j.mean(dim=1, keepdim=True)

    x_i_centered = x_i - x_i_mean
    x_j_centered = x_j - x_j_mean

    numerator = (x_i_centered * x_j_centered).sum(dim=1)
    denominator = (x_i_centered**2).sum(dim=1).sqrt() * (x_j_centered**2).sum(
        dim=1
    ).sqrt()
    corr = numerator / (denominator + 1e-8)
    corr[torch.isnan(corr)] = 0
    weight = corr + 1
    M = weight.mean() / (2 * node_embedding.shape[0])
    weight = weight + M
    return weight


def get_weight(node_embedding, edge_index):
    """
    计算边的权重，使用皮尔逊相关系数。
    """
    node_num = node_embedding.shape[0]
    links = node_embedding[edge_index]
    weight = []
    for i in range(links.shape[0]):
        weight.append(torch.corrcoef(links[i])[0, 1])
    weight = torch.tensor(weight) + 1
    weight[torch.isnan(weight)] = 0
    M = weight.mean() / (2 * node_num)
    weight = weight + M
    return weight


def knn_maxE1(node_embedding: torch.Tensor, device, edge_index=None):
    USE_OLD = False
    old_e1 = 0
    node_num = node_embedding.shape[0]
    k = 1
    while k < 50:
        edge_index_k = add_knn(k, node_embedding, device, edge_index)
        if USE_OLD:
            weight = get_weight(node_embedding, edge_index_k)
            adj = get_adj_matrix(node_num, edge_index_k, weight)
            e1 = calc_e1(adj)
        else:
            weight = get_weight_fast(node_embedding, edge_index_k)
            e1 = calc_e1_sparse(node_num, edge_index_k, weight)

        if e1 - old_e1 > 0.1:
            k += 5
        elif e1 - old_e1 > 0.01:
            k += 3
        elif e1 - old_e1 > 0.001:
            k += 1
        else:
            break
        old_e1 = e1
    return k


if __name__ == "__main__":
    cat_tensor = torch.load("BAF_pt/baf_cat_properties_tensor.pt")
    num_tensor = torch.load("BAF_pt/baf_num_properties_tensor.pt")

    node_embedding = torch.cat((cat_tensor, num_tensor), dim=1)

    knn_maxE1(
        node_embedding, torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
