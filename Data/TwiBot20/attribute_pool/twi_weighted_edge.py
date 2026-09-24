import dgl
import pandas as pd
from collections import Counter
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
import numpy as np
import torch
from typing import Literal
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class WeightedEdge:
    def __init__(self, graph):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.graph: dgl.DGLGraph = graph.to(self.device)

    def __num_to_weight(
        self, edge_nums: list, method: Literal["power", "log"] = "power"
    ) -> list:
        if method == "log":
            weights = np.log1p(np.array(edge_nums))
        elif method == "power":
            scaler = PowerTransformer(method="box-cox")
            weights = np.array(edge_nums).reshape(-1, 1)
            weights = scaler.fit_transform(weights).flatten()
            weights = weights + -weights.min()
        else:
            raise ValueError("Invalid method. Use 'power' or 'log'.")

        scaler = MinMaxScaler()
        weights = scaler.fit_transform(weights.reshape(-1, 1)).flatten()
        weights = weights + 1e-4

        return weights.tolist()

    def run(self):
        new_edges = {}
        edge_weights_power = {}
        edge_weights_log = {}

        for etype in self.graph.canonical_etypes:
            src, dst = self.graph.edges(etype=etype)
            edge_counts = Counter(zip(src.tolist(), dst.tolist()))

            unique_src, unique_dst, weights = [], [], []

            for (u, v), count in edge_counts.items():
                unique_src.append(u)
                unique_dst.append(v)
                weights.append(count)

            weights_power = self.__num_to_weight(weights, method="power")
            weights_log = self.__num_to_weight(weights, method="log")

            new_edges[etype] = (unique_src, unique_dst)

            edge_weights_power[etype] = weights_power
            edge_weights_log[etype] = weights_log

            print(f"Edge type: {etype}")

        new_graph = dgl.heterograph(new_edges).to(self.device)

        for etype in self.graph.canonical_etypes:
            new_graph.edges[etype].data["weight_power"] = torch.tensor(
                edge_weights_power[etype], dtype=torch.float32
            ).to(self.device)
            new_graph.edges[etype].data["weight_log"] = torch.tensor(
                edge_weights_log[etype], dtype=torch.float32
            ).to(self.device)

        for ntype in self.graph.ntypes:
            new_graph.nodes[ntype].data.update(self.graph.nodes[ntype].data)

        dgl.save_graphs(os.path.join(BASE_DIR, "twi_weighted_graph.dgl"), new_graph)
        print("Weighted graph saved to 'twi_weighted_graph.dgl'.")


if __name__ == "__main__":
    g = dgl.load_graphs("twi_Data_hetero_graph.dgl")[0][0]

    weighted_edge = WeightedEdge(g)

    weighted_edge.run()
