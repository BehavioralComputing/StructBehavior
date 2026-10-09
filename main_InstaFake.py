from attribute_tree.sep_g import SEP_G
from backbone.rgcn import FACNConv as RGCNConv
from backbone.relation import SharedRelationHeteroConv
from backbone.training import training_selection
from backbone.fusion import (
    AdaptiveGranularityFusion,
    GlobalLearnableFusion,
    HybridGranularityFusion,
)
from backbone.utils_experiment import set_model_seed, summarize_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import os
from torch_geometric.data import Data
from torch_geometric.nn import APPNP
import argparse
import pickle
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
import dgl
import dgl.nn.pytorch as dglnn
import dgl.function as fn
import pandas as pd

PWD = os.path.dirname(os.path.realpath(__file__))


def edge_mask(edge_index, edge_attr, pe):
    edge_index = edge_index.clone()
    edge_num = edge_index.shape[1]
    pre_index = torch.bernoulli(torch.ones(edge_num) * pe) == 0
    pre_index.to(edge_index.device)
    edge_index = edge_index[:, pre_index]
    edge_attr = edge_attr.clone()
    edge_attr = edge_attr[pre_index]
    return edge_index, edge_attr


def relational_undirected(edge_index, edge_type):
    device = edge_index.device
    relation_num = edge_type.max() + 1
    edge_index = edge_index.clone()
    edge_type = edge_type.clone()
    r_edge = []
    for i in range(relation_num):
        e1 = edge_index[:, edge_type == i].unique(dim=1)
        e2 = e1.flip(0)
        edges = torch.cat((e1, e2), dim=1)
        r_edge.append(edges)
    edge_type = torch.cat(
        [torch.tensor([i] * e.shape[1]) for i, e in enumerate(r_edge)], dim=0
    ).to(device)
    edge_index = torch.cat(r_edge, dim=1)

    return edge_index, edge_type


class Attribute_Layer(nn.Module):
    def __init__(self, args):
        super(Attribute_Layer, self).__init__()
        self.args = args
        self.sep_g = SEP_G(self.args)

    def forward(self, data):
        x = self.sep_g(data)
        return x


class RGCN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.rel_names = args.rel_names
        self.conv1 = SharedRelationHeteroConv(
            {
                rel: dglnn.GraphConv(args.input_dim, args.input_dim)
                for rel in self.rel_names
            },
            aggregate="sum",
        )
        self.conv2 = SharedRelationHeteroConv(
            {
                rel: dglnn.GraphConv(args.input_dim, args.input_dim)
                for rel in self.rel_names
            },
            aggregate="sum",
        )

    def forward(self, graph, node_feats, return_layers=False):
        edge_weight_dict = {
            rel: graph.edges[rel].data["weight_power"] for rel in self.rel_names
        }

        h1 = self.conv1(
            graph,
            node_feats,
            mod_kwargs={
                rel: {"edge_weight": edge_weight_dict[rel]} for rel in self.rel_names
            },
        )
        h1 = {k: F.relu(v) for k, v in h1.items()}

        h2 = self.conv2(
            graph,
            h1,
            mod_kwargs={
                rel: {"edge_weight": edge_weight_dict[rel]} for rel in self.rel_names
            },
        )
        if return_layers:
            return [h1, h2]
        return h2


class RGATLite(nn.Module):
    """RGAT-lite: relation-aware GAT for molecular layer.

    2 layers, 2 heads, residual=True, LayerNorm=True, dropout=0.5.
    Relation awareness: separate GATConv per edge type in HeteroGraphConv.
    """

    def __init__(self, args):
        super().__init__()
        self.rel_names = args.rel_names
        in_dim = args.input_dim
        num_heads = 2
        head_dim = in_dim // num_heads

        self.conv1 = SharedRelationHeteroConv(
            {
                rel: dglnn.GATConv(
                    in_dim, head_dim, num_heads, feat_drop=0.5, attn_drop=0.5
                )
                for rel in self.rel_names
            },
            aggregate="sum",
        )
        self.norm1 = nn.LayerNorm(in_dim)

        self.conv2 = SharedRelationHeteroConv(
            {
                rel: dglnn.GATConv(
                    in_dim, head_dim, num_heads, feat_drop=0.5, attn_drop=0.5
                )
                for rel in self.rel_names
            },
            aggregate="sum",
        )
        self.norm2 = nn.LayerNorm(in_dim)

        self.dropout = nn.Dropout(0.5)

    def forward(self, graph, node_feats, return_layers=False):
        h1 = self.conv1(graph, node_feats)

        h1 = {k: v.reshape(v.shape[0], -1) for k, v in h1.items()}
        h1 = {k: self.norm1(F.relu(v + node_feats[k])) for k, v in h1.items()}
        h1 = {k: self.dropout(v) for k, v in h1.items()}

        h2 = self.conv2(graph, h1)
        h2 = {k: v.reshape(v.shape[0], -1) for k, v in h2.items()}
        h2 = {k: self.norm2(F.relu(v + h1[k])) for k, v in h2.items()}

        if return_layers:
            return [h1, h2]
        return h2


class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(args.input_dim, args.input_dim),
            nn.ReLU(),
            nn.Linear(args.input_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
        )

    def forward(self, x):
        return self.layers(x)


class BotRGCN(nn.Module):
    def __init__(self, args):
        super(BotRGCN, self).__init__()
        self.num_prop_size = args.prop_num
        self.cat_prop_size = args.cat_num
        self.dropout = args.dropout
        self.node_num = args.node_num
        self.pe = args.pe
        self.pf = args.pf
        input_dimension = args.input_dim
        embedding_dimension = args.hidden_dim

        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(self.num_prop_size, int(input_dimension / 2)), nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(self.cat_prop_size, int(input_dimension / 2)), nn.LeakyReLU()
        )

        self.linear_relu_input = nn.Sequential(
            nn.Linear(input_dimension, embedding_dimension),
            nn.PReLU(embedding_dimension),
        )

        self.rgcn1 = RGCNConv(
            embedding_dimension, embedding_dimension, num_relations=args.num_relations
        )
        self.rgcn2 = RGCNConv(
            embedding_dimension, embedding_dimension, num_relations=args.num_relations
        )

        self.material_jk = getattr(args, "material_jk", "true") == "true"
        self.material_appnp = getattr(args, "material_appnp", "false") == "true"
        if self.material_appnp:
            self.appnp = APPNP(K=5, alpha=0.2, dropout=0.0)

        if self.material_jk:
            self.out_proj = nn.Sequential(
                nn.Linear(2 * embedding_dimension, embedding_dimension), nn.LeakyReLU()
            )
        else:
            self.out_proj = nn.Sequential(
                nn.Linear(embedding_dimension, embedding_dimension), nn.LeakyReLU()
            )

        self.relu = nn.LeakyReLU()

    def forward(self, data, return_attention=False):
        x = data.x
        edge_index = data.edge_index
        edge_type = data.edge_type
        edge_index_orig = edge_index.clone()

        if self.training:
            edge_index, edge_type = edge_mask(edge_index, edge_type, self.pe)

        num_prop = x[:, : self.num_prop_size]
        cat_prop = x[:, self.num_prop_size : self.num_prop_size + self.cat_prop_size]
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        x = torch.cat((n, c), dim=1)

        x = self.linear_relu_input(x)
        if self.material_jk:
            h1 = self.rgcn1(x, edge_index, edge_type, return_attention)
            h1 = F.dropout(h1, p=self.dropout, training=self.training)
            h2 = self.rgcn2(h1, edge_index, edge_type)
            h2 = F.dropout(h2, p=self.dropout, training=self.training)
            h_jk = torch.cat([h1, h2], dim=-1)
            x = self.out_proj(h_jk)
        else:
            x = self.rgcn1(x, edge_index, edge_type, return_attention)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.rgcn2(x, edge_index, edge_type)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.out_proj(x)
        if self.material_appnp:
            x = self.appnp(x, edge_index_orig)
        return x


class AttributePool(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_linear = nn.Linear(args.feature_num, args.input_dim)
        mol_enc = getattr(args, "molecular_encoder", "rgcn")
        if mol_enc == "rgat_lite":
            self.rgcn = RGATLite(args)
        else:
            self.rgcn = RGCN(args)
        self.mlp = MLP(args)
        self.device = args.device
        self.hidden_dim = args.input_dim
        self.map_data = args.map_data

        self.molecular_jk = getattr(args, "molecular_jk", "true") == "true"
        if self.molecular_jk:
            self.out_proj = nn.Sequential(
                nn.Linear(2 * args.input_dim, args.input_dim), nn.ReLU()
            )

    def _pool_layer(self, h_dict):
        """Pool a dict[str, Tensor] -> [node_num, hidden_dim] via map_data."""
        columns = list(self.map_data.columns)
        index_matrix = torch.full((args.node_num, len(columns)), -1, dtype=torch.long)
        for i, col in enumerate(columns):
            col_values = self.map_data[col].values[: args.node_num]
            index_matrix[:, i] = torch.tensor(
                [int(v) if pd.notna(v) else -1 for v in col_values], dtype=torch.long
            )

        all_feat_sum = torch.zeros(args.node_num, self.hidden_dim, device=self.device)
        valid_counts = torch.zeros(args.node_num, device=self.device)
        for i, col in enumerate(columns):
            idx = index_matrix[:, i]
            valid_mask = idx != -1
            selected_feats = h_dict[col][idx.clamp(min=0)]
            selected_feats[~valid_mask] = 0
            all_feat_sum += selected_feats
            valid_counts += valid_mask.float().to(self.device)

        valid_counts[valid_counts == 0] = 1
        return all_feat_sum / valid_counts.unsqueeze(1)

    def forward(self, g: dgl.DGLGraph):
        h = g.ndata["feat"]
        h = {k: F.relu(self.input_linear(v)) for k, v in h.items()}

        if self.molecular_jk:
            h_list = self.rgcn(g, h, return_layers=True)
            pooled = [self._pool_layer(layer_h) for layer_h in h_list]
            h_jk = torch.cat(pooled, dim=-1)
            return self.mlp(self.out_proj(h_jk))
        else:
            h = self.rgcn(g, h)
            all_mean_feat = self._pool_layer(h)
            return self.mlp(all_mean_feat)


class StructBehavior(nn.Module):
    def __init__(self, args, idx):
        super(StructBehavior, self).__init__()
        self.args = args
        self.idx = idx

        self.attribute_layer = Attribute_Layer(self.args)

        self.attribute_pool_layer = AttributePool(self.args)

        self.backbone = BotRGCN(self.args)

        if self.args.fusion == "adaptive":
            use_ln = getattr(self.args, "gate_use_layernorm", True)
            self.fusion = AdaptiveGranularityFusion(
                hidden_dim=self.args.hidden_dim,
                dropout=self.args.dropout,
                gate_scale=self.args.gate_scale,
                use_layernorm=use_ln,
            )
        elif self.args.fusion == "global_learnable":
            self.fusion = GlobalLearnableFusion(
                hidden_dim=self.args.hidden_dim,
                gate_scale=self.args.gate_scale,
            )
        elif self.args.fusion == "hybrid":
            use_ln = getattr(self.args, "gate_use_layernorm", True)
            self.fusion = HybridGranularityFusion(
                hidden_dim=self.args.hidden_dim,
                dropout=self.args.dropout,
                gate_scale=self.args.gate_scale,
                use_layernorm=use_ln,
                hybrid_eta=self.args.hybrid_eta,
                learnable_eta=self.args.learnable_eta,
                eta_init=self.args.eta_init,
            )

        classifier_in = self.args.hidden_dim * 3
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, self.args.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.args.hidden_dim, self.args.num_classes),
        )
        # The two alignment paths each have independent endpoint heads.
        self.proj_ra_material = nn.Sequential(
            nn.Linear(self.args.hidden_dim, self.args.proj_dim),
            nn.LeakyReLU(),
            nn.Linear(self.args.proj_dim, self.args.hidden_dim),
        )
        self.proj_ra_molecular = nn.Sequential(
            nn.Linear(self.args.hidden_dim, self.args.proj_dim),
            nn.LeakyReLU(),
            nn.Linear(self.args.proj_dim, self.args.hidden_dim),
        )
        self.proj_fsa_molecular = nn.Sequential(
            nn.Linear(self.args.hidden_dim, self.args.proj_dim),
            nn.LeakyReLU(),
            nn.Linear(self.args.proj_dim, self.args.hidden_dim),
        )
        self.proj_fsa_atomic = nn.Sequential(
            nn.Linear(self.args.hidden_dim, self.args.proj_dim),
            nn.LeakyReLU(),
            nn.Linear(self.args.proj_dim, self.args.hidden_dim),
        )
        self.test_results = []
        self.granularity_weights = []

    def infonce_loss(self, emb_i, emb_j, temperature=0.1):
        """Symmetric cross-view InfoNCE, matching the manuscript objective."""
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)
        logits = torch.mm(z_i, z_j.t()) / temperature
        targets = torch.arange(logits.size(0), device=logits.device)
        return 0.5 * (
            F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets)
        )

    def forward(self, batch, instance_idx=None, split="val"):
        # Shared graph propagation may use unlabeled transductive context.
        h_material = self.backbone(batch["data"])
        h_material = h_material[: self.args.node_num, :]
        h_molecular = self.attribute_pool_layer(batch["graph"])
        h_atomic = self.attribute_layer(batch)

        if self.training:
            selected_idx, selected_labels = training_selection(
                self.idx, instance_idx, h_atomic.device, self.args.node_num
            )
        else:
            if split not in ("val", "test"):
                raise ValueError("Evaluation split must be 'val' or 'test'.")
            selected_idx = torch.as_tensor(
                self.idx[f"{split}_idx"], dtype=torch.long, device=h_atomic.device
            )
            selected_labels = self.idx[f"{split}_label"].to(h_atomic.device)

        # Align and classify only the selected instances, with the same order
        # on all three views and the corresponding supervised labels.
        h_atomic = h_atomic.index_select(0, selected_idx)
        h_molecular = h_molecular.index_select(0, selected_idx)
        h_material = h_material.index_select(0, selected_idx)
        gw = None
        if self.args.fusion == "concat":
            fused_feat = torch.cat([h_atomic, h_molecular, h_material], dim=1)
        else:
            fused_feat, gw = self.fusion(
                h_atomic, h_molecular, h_material, return_weights=True
            )
        logits = self.classifier(fused_feat)
        loss_ce = F.cross_entropy(logits, selected_labels)

        if self.training:
            # Positives and negatives are confined to the training mini-batch.
            loss_ra = self.infonce_loss(
                self.proj_ra_material(h_material),
                self.proj_ra_molecular(h_molecular),
                self.args.temperature,
            )
            loss_fsa = self.infonce_loss(
                self.proj_fsa_molecular(h_molecular),
                self.proj_fsa_atomic(h_atomic),
                self.args.temperature,
            )
            loss = (
                loss_ce
                + self.args.lambda_ra * loss_ra
                + self.args.lambda_fsa * loss_fsa
            )
            if gw is not None and self.args.lambda_gate > 0:
                loss = loss + self.args.lambda_gate * ((gw - 1.0) ** 2).mean()
            return loss

        labels_np = selected_labels.detach().cpu().numpy()
        pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
        metrics = {
            "acc": accuracy_score(labels_np, pred),
            "loss": loss_ce.item(),
            "f1": f1_score(labels_np, pred, average="binary", zero_division=0),
            "recall": recall_score(labels_np, pred, average="binary", zero_division=0),
            "precision": precision_score(labels_np, pred, average="binary", zero_division=0),
        }
        if split == "test":
            if gw is not None:
                self.granularity_weights.append(gw.detach().cpu())
            self.test_results.append(
                [metrics["acc"], metrics["f1"], metrics["recall"], metrics["precision"]]
            )
        return metrics

    def get_test_results(self):
        """Results from final evaluation after validation checkpoint selection."""
        return self.test_results


class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()

        set_model_seed(args.model_seed)
        self.args = args
        self.load_data()
        self.model = StructBehavior(self.args, self.idx).to(self.args.device)
        self.save_top_k = args.save_top_k
        self.patience = 0
        self.best_loss_epoch = 0
        self.best_acc_epoch = 0
        self.best_loss = 1e9
        self.best_loss_acc = -1e9
        self.best_acc = -1e9
        self.best_acc_loss = 1e9
        self.best_val_f1 = -1e9
        self.best_val_f1_epoch = 0
        self.test_results = []

    def load_data(self):

        attri_raw_path = os.path.join(PWD, "Data/InstaFake/attribute_tree/mid")
        tree_path = os.path.join(
            PWD,
            "Data/InstaFake/attribute_tree/trees",
            "%s_%s.pickle" % (self.args.dataset, self.args.tree_depth),
        )

        with open(
            os.path.join(attri_raw_path, "knn_graphs_edge_index.pickle"), "rb"
        ) as fp2:
            self.knn_edge_index_list = pickle.load(fp2)

        with open(tree_path, "rb") as fp1:
            self.layer_data = pickle.load(fp1)
        self.args.num_features = self.args.hidden_dim

        self.x_list = torch.load(
            os.path.join(attri_raw_path, "attribute_nodes_feature.pt")
        )

        attribute_pool_graph_path = os.path.join(
            PWD, "Data/InstaFake/attribute_pool/insta_weighted_graph.dgl"
        )
        loaded_graphs, _ = dgl.load_graphs(attribute_pool_graph_path)
        loaded_graphs = loaded_graphs[0]
        self.graph = loaded_graphs.to(self.args.device)
        args.rel_names = self.graph.etypes

        map_data_path = os.path.join(
            PWD, "Data/InstaFake/attribute_pool/insta_data_df.csv"
        )
        map_data = pd.read_csv(map_data_path)
        args.map_data = map_data

        whole_raw_path = os.path.join(PWD, "Data/InstaFake/material_data")
        label = torch.load(os.path.join(whole_raw_path, "insta_label.pt"))
        x = torch.cat(
            [
                torch.load(
                    os.path.join(whole_raw_path, "insta_num_properties_tensor.pt")
                ),
                torch.load(
                    os.path.join(whole_raw_path, "insta_cat_properties_tensor.pt")
                ),
            ],
            dim=1,
        )

        edge_index = torch.load(os.path.join(whole_raw_path, "insta_edge_index.pt"))
        edge_type = torch.load(os.path.join(whole_raw_path, "insta_edge_type.pt"))

        edge_index, edge_type = relational_undirected(edge_index, edge_type)
        self.args.num_relations = edge_type.max() + 1
        data = Data(x=x, edge_index=edge_index, edge_type=edge_type, y=label).to(
            self.args.device
        )
        self.data = data

        train_val_test_idx = np.arange(self.args.node_num)
        train_idx, test_idx = train_test_split(
            train_val_test_idx, test_size=0.3, random_state=self.args.split_seed
        )
        val_idx, test_idx = train_test_split(
            test_idx, test_size=0.3333, random_state=self.args.split_seed
        )

        train_label = label[train_idx].to(self.args.device)
        val_label = label[val_idx].to(self.args.device)
        test_label = label[test_idx].to(self.args.device)

        self.idx = {
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "train_label": train_label,
            "val_label": val_label,
            "test_label": test_label,
        }
        summarize_split(label, train_idx, val_idx, test_idx)

    def train(self):
        self.args.batch_size = getattr(self.args, "batch_size", 256)
        if self.args.epochs < 1 or self.args.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive.")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=16, eta_min=0
        )
        batch = {
            "data": self.data.to(self.args.device),
            "graph": self.graph,
            "x_list": self.x_list.to(self.args.device),
            "knn_edge_index_list": self.knn_edge_index_list,
            "layer_data": self.layer_data,
        }
        train_idx = torch.as_tensor(
            self.idx["train_idx"], dtype=torch.long, device=self.args.device
        )
        if train_idx.numel() == 0:
            raise ValueError("The training split is empty.")
        primary_metric = "f1"
        best_score = float("-inf")
        best_state = None
        self.validation_history = []
        print(
            f"Training with lambda_ra={self.args.lambda_ra}, "
            f"lambda_fsa={self.args.lambda_fsa}, lr={self.args.lr}, "
            f"batch_size={self.args.batch_size}"
        )

        for epoch in range(self.args.epochs):
            self.model.train()
            order = torch.randperm(train_idx.numel(), device=train_idx.device)
            shuffled_idx = train_idx[order]
            total_loss = 0.0
            for selected_idx in shuffled_idx.split(self.args.batch_size):
                self.optimizer.zero_grad()
                loss = self.model(batch, instance_idx=selected_idx)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * selected_idx.numel()
            self.scheduler.step()

            # Checkpoint selection and early stopping use validation only.
            val_metrics = self.eval(batch, split="val")
            self.validation_history.append({"epoch": epoch, **val_metrics})
            score = val_metrics[primary_metric]
            eta_str = ""
            if self.args.fusion == "hybrid":
                eta_str = ", eta: %.4f" % self.model.fusion.get_eta().item()
            print(
                f"epoch: {epoch}, train_loss: {total_loss / train_idx.numel():.4f}, "
                f"val_acc: {val_metrics['acc']:.4f}, val_loss: {val_metrics['loss']:.4f}, "
                f"val_{primary_metric}: {score:.4f}{eta_str}"
            )
            if score > best_score:
                best_score = score
                self.best_epoch = epoch
                # Clone tensors so later optimizer updates cannot mutate this checkpoint.
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in self.model.state_dict().items()
                }
                self.patience = 0
            else:
                self.patience += 1
            if self.patience > self.args.patience:
                break

        if best_state is None:
            raise RuntimeError("No finite validation metric was available for selection.")
        self.model.load_state_dict(best_state)
        self.best_validation_metric = best_score
        # Test is evaluated once, only after restoring the selected checkpoint.
        self.final_test_metrics = self.eval(batch, split="test")
        print(
            f"best_epoch: {self.best_epoch}, best_val_{primary_metric}: {best_score:.4f}, "
            + ", ".join(
                f"test_{name}: {value:.4f}"
                for name, value in self.final_test_metrics.items()
                if name != "loss"
            )
        )
        # Retain the legacy top-k option for validation summaries, never extra tests.
        top_k = min(self.save_top_k, len(self.validation_history))
        for item in sorted(
            self.validation_history, key=lambda row: row[primary_metric], reverse=True
        )[:top_k]:
            print(
                f"validation_rank epoch: {item['epoch']}, "
                f"val_{primary_metric}: {item[primary_metric]:.4f}"
            )
        return self.final_test_metrics[primary_metric]

    @torch.no_grad()
    def eval(self, batch, split="val"):
        self.model.eval()
        return self.model(batch, split=split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SEP")
    parser.add_argument("--dataset", type=str, default="insta")
    parser.add_argument("--node_num", type=int, default=1400)
    parser.add_argument("--attribute_num", type=int, default=14)
    parser.add_argument("--num_classes", type=int, default=2)

    parser.add_argument("--feature_num", type=int, default=768)
    parser.add_argument("--prop_num", type=int, default=12)
    parser.add_argument("--cat_num", type=int, default=2)

    parser.add_argument("--pe", type=float, default=0.2)
    parser.add_argument("--pf", type=float, default=0.2)

    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--tree_depth", type=int, default=3)
    parser.add_argument("--conv", type=str, default="GCN")
    parser.add_argument("--input_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--proj_dim", type=int, default=16)
    parser.add_argument(
        "--batch_size", type=int, default=256,
        help="Training instances per CE/FSA/RA mini-batch.",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--save_top_k", type=int, default=6)
    parser.add_argument(
        "--lambda_ra",
        "--alpha_1_2",
        dest="lambda_ra",
        type=float,
        default=0.12,
        help="Relational Alignment weight (material-molecular).",
    )
    parser.add_argument(
        "--lambda_fsa",
        "--alpha_2_3",
        dest="lambda_fsa",
        type=float,
        default=0.1,
        help="Fine-Structure Alignment weight (molecular-atomic).",
    )

    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument(
        "--model_seed",
        type=int,
        default=None,
        help="Seed for model initialization (falls back to --seed).",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=None,
        help="Seed for train/val/test split (falls back to --seed).",
    )
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=3e-3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument(
        "--material_jk",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Enable JK (Jumping Knowledge) in material BotRGCN: concat layer1+layer2 outputs",
    )
    parser.add_argument(
        "--material_appnp",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Enable APPNP post-propagation after BotRGCN",
    )
    parser.add_argument(
        "--atomic_jk",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Enable JK in atomic SEP_G: concat pooled_xs across tree depths",
    )
    parser.add_argument(
        "--molecular_jk",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Enable JK (Jumping Knowledge) in molecular RGCN (DGL): concat layer1+layer2 dict outputs",
    )
    parser.add_argument(
        "--molecular_encoder",
        type=str,
        default="rgcn",
        choices=["rgcn", "rgat_lite"],
        help="Encoder for molecular layer: rgcn (DGL GraphConv) or rgat_lite (DGL GATConv)",
    )
    parser.add_argument(
        "--fusion",
        type=str,
        default="hybrid",
        choices=["concat", "adaptive", "global_learnable", "hybrid"],
    )
    parser.add_argument("--gate_scale", type=float, default=0.5)
    parser.add_argument("--lambda_gate", type=float, default=0.0)
    parser.add_argument(
        "--gate_use_layernorm", action=argparse.BooleanOptionalAction, default=True
    )

    parser.add_argument(
        "--hybrid_eta",
        type=float,
        default=0.5,
        help="Fixed interpolation ratio between global and instance gates. Larger eta means more global.",
    )
    parser.add_argument(
        "--learnable_eta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a learnable eta instead of fixed hybrid_eta.",
    )
    parser.add_argument(
        "--eta_init",
        type=float,
        default=0.5,
        help="Initial eta when learnable_eta is enabled.",
    )
    parser.add_argument("--conv_dropout", type=float, default=0.5)
    parser.add_argument("--pooling_dropout", type=float, default=0.5)
    parser.add_argument(
        "-gp",
        "--global-pooling",
        type=str,
        default="average",
        choices=["sum", "average"],
        help="Pooling for over nodes: sum or average",
    )

    args = parser.parse_args()
    if args.model_seed is None:
        args.model_seed = args.seed
    if args.split_seed is None:
        args.split_seed = args.seed
    args.device = torch.device(
        "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
    )

    trainer = Trainer(args)
    test_f1 = trainer.train()
    print("test_f1: ", test_f1)
